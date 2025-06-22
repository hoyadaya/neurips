from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj
from torch.cuda.amp import autocast

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class CrossModalityAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        
    def forward(self, x1, x2):
        # x1 attends to x2
        q = self.query_proj(x1)
        k = self.key_proj(x2)
        v = self.value_proj(x2)
        
        x1_attended = self.ln(x1)
        x1_attended, _ = self.attn(q, k, v)
        x1_out = x1 + x1_attended
        x1_out = x1_out + self.mlp(self.ln(x1_out))
        return x1_out

class SingleModel(nn.Module):
    """Visual-only model for self-distillation"""
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 device):
        super().__init__()
        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.device = device

        # 수정: 입력이 이미 512차원임에 따라 512 → visual_width
        self.visual_dim_conv = nn.Linear(512, visual_width)
        
        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0
        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) 
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = F.threshold(tmp, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = F.threshold(tmp, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2
        return output
        
    def encode_video(self, images, padding_mask, lengths):
        # images는 이미 512차원 feature라고 가정
        images = self.visual_dim_conv(images.to(torch.float))
        
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)
        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))
        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))
        x = torch.cat((x1, x2), 2)
        x = self.linear(x)
        return x

    def forward(self, visual, padding_mask, lengths):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        logits1 = self.classifier(visual_features + self.mlp1(visual_features))
        return visual_features, logits1


### ------------ Current Model ------------------------
class CLIPVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 audio_dim: int,
                 device):
        super().__init__()
        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.audio_dim = audio_dim
        self.device = device

        # 수정: 입력이 512차원 visual feature라고 가정
        self.visual_dim_conv = nn.Linear(512, visual_width)
        self.audio_dim_conv = nn.Linear(audio_dim, visual_width)
        
        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        self.audio_temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )
        
        # LGT - Local Module
        self.cross_attn_v2a = CrossModalityAttention(visual_width, visual_head)
        self.cross_attn_a2v = CrossModalityAttention(visual_width, visual_head)

        # LGT - Global Module
        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        ###### ---------추가된 부분---------------------------------------------(ver1)
        self.gc1_a = GraphConvolution(visual_width, width, residual=True)
        self.gc2_a = GraphConvolution(width, width, residual=True)
        self.gc3_a = GraphConvolution(visual_width, width, residual=True)
        self.gc4_a = GraphConvolution(width, width, residual=True)
        self.disAdj_a = DistanceAdj()
        self.linear_a = nn.Linear(visual_width, visual_width)
        ###### ---------추가된 부분---------------------------------------------(ver1)



        ###### ---------추가된 부분---------------------------------------------
        # classifier? -> 경량화
        # visual prompt에서 사용된는 FFN
        self.mlp_t = nn.Sequential(OrderedDict([ 
            ("c_fc", nn.Linear(visual_width, visual_width * 4)), # visual_width * 4 => visual_width * 2
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        ###### ---------추가된 부분---------------------------------------------
        #init_w = 1.0
        #self.scale_v = nn.Parameter(torch.full([], float(init_w)))
        #self.scale_a = nn.Parameter(torch.full([], float(init_w)))
        
        p = 0.3 # 0.3: 원본 
        self.classifier = nn.Sequential(
            nn.Linear(visual_width, visual_width // 4),
            QuickGELU(),
            nn.Dropout(p),
            nn.Linear(visual_width // 4, 1),
        )
        self.audio_classifier =  nn.Sequential(
            nn.Linear(visual_width, visual_width // 4),
            QuickGELU(),
            nn.Dropout(p),
            nn.Linear(visual_width // 4, 1),
        )
        self.av_classifier = nn.Sequential(
            nn.Linear(visual_width * 2, visual_width),
            QuickGELU(),
            nn.Dropout(p),
            nn.Linear(visual_width, visual_width // 4),
            QuickGELU(),
            nn.Dropout(p),
            nn.Linear(visual_width // 4, 1),
        )
        ###### ---------추가된 부분---------------------------------------------

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.audio_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)
        nn.init.normal_(self.audio_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0
        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1))
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = F.threshold(tmp, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = F.threshold(tmp, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2
        return output

    def encode_video(self, images, audio, padding_mask, lengths):
        # Convert image features (expected to be 512-dim) 
        images = self.visual_dim_conv(images.to(torch.float))
        # Convert audio features (128-dim -> 512-dim)
        audio = self.audio_dim_conv(audio.to(torch.float))
        
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings
        
        audio_position_embeddings = self.audio_position_embeddings(position_ids)
        audio_position_embeddings = audio_position_embeddings.permute(1, 0, 2)
        audio = audio.permute(1, 0, 2) + audio_position_embeddings

        x_visual, _ = self.temporal((images, None))
        x_audio, _ = self.audio_temporal((audio, None))
        
        x_visual = x_visual.permute(1, 0, 2)
        x_audio = x_audio.permute(1, 0, 2) 

        # GCN processing for visual branch
        adj = self.adj4(x_visual, lengths)
        disadj = self.disAdj(x_visual.shape[0], x_visual.shape[1])
        x1_h = self.gelu(self.gc1(x_visual, adj))
        x2_h = self.gelu(self.gc3(x_visual, disadj))
        x1 = self.gelu(self.gc2(x1_h, adj)) # H_sim - # [B, 256, 256]
        x2 = self.gelu(self.gc4(x2_h, disadj)) # H_dis - # [B, 256, 256]
        x_visual = torch.cat((x1, x2), 2) # [B, 256, 512]
        
        #### -----------------추가된 부분------------------------------------(ver1)
        # GCN processing for audio branch
        adj_a = self.adj4(x_audio, lengths)
        disadj_a = self.disAdj(x_audio.shape[0], x_audio.shape[1])
        x1_ha = self.gelu(self.gc1(x_audio, adj_a))
        x2_ha = self.gelu(self.gc3(x_audio, disadj_a))
        x1_a = self.gelu(self.gc2(x1_ha, adj_a)) # H_sim - # [B, 256, 256]
        x2_a = self.gelu(self.gc4(x2_ha, disadj_a)) # H_dis - # [B, 256, 256]
        x_audio = torch.cat((x1_a, x2_a), 2) # [B, 256, 512]

        #### -----------------추가된 부분------------------------------------(ver1)
        # Process FC
        x_visual = self.linear(x_visual)
        x_audio = self.linear_a(x_audio)

        return x_visual, x_audio

    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat(len(text), 1, 1)
        text_tokens = torch.zeros(len(text), 77).to(self.device)
        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]
        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)
        return text_features

    def forward(self, visual, audio, padding_mask, text, lengths):
        visual_features, audio_features = self.encode_video(visual, audio, padding_mask, lengths)
        
        logits_visual = self.classifier(visual_features) # [B, 256, 1]
        logits_audio = self.audio_classifier(audio_features) # [B, 256, 1]
        
        logits1 = torch.maximum(logits_visual, logits_audio)
        text_features_ori = self.encode_textprompt(text) # [N, 512]
        text_features = text_features_ori

        # ----- Feature Fusion
        # 원본
        combined_features = torch.cat([visual_features, audio_features], dim=-1) # [B, 256, 1024]
        logits_av_3d = self.av_classifier(combined_features)  # 3차원 텐서 [batch_size, seq_len, 1] -> [B, 256, 1]
        logits_av = logits_av_3d.squeeze(-1)  # 2차원 텐서 [batch_size, seq_len] -> [B, 256]

        # ----- Aggregation -> [B, 1, 512]가 나와야 함.
        # 원본
        logits_attn = logits_av_3d.permute(0, 2, 1) # 원본 # [batch_size, 1, seq_len]
        #logits_attn = logits_visual.permute(0, 2, 1) # 수정
        visual_attn = logits_attn @ visual_features # [B, 1, 256] @ [B, 256, 512] = [B, 1, 512] => 비디오를 하나의 피쳐로 압축
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True) # [B, 1, 512]
        # 수정본

        # ----- Aggregation - expand
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2]) # [B, N, 512]
        text_features = text_features_ori.unsqueeze(0) # [1, N, 512]
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2]) # [B, N, 512]

        # Visual Prompt
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp_t(text_features) # [B, N, 512]
        
        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1) # [B, 512, N]
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07 # [B, 256, 512] @ [B, 512, N] = [B, 256, N] 
        return text_features_ori, logits1, logits2, logits_visual, logits_audio, logits_av

