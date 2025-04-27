# xd_train.py
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import os
import sys

from model import CLIPVAD, SingleModel, UncertaintyPredictor
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label, cosine_scheduler
from utils.CMA_MIL import CMAL
from utils.losses import UKDLoss
import xd_option

def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0]) # [B]
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))
    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def DISTILL(logits_target, logits_source, temperature):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    source_audio_student = F.log_softmax(logits_source/temperature, dim=1)
    target_visual_student = F.softmax(logits_target/temperature, dim=1)
    return kl_loss(source_audio_student, target_visual_student)

def AP2_DISTILL(logits_target, logits_source, temperature):
    """
    - logits_target: fine-logits [B, 256, 7]
    - logits_source: [B, 256]
    - feat_lengths: [B]
    """
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    source_audio_student = F.log_softmax(logits_source/temperature, dim=1) # [B, 256]
    target_visual_student = 1 - F.softmax(logits_target/temperature, dim=-1)[:, :, 0] # [B, 256]
    return kl_loss(source_audio_student, target_visual_student)

def train(av_model, v_model, train_loader, test_loader, args, label_map: dict, device):
    av_model.to(device)
    v_model.to(device)

    # 불확실성 예측 네트워크 및 UKD 손실 초기화
    uncertainty_predictor = UncertaintyPredictor(args.visual_width).to(device)
    ukd_loss_fn = UKDLoss()

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    # 옵티마이저 설정
    optimizer_av = torch.optim.AdamW(av_model.parameters(), lr=args.lr)
    optimizer_v = torch.optim.AdamW([
        {'params': v_model.parameters(), 'lr': args.v_lr},
        {'params': uncertainty_predictor.parameters(), 'lr': args.lr}
    ])
    
    scheduler_av = MultiStepLR(optimizer_av, args.scheduler_milestones, args.scheduler_rate)
    scheduler_v = MultiStepLR(optimizer_v, args.scheduler_milestones, args.scheduler_rate)
    
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0

    # 체크포인트 로딩 (있는 경우)
    if args.use_checkpoint == True and os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        av_model.load_state_dict(checkpoint['av_model_state_dict'])
        v_model.load_state_dict(checkpoint['v_model_state_dict'])
        if 'uncertainty_predictor_state_dict' in checkpoint:
            uncertainty_predictor.load_state_dict(checkpoint['uncertainty_predictor_state_dict'])
        optimizer_av.load_state_dict(checkpoint['optimizer_av_state_dict'])
        optimizer_v.load_state_dict(checkpoint['optimizer_v_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("Checkpoint info:")
        print("Epoch:", epoch+1, " AP:", ap_best)

    for e in range(epoch, args.max_epoch):
        av_model.train()
        v_model.train()
        uncertainty_predictor.train()
        
        loss_total1 = 0
        loss_total2 = 0
        loss_total3 = 0
        loss_total_cmal = 0
        loss_total_ukd = 0
        
        for i, item in enumerate(train_loader):
            step = i * train_loader.batch_size
            visual_feat, audio_feat, text_labels, feat_lengths = item
            visual_feat = visual_feat.to(device)
            audio_feat = audio_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)
            
            # ===== Stage 1: AV-모델 학습 =====
            optimizer_av.zero_grad()
            
            # AV forward (프롬프트·융합 포함)
            text_features, logits_av, logits2, v_logits, a_logits = av_model(
                visual_feat, audio_feat, None, prompt_text, feat_lengths, return_fused=False
            )

            # 분류 손실 계산
            loss1 = CLAS2(logits_av, text_labels, feat_lengths, device)  # Coarse
            loss_total1 += loss1.item()

            loss2 = CLASM(logits2, text_labels, feat_lengths, device)  # Fine
            loss_total2 += loss2.item()

            # 텍스트 특징 제약 손실
            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 6
            loss_total3 += loss3.item()

            # # Chain-each branches & BCE_Audio & Distill_av_to_v
            # loss4 = DISTILL(v_logits.squeeze(-1), a_logits.squeeze(-1), 3.0) + DISTILL(a_logits.squeeze(-1), v_logits.squeeze(-1), 3.0)
            # loss5 = DISTILL(logits_av, v_logits.squeeze(-1), 3.0)
            # loss6 = CLAS2(a_logits.squeeze(-1), text_labels, feat_lengths, device)
            # added_loss = loss5 + loss6 + loss4

            # CMAL 손실 계산
            visual_features, audio_features = av_model.encode_video(visual_feat, audio_feat, None, feat_lengths)
            loss_a2v_a2b, loss_a2v_a2n, loss_v2a_a2b, loss_v2a_a2n = CMAL(
                text_labels, 
                a_logits.squeeze(-1), 
                v_logits.squeeze(-1), 
                feat_lengths, 
                audio_features, 
                visual_features
            )
            
            # CMAL 손실 가중치 계산
            lamda_a2b = min(args.lamda_a2b, args.lamda_cof * e)
            lamda_a2n = min(args.lamda_a2n, args.lamda_cof * e)
            w_a2v = 1.5
            w_v2a = 1.0
            cmal_loss = lamda_a2b * (w_a2v * loss_a2v_a2b + w_v2a * loss_v2a_a2b) + \
                        lamda_a2n * (w_a2v * loss_a2v_a2n + w_v2a * loss_v2a_a2n)
            
            if isinstance(cmal_loss, torch.Tensor):
                loss_total_cmal += cmal_loss.item()
            else:
                loss_total_cmal += cmal_loss


            loss23 = 0.5 * (loss2 + loss3)

            # Stage 1 최종 손실 계산 및 역전파
            loss_av = loss1 + loss23 + cmal_loss * 0.2
            loss_av.backward()
            optimizer_av.step()
            
            # ===== Stage 2: 학생 모델 학습 (Uncertainty-KD) =====
            optimizer_v.zero_grad()
            
            # ① 교사 모델에서 fused 특성만 꺼내고 gradient 차단
            with torch.no_grad():
                _, _, _, _, _, fused_teacher = av_model(
                    visual_feat, audio_feat, None, prompt_text, feat_lengths, return_fused=True
                )
            teacher_feat = fused_teacher.detach()  # 교사 모델 파라미터 고정
            
            # ② 학생 모델(Visual-only) forward
            v_features, _ = v_model(visual_feat, None, feat_lengths)
            
            # ③ 불확실성 예측
            var_pred = uncertainty_predictor(v_features)  # σ²_i
            
            # ④ UKD 손실 계산 (Eq.11)
            ukd_loss = ukd_loss_fn(teacher_feat, v_features, var_pred)
            loss_total_ukd += ukd_loss.item()
            
            # Stage 2 역전파
            ukd_loss.backward()
            optimizer_v.step()
            
            # 로깅
            if step % 4800 == 0 and step != 0:
                print(f"Epoch {e+1}, Step {step}:")
                print("  AV Loss1: {:.4f}, AV Loss2: {:.4f}, AV Loss3: {:.4f}, CMAL Loss: {:.4f}, UKD Loss: {:.4f}".format(
                    loss_total1/(i+1), loss_total2/(i+1), loss_total3/(i+1), loss_total_cmal/(i+1), loss_total_ukd/(i+1)))
                
                # 중간 디버깅 평가
                print("  --> Running mid-epoch evaluation ...")
                auc, ap, mAP = test(av_model, v_model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
                sys.stdout.write(f"      [Mid-Epoch] AUC: {auc:.4f}, AP: {ap:.4f}, mAP: {mAP:.4f} \n")
                sys.stdout.flush()
                if ap > ap_best:
                    ap_best = ap
                    checkpoint = {
                        'epoch': e,
                        'av_model_state_dict': av_model.state_dict(),
                        'v_model_state_dict': v_model.state_dict(),
                        'uncertainty_predictor_state_dict': uncertainty_predictor.state_dict(),
                        'optimizer_av_state_dict': optimizer_av.state_dict(),
                        'optimizer_v_state_dict': optimizer_v.state_dict(),
                        'ap': ap_best
                    }
                    torch.save(checkpoint, args.checkpoint_path)
                    sys.stdout.write(f"Best model saved at Epoch {e+1}: New best AP = {ap_best:.4f} \n")

        scheduler_av.step()
        scheduler_v.step()
        
        # Epoch 종료 후 평가
        auc, ap, mAP = test(av_model, v_model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
        if ap > ap_best:
            ap_best = ap 
            checkpoint = {
                'epoch': e,
                'av_model_state_dict': av_model.state_dict(),
                'v_model_state_dict': v_model.state_dict(),
                'uncertainty_predictor_state_dict': uncertainty_predictor.state_dict(),
                'optimizer_av_state_dict': optimizer_av.state_dict(),
                'optimizer_v_state_dict': optimizer_v.state_dict(),
                'ap': ap_best
            }
            torch.save(checkpoint, args.checkpoint_path)
            sys.stdout.write(f"Best model saved at Epoch {e+1}: New best AP = {ap_best:.4f} \n")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = {
        'A': 'normal',
        'B1': 'fighting',
        'B2': 'shooting',
        'B4': 'riot',
        'B5': 'abuse',
        'B6': 'car accident',
        'G': 'explosion'
    }

    train_dataset = XDDataset(args.visual_length, args.train_list, args.audio_list, False, label_map)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = XDDataset(args.visual_length, args.test_list, args.test_audio_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    av_model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, 
                       args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, 
                       args.prompt_postfix, args.audio_dim, device)
    v_model = SingleModel(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                          args.visual_head, args.visual_layers, args.attn_window, device)
    
    train(av_model, v_model, train_loader, test_loader, args, label_map, device)
