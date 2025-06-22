import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import os
import sys
import wandb  # wandb 임포트 추가
wandb.login(key=os.environ["WANDB_API_KEY"])

from model import CLIPVAD, SingleModel
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label
from utils.CMA_MIL import CMAL  # 추가된 임포트
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

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer_av = torch.optim.AdamW(av_model.parameters(), lr=args.lr)
    optimizer_v = torch.optim.AdamW(v_model.parameters(), lr=args.v_lr)
    
    scheduler_av = MultiStepLR(optimizer_av, args.scheduler_milestones, args.scheduler_rate)
    scheduler_v = MultiStepLR(optimizer_v, args.scheduler_milestones, args.scheduler_rate)
    
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0

    if args.use_checkpoint == True and os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        av_model.load_state_dict(checkpoint['av_model_state_dict'])
        v_model.load_state_dict(checkpoint['v_model_state_dict'])
        optimizer_av.load_state_dict(checkpoint['optimizer_av_state_dict'])
        optimizer_v.load_state_dict(checkpoint['optimizer_v_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("Checkpoint info:")
        print("Epoch:", epoch+1, " AP:", ap_best)

    for e in range(epoch, args.max_epoch):
        av_model.train()
        v_model.train()
        loss_total1 = 0
        loss_total2 = 0
        loss_total3 = 0
        loss_total_cmal = 0
        
        # WandB 로깅: 에포크 시작
        wandb.log({"epoch": e+1})
        
        for i, item in enumerate(train_loader):
            step = i * train_loader.batch_size
            visual_feat, audio_feat, text_labels, feat_lengths = item
            visual_feat = visual_feat.to(device)
            audio_feat = audio_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            # text_labels는 dataset에서 그대로 사용 (get_prompt_text로 생성한 prompt_text와 함께)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)
            
            # Audio-visual 모델 forward
            text_features, logtis1, logits2, v_logits, a_logits, logits_av = av_model(
                visual_feat, audio_feat, None, prompt_text, feat_lengths)


            # 수정: 기존 logits1 대신 logits_av를 사용하여 분류 loss 계산
            loss1 = CLAS2(logits_av, text_labels, feat_lengths, device) # Coarse
            loss_total1 += loss1.item()

            loss2 = CLASM(logits2, text_labels, feat_lengths, device) # Fine
            loss_total2 += loss2.item()

            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True) # [1, 512]
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True) # [1, 512]
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 6
            loss_total3 += loss3.item()

            # Chain-each branches & BCE_Audio & Distill_av_to_v
            loss4 = DISTILL(v_logits.squeeze(-1), a_logits.squeeze(-1), 3.0) + DISTILL(a_logits.squeeze(-1), v_logits.squeeze(-1), 3.0) # Chain-each branches
            loss5 = DISTILL(logits_av, v_logits.squeeze(-1), 3.0) # Distill_av_to_v
            loss6 = CLAS2(a_logits.squeeze(-1), text_labels, feat_lengths, device) # BCE_Audio

            added_loss = loss5 + loss6 + loss4

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
            
            # CMAL 손실 합산 (가중치는 실험에 따라 조정)
            cmal_loss = (loss_a2v_a2b + loss_a2v_a2n + loss_v2a_a2b + loss_v2a_a2n) * 0.25 
            
            # 수정: cmal_loss가 tensor인지 float인지 확인하고 처리
            if isinstance(cmal_loss, torch.Tensor):
                loss_total_cmal += cmal_loss.item()
            else:
                loss_total_cmal += cmal_loss  # float인 경우 직접 더함
            
            # 최종 손실 계산 (CMAL 손실 추가)
            loss_av = loss1 + loss2 + loss3 * 1e-4 + cmal_loss + added_loss
            
            optimizer_v.zero_grad()
            optimizer_v.step()
            
            optimizer_av.zero_grad()
            loss_av.backward()
            optimizer_av.step()
            
            if step % 4800 == 0 and step != 0:
                print(f"Epoch {e+1}, Step {step}:")
                print("  AV Loss1: {:.4f}, AV Loss2: {:.4f}, AV Loss3: {:.4f}, CMAL Loss: {:.4f}".format(
                    loss_total1/(i+1), loss_total2/(i+1), loss_total3/(i+1), loss_total_cmal/(i+1)))
                
                # 중간 디버깅 평가: 테스트 데이터셋으로 AUC, AP, mAP 출력
                print("  --> Running mid-epoch evaluation ...")
                auc, ap, mAP = test(av_model, v_model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
                sys.stdout.write(f"      [Mid-Epoch] AUC: {auc:.4f}, AP: {ap:.4f}, mAP: {mAP:.4f} \n")
                sys.stdout.flush()
                
                # WandB 로깅: 중간 평가 결과
                wandb.log({
                    "step": step,
                    "loss_classification": loss_total1/(i+1),
                    "loss_clasm": loss_total2/(i+1),
                    "loss_textual": loss_total3/(i+1),
                    "loss_cmal": loss_total_cmal/(i+1),
                    "mid_epoch_auc": auc,
                    "mid_epoch_ap": ap,
                    "mid_epoch_map": mAP
                })
                
                if ap > ap_best:
                    ap_best = ap
                    checkpoint = {
                        'epoch': e,
                        'av_model_state_dict': av_model.state_dict(),
                        'v_model_state_dict': v_model.state_dict(),
                        'optimizer_av_state_dict': optimizer_av.state_dict(),
                        'optimizer_v_state_dict': optimizer_v.state_dict(),
                        'ap': ap_best
                    }
                    torch.save(checkpoint, args.checkpoint_path)
                    # 추론용 모델 가중치만 따로 저장 (av_model만 저장)
                    torch.save(av_model.state_dict(), args.model_path)
                    sys.stdout.write(f"Best model saved at Epoch {e+1}: New best AP = {ap_best:.4f} \n")
                    sys.stdout.write(f"Inference model saved at: {args.model_path}\n")
                    sys.stdout.flush()

        scheduler_av.step()
        scheduler_v.step()
        
        # Epoch 종료 후 평가
        auc, ap, mAP = test(av_model, v_model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
        
        # WandB 로깅: 에포크 종료 평가 결과
        wandb.log({
            "epoch_end_auc": auc,
            "epoch_end_ap": ap,
            "epoch_end_map": mAP,
            "best_ap": ap_best
        })
        
        if ap > ap_best:
            ap_best = ap 
            checkpoint = {
                'epoch': e,
                'av_model_state_dict': av_model.state_dict(),
                'v_model_state_dict': v_model.state_dict(),
                'optimizer_av_state_dict': optimizer_av.state_dict(),
                'optimizer_v_state_dict': optimizer_v.state_dict(),
                'ap': ap_best
            }
            torch.save(checkpoint, args.checkpoint_path)
            # 추론용 모델 가중치만 따로 저장 (av_model만 저장)
            torch.save(av_model.state_dict(), args.model_path)
            sys.stdout.write(f"Best model saved at Epoch {e+1}: New best AP = {ap_best:.4f} \n")
            sys.stdout.write(f"Inference model saved at: {args.model_path}\n")
            sys.stdout.flush()
    
    # 학습 종료 후 최종 모델 로드 및 추론용 모델 저장 확인
    print(f"Training finished. Best AP: {ap_best:.4f}")
    print(f"Final checkpoint saved at: {args.checkpoint_path}")
    print(f"Final inference model saved at: {args.model_path}")

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
    
    # 모델 저장 디렉토리 생성 (이전 오류 해결용)
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # WandB 초기화
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "ASTRA"),
        name=os.environ.get("WANDB_NAME", "default_run"),
        config=vars(args)
    )

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
