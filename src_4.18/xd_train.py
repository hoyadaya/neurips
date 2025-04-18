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

from model import CLIPVAD, SingleModel
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
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
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

def train(av_model, v_model, train_loader, test_loader, args, label_map: dict, device):
    av_model.to(device)
    v_model.to(device)
    
    # 불확실성 기반 증류 손실 초기화
    ukd_criterion = UKDLoss()

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
        loss_total4 = 0
        loss_total5 = 0
        loss_total_cmal = 0
        loss_total_ukd = 0
        loss_total_cls = 0
        loss_total_align = 0
        loss_total = 0
        
        for i, item in enumerate(train_loader):
            step = i * train_loader.batch_size
            visual_feat, audio_feat, text_labels, feat_lengths = item
            visual_feat = visual_feat.to(device)
            audio_feat = audio_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)
            
            # 오디오-비주얼 모델 forward (새로운 반환값 형식)
            text_features, logits_v, logits_a, logits_av, logits_tv, logits_tav, x_av, var_pred, x_visual, x_audio = av_model(
                visual_feat, audio_feat, None, prompt_text, feat_lengths)
            
            # 시각 모델 forward (불확실성 정보 포함)
            v_features, v_var_pred, _ = v_model(visual_feat, None, feat_lengths)
            
            # 손실 계산 - 개선된 방식
            loss1 = CLAS2(logits_v.squeeze(-1), text_labels, feat_lengths, device)
            loss_total1 += loss1.item()
            
            loss2 = CLASM(logits_tv, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()
            
            loss3 = CLAS2(logits_av, text_labels, feat_lengths, device)
            loss_total3 += loss3.item()
            
            loss4 = CLASM(logits_tav, text_labels, feat_lengths, device)
            loss_total4 += loss4.item()
            
            loss5 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss5 += torch.abs(text_feature_normal @ text_feature_abr)
            loss5 = loss5 / 6 * 1e-4
            loss_total5 += loss5.item()
            
            # CMAL 손실 계산
            loss_a2v_a2b, loss_a2v_a2n, loss_v2a_a2b, loss_v2a_a2n = CMAL(
                text_labels,
                logits_a.squeeze(-1),
                logits_v.squeeze(-1), 
                feat_lengths,
                x_audio,     # 변환된 오디오 특징 (512차원)
                x_visual     # 변환된 비주얼 특징 (512차원)
            )
            
            # 가중치 적용
            lambda_cls = args.lambda_cls
            lambda_align = args.lambda_align
            lambda_cmal = args.lambda_cmal
            lambda_ukd = args.ukd_weight
            
            # 손실 그룹화
            loss_cls = lambda_cls * (loss1 + loss3) / 2
            loss_align = lambda_align * (loss2 + loss4) / 2
            
            loss6 = (loss_a2v_a2b + loss_a2v_a2n + loss_v2a_a2b + loss_v2a_a2n) * 0.25
            loss_cmal = lambda_cmal * loss6
            loss_total_cmal += loss6.item()
            
            # 불확실성 기반 증류(UKD) 손실 계산
            loss7 = ukd_criterion(x_av.detach(), v_features, v_var_pred) * lambda_ukd
            loss_total_ukd += loss7.item()
            
            # 최종 손실 계산 (단일 손실로 통합)
            loss = loss_cls + loss_align + loss5 + loss_cmal + loss7
            
            # 각 구성 요소 누적 (로깅용)
            loss_total_cls += loss_cls.item()
            loss_total_align += loss_align.item()
            loss_total += loss.item()
            
            # 옵티마이저 단계
            optimizer_av.zero_grad()
            loss.backward()
            optimizer_av.step()
            
            if step % 4800 == 0 and step != 0:
                print(f"Epoch {e+1}, Step {step}:")
                print("  개별 손실: CLAS_V: {:.4f}, CLASM_TV: {:.4f}, CLAS_AV: {:.4f}, CLASM_TAV: {:.4f}".format(
                    loss_total1/(i+1), loss_total2/(i+1), loss_total3/(i+1), loss_total4/(i+1)))
                print("  그룹 손실: CLS: {:.4f}, ALIGN: {:.4f}, Contrast: {:.4f}, CMAL: {:.4f}, UKD: {:.4f}".format(
                    loss_total_cls/(i+1), loss_total_align/(i+1), loss_total5/(i+1), 
                    loss_total_cmal/(i+1) * lambda_cmal, loss_total_ukd/(i+1)))
                print("  총 손실: {:.4f}".format(loss_total/(i+1)))
                
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
        print(f"\nEpoch {e+1} 결과:")
        print(f"  AUC: {auc:.4f}, AP: {ap:.4f}, mAP: {mAP:.4f}")
        
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