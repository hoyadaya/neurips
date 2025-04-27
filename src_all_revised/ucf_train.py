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
from ucf_test import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
from utils.CMA_MIL import CMAL  # CMAL 임포트 추가
import ucf_option

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
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def DISTILL(logits_target, logits_source, temperature):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    source_audio_student = F.log_softmax(logits_source/temperature, dim=1)
    target_visual_student = F.softmax(logits_target/temperature, dim=1)
    return kl_loss(source_audio_student, target_visual_student)

def train(av_model, v_model, normal_loader, anomaly_loader, testloader, args, label_map, device):
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
        
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = i * normal_loader.batch_size * 2
            
            normal_features, normal_audio, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_audio, anomaly_label, anomaly_lengths = next(anomaly_iter)

            visual_feat = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            audio_feat = torch.cat([normal_audio, anomaly_audio], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)
            
            # 오디오-비주얼 모델 포워드 패스
            text_features, logits1, logits2, v_logits, a_logits, logits_av = av_model(
                visual_feat, audio_feat, None, prompt_text, feat_lengths)

            # 손실 계산
            loss1 = CLAS2(logits_av, text_labels, feat_lengths, device)
            loss_total1 += loss1.item()

            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()

            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 13 * 1e-4
            loss_total3 += loss3.item()

            # 추가된 손실: 증류와 크로스모달 연결
            loss4 = DISTILL(v_logits.squeeze(-1), a_logits.squeeze(-1), 3.0) + DISTILL(a_logits.squeeze(-1), v_logits.squeeze(-1), 3.0)
            loss5 = DISTILL(logits_av, v_logits.squeeze(-1), 3.0)
            loss6 = CLAS2(a_logits.squeeze(-1), text_labels, feat_lengths, device)

            added_loss = loss5 + loss6 + loss4

            # CMAL 손실 계산 - XD-Violence와 동일
            visual_features, audio_features = av_model.encode_video(visual_feat, audio_feat, None, feat_lengths)
            
            loss_a2v_a2b, loss_a2v_a2n, loss_v2a_a2b, loss_v2a_a2n = CMAL(
                text_labels, 
                a_logits.squeeze(-1), 
                v_logits.squeeze(-1), 
                feat_lengths, 
                audio_features, 
                visual_features
            )
            
            # CMAL 손실 합산
            cmal_loss = (loss_a2v_a2b + loss_a2v_a2n + loss_v2a_a2b + loss_v2a_a2n) * 0.25
            
            if isinstance(cmal_loss, torch.Tensor):
                loss_total_cmal += cmal_loss.item()
            else:
                loss_total_cmal += cmal_loss

            # 최종 손실 계산
            loss_av = loss1 + loss2 + loss3 + cmal_loss + added_loss
            
            optimizer_v.zero_grad()
            optimizer_v.step()
            
            optimizer_av.zero_grad()
            loss_av.backward()
            optimizer_av.step()
            
            if step % 1280 == 0 and step != 0:
                print(f"Epoch {e+1}, Step {step}:")
                print("  AV Loss1: {:.4f}, AV Loss2: {:.4f}, AV Loss3: {:.4f}, CMAL Loss: {:.4f}".format(
                    loss_total1/(i+1), loss_total2/(i+1), loss_total3/(i+1), loss_total_cmal/(i+1)))
                
                # 중간 평가
                auc, ap, mAP = test(av_model, v_model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
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
                
        # 에폭 종료 후 스케줄러 업데이트
        scheduler_av.step()
        scheduler_v.step()
        
        # 현재 모델 상태 저장
        torch.save(av_model.state_dict(), 'model/model_cur.pth')
        
        # Epoch 종료 후 평가
        auc, ap, mAP = test(av_model, v_model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
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
            # 추론용 모델 가중치만 따로 저장
            torch.save(av_model.state_dict(), args.model_path)
            sys.stdout.write(f"Best model saved at Epoch {e+1}: New best AP = {ap_best:.4f} \n")
            sys.stdout.write(f"Inference model saved at: {args.model_path}\n")
    
    # 학습 종료 후 최종 체크포인트에서 av_model 가중치 추출하여 저장
    checkpoint = torch.load(args.checkpoint_path)
    av_model.load_state_dict(checkpoint['av_model_state_dict'])
    torch.save(checkpoint['av_model_state_dict'], args.model_path)
    sys.stdout.write(f"Final inference model saved to {args.model_path}\n")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    # 오디오 데이터를 포함하도록 데이터셋 생성 수정
    normal_dataset = UCFDataset(args.visual_length, args.train_list, args.audio_list, False, label_map, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, args.audio_list, False, label_map, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, args.test_audio_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 오디오 차원 추가하여 모델 생성
    av_model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, 
                      args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, 
                      args.prompt_postfix, args.audio_dim, device)
    
    # SingleModel도 추가
    v_model = SingleModel(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                         args.visual_head, args.visual_layers, args.attn_window, device)

    train(av_model, v_model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
