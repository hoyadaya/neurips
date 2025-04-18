# xd_test.py
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import CLIPVAD, SingleModel
from utils.dataset import XDDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.xd_detectionMAP import getDetectionMAP as dmAP
import xd_option

def test(av_model, v_model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):    
    av_model.to(device)
    av_model.eval()
    v_model.to(device)
    v_model.eval()

    element_logits2_stack = []

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            audio = item[1].squeeze(0)
            length = item[3]

            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
                audio = audio.unsqueeze(0)

            visual = visual.to(device)
            audio = audio.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            
            # 확장된 모델 출력 처리 (추가 반환값 처리)
            text_features, logits_v, logits_a, logits_av, logits_tv, logits_tav, _, _, _, _ = av_model(
                visual, audio, padding_mask, prompt_text, lengths)
            
            # 시각 모델 평가
            _, _, v_model_logits = v_model(visual, padding_mask, lengths)
            
            # reshape 작업
            logits_v = logits_v.squeeze(-1).reshape(-1)
            logits_a = logits_a.squeeze(-1).reshape(-1)
            logits_av = logits_av.reshape(-1)
            logits_tv = logits_tv.reshape(logits_tv.shape[0] * logits_tv.shape[1], logits_tv.shape[2])
            logits_tav = logits_tav.reshape(logits_tav.shape[0] * logits_tav.shape[1], logits_tav.shape[2])
            v_model_logits = v_model_logits.squeeze(-1).reshape(-1)
            
            # 확률 계산
            prob_v = torch.sigmoid(logits_v[0:len_cur])
            prob_av = torch.sigmoid(logits_av[0:len_cur])
            prob_tv = (1 - logits_tv[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob_tav = (1 - logits_tav[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))

            # 결과 누적
            if i == 0:
                ap_v = prob_v
                ap_av = prob_av
                ap_tv = prob_tv
                ap_tav = prob_tav
            else:
                ap_v = torch.cat([ap_v, prob_v], dim=0)
                ap_av = torch.cat([ap_av, prob_av], dim=0)
                ap_tv = torch.cat([ap_tv, prob_tv], dim=0)
                ap_tav = torch.cat([ap_tav, prob_tav], dim=0)

            # mAP 계산 준비 - TAV 로짓 사용
            element_logits_tav = logits_tav[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits_tav = np.repeat(element_logits_tav, 16, 0)
            element_logits2_stack.append(element_logits_tav)

    # 모든 확률을 numpy로 변환
    ap_v = ap_v.cpu().numpy()
    ap_av = ap_av.cpu().numpy()
    ap_tv = ap_tv.cpu().numpy()
    ap_tav = ap_tav.cpu().numpy()
    
    # 성능 평가
    print("\n===== 모델 평가 결과 =====")
    print("시각 모델 (V):")
    ROC_v = roc_auc_score(gt, np.repeat(ap_v, 16))
    AP_v = average_precision_score(gt, np.repeat(ap_v, 16))
    print(f"  AUC: {ROC_v:.4f}, AP: {AP_v:.4f}")
    
    print("오디오-비주얼 모델 (AV):")
    ROC_av = roc_auc_score(gt, np.repeat(ap_av, 16))
    AP_av = average_precision_score(gt, np.repeat(ap_av, 16))
    print(f"  AUC: {ROC_av:.4f}, AP: {AP_av:.4f}")
    
    print("텍스트-비주얼 정렬 (TV):")
    ROC_tv = roc_auc_score(gt, np.repeat(ap_tv, 16))
    AP_tv = average_precision_score(gt, np.repeat(ap_tv, 16))
    print(f"  AUC: {ROC_tv:.4f}, AP: {AP_tv:.4f}")
    
    print("텍스트-오디오-비주얼 정렬 (TAV):")
    ROC_tav = roc_auc_score(gt, np.repeat(ap_tav, 16))
    AP_tav = average_precision_score(gt, np.repeat(ap_tav, 16))
    print(f"  AUC: {ROC_tav:.4f}, AP: {AP_tav:.4f}")
    
    print("\n===== 위치 검출 성능 (TAV 로짓 사용) =====")
    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} = {1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP/(i+1)
    print('Average mAP: {:.2f}%'.format(averageMAP))

    # 최고 성능 모델을 기준으로 반환 (AV 모델과 TAV 정렬 성능)
    return ROC_av, AP_tav, averageMAP


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 
                      'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    test_dataset = XDDataset(args.visual_length, args.test_list, args.test_audio_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    av_model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, 
                    args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, 
                    args.prompt_postfix, args.audio_dim, device)
                   
    v_model = SingleModel(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                          args.visual_head, args.visual_layers, args.attn_window, device)
    
    # 모델 로드
    try:
        checkpoint = torch.load(args.model_path)
        if isinstance(checkpoint, dict) and 'av_model_state_dict' in checkpoint:
            av_model.load_state_dict(checkpoint['av_model_state_dict'])
            v_model.load_state_dict(checkpoint['v_model_state_dict'])
        else:
            # 이전 형식 호환
            av_model.load_state_dict(checkpoint)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

    test(av_model, v_model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
