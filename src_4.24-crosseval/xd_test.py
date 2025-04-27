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

def test(model, v_model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):    
    model.to(device)
    model.eval()

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
            
            # 모델 출력
            text_features, logtis1, logits2, v_logits, a_logits, logits_av = model(
                visual, audio, padding_mask, prompt_text, lengths)
            
            # 출력 처리
            logits_av = logits_av.reshape(-1)
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            
            # 이진 분류 확률 계산
            prob_av = torch.sigmoid(logits_av[0:len_cur])
            
            # 주의: UCF 모델은 14개 클래스, XD는 7개 클래스
            # 따라서 logits2의 첫 번째 클래스(정상)만 사용
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))

            if i == 0:
                ap_av = prob_av
                ap2 = prob2
            else:
                ap_av = torch.cat([ap_av, prob_av], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            # XD-Violence는 프레임당 16개 샘플 사용
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    ap_av = ap_av.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap_av = ap_av.tolist()
    ap2 = ap2.tolist()

    # 평가 지표 계산
    ROC_av = roc_auc_score(gt, np.repeat(ap_av, 16))
    AP_av = average_precision_score(gt, np.repeat(ap_av, 16))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))

    print("AUC (using logits_av): ", ROC_av, " AP (using logits_av): ", AP_av)
    print("AUC2: ", ROC2, " AP2:", AP2)

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP/(i+1)
    print('average MAP: {:.2f}'.format(averageMAP))

    return ROC_av, AP2, averageMAP


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

    # 중요: UCF 모델 구성과 일치하도록 매개변수 수정
    model = CLIPVAD(
        num_class=14,             # UCF 모델은 14개 클래스 사용
        embed_dim=args.embed_dim, # 512 유지
        visual_length=args.visual_length, # 256 유지
        visual_width=args.visual_width,   # 512 유지
        visual_head=args.visual_head,     # 헤드 수 유지
        visual_layers=2,  # UCF 모델은 2개 레이어 사용
        attn_window=8,    # UCF 모델은 8 사용
        prompt_prefix=args.prompt_prefix,
        prompt_postfix=args.prompt_postfix,
        audio_dim=args.audio_dim,
        device=device
    )
    
    # SingleModel도 같은 파라미터로 맞춰줌
    v_model = SingleModel(
        num_class=14,  # UCF 모델은 14개 클래스
        embed_dim=args.embed_dim,
        visual_length=args.visual_length,
        visual_width=args.visual_width,
        visual_head=args.visual_head,
        visual_layers=2,  # UCF 모델과 일치
        attn_window=8,    # UCF 모델과 일치
        device=device
    )
    

    model_param = torch.load(args.model_path)
    
    # 모델 가중치 로드
    model.load_state_dict(model_param, strict=False)  
    # UCF 모델로 XD 데이터셋 테스트 실행
    test(model, v_model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
