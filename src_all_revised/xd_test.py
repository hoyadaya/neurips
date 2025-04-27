# xd_test.py
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import CLIPVAD, SingleModel, UncertaintyPredictor
from utils.dataset import XDDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.xd_detectionMAP import getDetectionMAP as dmAP
import xd_option



def test(model, visual_model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):    
    model.to(device)
    model.eval()
    visual_model.to(device)
    visual_model.eval()


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
            
            # AV 모델로 예측
            text_features, logits_av, logits2, v_logits, a_logits = model(
                visual, audio, padding_mask, prompt_text, lengths, return_fused=False
            )
            
            # 시각 모델로 예측 (옵션)
            # v_features, v_logits_only = visual_model(visual, padding_mask, lengths)
            
            logits_av = logits_av.reshape(-1)  # 수정된 부분
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
        
            prob_av = torch.sigmoid(logits_av[0:len_cur]) 
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))

            if i == 0:
                ap_av = prob_av # 수정본
                ap2 = prob2
            else:
                ap_av = torch.cat([ap_av, prob_av], dim=0) # 수정본
                ap2 = torch.cat([ap2, prob2], dim=0) # 수정본

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    ap_av = ap_av.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap_av = ap_av.tolist()
    ap2 = ap2.tolist()

    # logits_av를 사용한 분류 성능 계산
    ROC_av = roc_auc_score(gt, np.repeat(ap_av, 16))
    AP_av = average_precision_score(gt, np.repeat(ap_av, 16)) # y_true, y_score 
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

    av_model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, 
                    args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, 
                    args.prompt_postfix, args.audio_dim, device)
    
    v_model = SingleModel(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                          args.visual_head, args.visual_layers, args.attn_window, device)
                   
    model_param = torch.load(args.model_path)
    av_model.load_state_dict(model_param['av_model_state_dict'])
    v_model.load_state_dict(model_param['v_model_state_dict'])

    test(av_model, v_model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
