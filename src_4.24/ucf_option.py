import argparse

parser = argparse.ArgumentParser(description='VadCLIP + MACIL-SD')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=2, type=int)
parser.add_argument('--attn-window', default=8, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=14, type=int)
parser.add_argument('--audio-dim', default=128, type=int)  # 오디오 특징 차원

# Self-distillation parameters
parser.add_argument('--m', default=0.91, type=float, help='Self-distillation mixing coefficient')
parser.add_argument('--distill-weight', default=1.0, type=float, help='Weight for distillation loss')

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='model_ucf/model_ucf.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='model_ucf/checkpoint.pth')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--train-list', default='list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--audio-list', default='list/ucf_CLIP_audio.csv')  # 이미 있음
parser.add_argument('--test-audio-list', default='list/ucf_CLIP_audiotest.csv')  # 테스트 오디오 목록

parser.add_argument('--gt-path', default='list/gt_ucf.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default='list/gt_label_ucf.npy')

parser.add_argument('--lr', default=2e-5)
parser.add_argument('--v-lr', default=2e-6, help='Learning rate for visual teacher model')
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[4, 8])
