import argparse

parser = argparse.ArgumentParser(description='VadCLIP + MACIL-SD')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=1, type=int)
parser.add_argument('--attn-window', default=64, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=7, type=int)
parser.add_argument('--audio-dim', default=128, type=int)

# Self-distillation parameters
parser.add_argument('--m', default=0.91, type=float, help='Self-distillation mixing coefficient')
parser.add_argument('--distill-weight', default=1.0, type=float, help='Weight for distillation loss')

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='model_xd/model_xd.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='model_xd/checkpoint.pth')
parser.add_argument('--batch-size', default=96, type=int)  # Reduced batch size for dual model training
parser.add_argument('--train-list', default='list/xd_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/xd_CLIP_rgbtest.csv')
parser.add_argument('--audio-list', default='list/xd_CLIP_audio.csv')  # Added for MACIL-SD
parser.add_argument('--test-audio-list', default='list/xd_CLIP_audiotest.csv')  # Added for MACIL-SD
parser.add_argument('--gt-path', default='list/gt.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment.npy')
parser.add_argument('--gt-label-path', default='list/gt_label.npy')

parser.add_argument('--lr', default=1e-5)
parser.add_argument('--v-lr', default=2e-6, help='Learning rate for visual teacher model')
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[3, 6, 10])