# visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from tqdm import tqdm

def visualize_modality_scores(video_id, time_steps, visual_scores, audio_scores, av_scores, 
                             gt=None, threshold=0.5, save_dir='./visualizations'):
    """
    각 모달리티(visual, audio, visual+audio)의 이상 탐지 점수를 시각화
    
    Args:
        video_id: 비디오 ID 또는 이름
        time_steps: 시간 축 (일반적으로 프레임 인덱스)
        visual_scores: 시각적 모달리티의 이상 점수
        audio_scores: 오디오 모달리티의 이상 점수
        av_scores: 시각-오디오 융합 모달리티의 이상 점수
        gt: 실제 정답 라벨 (선택 사항)
        threshold: 이상 감지 임계값
        save_dir: 시각화 결과를 저장할 디렉토리
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 그래프 생성
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1])
    
    # 시각적 모달리티 점수
    ax1 = plt.subplot(gs[0])
    ax1.plot(time_steps, visual_scores, 'b-', linewidth=2, label='Visual Score')
    if threshold is not None:
        ax1.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    
    # 이상 탐지 구간 강조
    if threshold is not None:
        for i in range(len(time_steps)):
            if visual_scores[i] > threshold:
                ax1.axvspan(time_steps[i]-0.5, time_steps[i]+0.5, color='red', alpha=0.3)
    
    if gt is not None:
        for i in range(len(time_steps)):
            if i < len(gt) and gt[i] == 1:
                ax1.axvspan(time_steps[i]-0.5, time_steps[i]+0.5, color='green', alpha=0.2)
    
    ax1.set_title('Visual Modality Anomaly Score')
    ax1.set_ylabel('Score')
    ax1.set_xlim(time_steps[0], time_steps[-1])
    ax1.set_ylim(0, 1.05)
    ax1.grid(True)
    ax1.legend()
    
    # 오디오 모달리티 점수
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(time_steps, audio_scores, 'g-', linewidth=2, label='Audio Score')
    if threshold is not None:
        ax2.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    
    # 이상 탐지 구간 강조
    if threshold is not None:
        for i in range(len(time_steps)):
            if audio_scores[i] > threshold:
                ax2.axvspan(time_steps[i]-0.5, time_steps[i]+0.5, color='red', alpha=0.3)
    
    if gt is not None:
        for i in range(len(time_steps)):
            if i < len(gt) and gt[i] == 1:
                ax2.axvspan(time_steps[i]-0.5, time_steps[i]+0.5, color='green', alpha=0.2)
    
    ax2.set_title('Audio Modality Anomaly Score')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True)
    ax2.legend()
    
    # 시각-오디오 융합 점수
    ax3 = plt.subplot(gs[2], sharex=ax1)
    ax3.plot(time_steps, av_scores, 'r-', linewidth=2, label='Audio-Visual Score')
    if threshold is not None:
        ax3.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    
    # 이상 탐지 구간 강조
    if threshold is not None:
        for i in range(len(time_steps)):
            if av_scores[i] > threshold:
                ax3.axvspan(time_steps[i]-0.5, time_steps[i]+0.5, color='red', alpha=0.3)
    
    if gt is not None:
        for i in range(len(time_steps)):
            if i < len(gt) and gt[i] == 1:
                ax3.axvspan(time_steps[i]-0.5, time_steps[i]+0.5, color='green', alpha=0.2)
    
    ax3.set_title('Audio-Visual Fusion Anomaly Score')
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Score')
    ax3.set_ylim(0, 1.05)
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'video_{video_id}_scores.png'), dpi=300)
    plt.close()

    # 3개 점수를 한 그래프에 통합
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, visual_scores, 'b-', linewidth=2, label='Visual Score')
    plt.plot(time_steps, audio_scores, 'g-', linewidth=2, label='Audio Score')
    plt.plot(time_steps, av_scores, 'r-', linewidth=2, label='Audio-Visual Score')
    
    if threshold is not None:
        plt.axhline(y=threshold, color='k', linestyle='--', label='Threshold')
    
    if gt is not None:
        for i in range(len(time_steps)):
            if i < len(gt) and gt[i] == 1:
                plt.axvspan(time_steps[i]-0.5, time_steps[i]+0.5, color='green', alpha=0.2)
    
    plt.title(f'Video {video_id} - All Modality Anomaly Scores')
    plt.xlabel('Frame Index')
    plt.ylabel('Score')
    plt.ylim(0, 1.05)
    plt.xlim(time_steps[0], time_steps[-1])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'video_{video_id}_combined_scores.png'), dpi=300)
    plt.close()
    
    return os.path.join(save_dir, f'video_{video_id}_scores.png'), os.path.join(save_dir, f'video_{video_id}_combined_scores.png')

def visualize_all_videos_scores(all_video_scores, video_ids, gt_data=None, threshold=0.5, save_dir='./visualizations'):
    """
    모든 테스트 비디오의 이상 점수를 시각화
    
    Args:
        all_video_scores: 각 비디오의 모달리티별 점수를 담은 리스트
                         [{"visual": [...], "audio": [...], "av": [...]}, ...]
        video_ids: 비디오 ID 리스트
        gt_data: 정답 데이터
        threshold: 이상 감지 임계값
    """
    os.makedirs(save_dir, exist_ok=True)
    
    all_vis_paths = []
    all_combined_paths = []
    
    for i, (scores, video_id) in enumerate(zip(all_video_scores, video_ids)):
        if gt_data is not None and i < len(gt_data):
            gt = gt_data[i]
        else:
            gt = None
            
        time_steps = np.arange(len(scores["visual"]))
        vis_path, combined_path = visualize_modality_scores(
            video_id, 
            time_steps, 
            scores["visual"], 
            scores["audio"], 
            scores["av"],
            gt=gt,
            threshold=threshold,
            save_dir=save_dir
        )
        all_vis_paths.append(vis_path)
        all_combined_paths.append(combined_path)
        
    # 모든 비디오의 평균 점수 시각화
    avg_visual = np.mean([np.array(s["visual"]) for s in all_video_scores], axis=0)
    avg_audio = np.mean([np.array(s["audio"]) for s in all_video_scores], axis=0)
    avg_av = np.mean([np.array(s["av"]) for s in all_video_scores], axis=0)
    
    if len(avg_visual) > 0:
        time_steps = np.arange(len(avg_visual))
        visualize_modality_scores(
            "average", 
            time_steps, 
            avg_visual, 
            avg_audio, 
            avg_av,
            threshold=threshold,
            save_dir=save_dir
        )
    
    return all_vis_paths, all_combined_paths

def plot_performance_metrics(epoch_list, auc_list, ap_list, map_list, save_path='./metrics_plot.png'):
    """
    학습 성능 지표 (AUC, AP, mAP)의 변화를 시각화
    
    Args:
        epoch_list: 에포크 리스트
        auc_list: AUC 값 리스트
        ap_list: AP 값 리스트
        map_list: mAP 값 리스트
        save_path: 저장 경로
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, auc_list, 'bo-', linewidth=2, label='AUC')
    plt.plot(epoch_list, ap_list, 'go-', linewidth=2, label='AP')
    plt.plot(epoch_list, map_list, 'ro-', linewidth=2, label='mAP')
    
    plt.title('Performance Metrics by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    
    # 저장 경로의 디렉토리가 존재하지 않으면 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path

def compare_modalities_performance(videos_data, threshold=0.5, save_dir='./comparisons'):
    """
    각 모달리티의 성능을 비교 분석
    
    Args:
        videos_data: 비디오별 점수 및 GT 데이터 
                    [{"visual": [...], "audio": [...], "av": [...], "gt": [...]}, ...]
        threshold: 이상 감지 임계값
        save_dir: 저장 디렉토리
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 각 모달리티별 true positives, false positives, false negatives, true negatives 계산
    tp_visual, fp_visual, fn_visual, tn_visual = 0, 0, 0, 0
    tp_audio, fp_audio, fn_audio, tn_audio = 0, 0, 0, 0
    tp_av, fp_av, fn_av, tn_av = 0, 0, 0, 0
    
    # 계산 및 카운트
    for video_data in videos_data:
        visual_scores = np.array(video_data["visual"])
        audio_scores = np.array(video_data["audio"])
        av_scores = np.array(video_data["av"])
        gt = np.array(video_data["gt"]) if "gt" in video_data else None
        
        if gt is not None:
            # Visual
            tp_visual += np.sum((visual_scores > threshold) & (gt == 1))
            fp_visual += np.sum((visual_scores > threshold) & (gt == 0))
            fn_visual += np.sum((visual_scores <= threshold) & (gt == 1))
            tn_visual += np.sum((visual_scores <= threshold) & (gt == 0))
            
            # Audio
            tp_audio += np.sum((audio_scores > threshold) & (gt == 1))
            fp_audio += np.sum((audio_scores > threshold) & (gt == 0))
            fn_audio += np.sum((audio_scores <= threshold) & (gt == 1))
            tn_audio += np.sum((audio_scores <= threshold) & (gt == 0))
            
            # Audio-Visual
            tp_av += np.sum((av_scores > threshold) & (gt == 1))
            fp_av += np.sum((av_scores > threshold) & (gt == 0))
            fn_av += np.sum((av_scores <= threshold) & (gt == 1))
            tn_av += np.sum((av_scores <= threshold) & (gt == 0))
    
    # 정확도, 정밀도, 재현율, F1 점수 계산
    # Visual
    accuracy_visual = (tp_visual + tn_visual) / (tp_visual + fp_visual + fn_visual + tn_visual) if (tp_visual + fp_visual + fn_visual + tn_visual) > 0 else 0
    precision_visual = tp_visual / (tp_visual + fp_visual) if (tp_visual + fp_visual) > 0 else 0
    recall_visual = tp_visual / (tp_visual + fn_visual) if (tp_visual + fn_visual) > 0 else 0
    f1_visual = 2 * precision_visual * recall_visual / (precision_visual + recall_visual) if (precision_visual + recall_visual) > 0 else 0
    
    # Audio
    accuracy_audio = (tp_audio + tn_audio) / (tp_audio + fp_audio + fn_audio + tn_audio) if (tp_audio + fp_audio + fn_audio + tn_audio) > 0 else 0
    precision_audio = tp_audio / (tp_audio + fp_audio) if (tp_audio + fp_audio) > 0 else 0
    recall_audio = tp_audio / (tp_audio + fn_audio) if (tp_audio + fn_audio) > 0 else 0
    f1_audio = 2 * precision_audio * recall_audio / (precision_audio + recall_audio) if (precision_audio + recall_audio) > 0 else 0
    
    # Audio-Visual
    accuracy_av = (tp_av + tn_av) / (tp_av + fp_av + fn_av + tn_av) if (tp_av + fp_av + fn_av + tn_av) > 0 else 0
    precision_av = tp_av / (tp_av + fp_av) if (tp_av + fp_av) > 0 else 0
    recall_av = tp_av / (tp_av + fn_av) if (tp_av + fn_av) > 0 else 0
    f1_av = 2 * precision_av * recall_av / (precision_av + recall_av) if (precision_av + recall_av) > 0 else 0
    
    # 시각화: 정확도, 정밀도, 재현율, F1 점수 비교
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    visual_values = [accuracy_visual, precision_visual, recall_visual, f1_visual]
    audio_values = [accuracy_audio, precision_audio, recall_audio, f1_audio]
    av_values = [accuracy_av, precision_av, recall_av, f1_av]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, visual_values, width, label='Visual', color='blue')
    bars2 = ax.bar(x, audio_values, width, label='Audio', color='green')
    bars3 = ax.bar(x + width, av_values, width, label='Audio-Visual', color='red')
    
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics by Modality')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # 표 추가
    plt.table(cellText=[[f"{v:.4f}" for v in visual_values],
                        [f"{v:.4f}" for v in audio_values],
                        [f"{v:.4f}" for v in av_values]],
             rowLabels=['Visual', 'Audio', 'Audio-Visual'],
             colLabels=metrics,
             loc='bottom',
             bbox=[0.0, -0.35, 1.0, 0.2])
    
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'modalities_comparison.png'), dpi=300)
    plt.close()
    
    print(f"Modalities comparison saved to {os.path.join(save_dir, 'modalities_comparison.png')}")
    
    return os.path.join(save_dir, 'modalities_comparison.png')
