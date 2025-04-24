import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import os
import re
import utils.tools as tools

class XDDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, audio_list: str, test_mode: bool, label_map: dict):
        self.df = pd.read_csv(file_path)
        self.audio_df = pd.read_csv(audio_list)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        
        # Create a mapping from video file key to audio path
        self.video_to_audio = {}
        
        # Format: movie__#timestamp_label_X__vggish.npy
        for _, row in self.audio_df.iterrows():
            audio_path = row['path']
            # Extract the base key (without __vggish.npy part)
            # Example: Bad.Boys.1995__#00-26-51_00-27-53_label_B2-0-0
            audio_key = os.path.basename(audio_path).replace('__vggish.npy', '')
            self.video_to_audio[audio_key] = audio_path
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        visual_path = self.df.loc[index]['path']
        clip_feature = np.load(visual_path)
        
        # Extract visual file key (removing the __number.npy part)
        # Example: /data/.../Bad.Boys.1995__#00-26-51_00-27-53_label_B2-0-0__0.npy
        # -> Bad.Boys.1995__#00-26-51_00-27-53_label_B2-0-0
        visual_basename = os.path.basename(visual_path)
        visual_key = re.sub(r'__\d+\.npy$', '', visual_basename)
        
        # Try to get corresponding audio feature
        if visual_key in self.video_to_audio:
            audio_path = self.video_to_audio[visual_key]
            try:
                audio_feature = np.load(audio_path)
            except:
                print(f"Warning: Failed to load audio feature from {audio_path}")
                audio_feature = np.zeros((clip_feature.shape[0], 128), dtype=np.float32)
        else:
            print(f"Warning: No matching audio for visual key {visual_key}")
            audio_feature = np.zeros((clip_feature.shape[0], 128), dtype=np.float32)
            
        if self.test_mode == False:
            clip_feature, audio_feature, clip_length = tools.process_feat_audio(clip_feature, audio_feature, self.clip_dim)
        else:
            clip_feature, audio_feature, clip_length = tools.process_split_audio(clip_feature, audio_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        audio_feature = torch.tensor(audio_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, audio_feature, clip_label, clip_length

class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, audio_list: str, test_mode: bool, label_map: dict, normal: bool = False):
        self.df = pd.read_csv(file_path)
        self.audio_df = pd.read_csv(audio_list)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df = self.df.reset_index()
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df = self.df.reset_index()
        
        # 변경된 부분: UCF-Crime 데이터셋 형식에 맞게 비디오-오디오 매핑 생성
        self.video_to_audio = {}
        for _, row in self.audio_df.iterrows():
            audio_path = row['path']
            # 예: /data/datasets/tarfiles/vggish/Abuse/Abuse001_x264__0.npy,Abuse
            # 에서 Abuse001_x264 추출
            audio_basename = os.path.basename(audio_path.split(',')[0])  # ,Abuse 부분 제거 및 파일명만 추출
            audio_key = re.sub(r'__\d+\.npy$', '', audio_basename)  # __0.npy 부분 제거
            self.video_to_audio[audio_key] = audio_path.split(',')[0]  # ,Abuse 부분 제거
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        visual_path = self.df.loc[index]['path']
        clip_feature = np.load(visual_path)
        
        # 변경된 부분: UCF-Crime 형식의 비주얼 키 추출
        # 예: /data/datasets/tarfiles/UCFClipFeatures/UCFClipFeatures/Abuse/Abuse001_x264__0.npy,Abuse
        # 에서 Abuse001_x264 추출
        visual_basename = os.path.basename(visual_path.split(',')[0])  # ,Abuse 부분 제거 및 파일명만 추출
        visual_key = re.sub(r'__\d+\.npy$', '', visual_basename)  # __0.npy 부분 제거
        
        # 해당 비주얼 키에 매칭되는 오디오 파일 검색
        if visual_key in self.video_to_audio:
            audio_path = self.video_to_audio[visual_key]
            try:
                audio_feature = np.load(audio_path)
            except:
                print(f"Warning: Failed to load audio feature from {audio_path}")
                audio_feature = np.zeros((clip_feature.shape[0], 128), dtype=np.float32)
        else:
            print(f"Warning: No matching audio for visual key {visual_key}")
            audio_feature = np.zeros((clip_feature.shape[0], 128), dtype=np.float32)
        
        if self.test_mode == False:
            clip_feature, audio_feature, clip_length = tools.process_feat_audio(clip_feature, audio_feature, self.clip_dim)
        else:
            clip_feature, audio_feature, clip_length = tools.process_split_audio(clip_feature, audio_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        audio_feature = torch.tensor(audio_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, audio_feature, clip_label, clip_length