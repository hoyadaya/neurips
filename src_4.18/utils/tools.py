import torch
import numpy as np
import math

def get_batch_label(texts, prompt_text, label_map: dict):
    label_vectors = torch.zeros(0)
    if len(label_map) != 7:
        if len(label_map) == 2:
            for text in texts:
                label_vector = torch.zeros(2)
                if text == 'Normal':
                    label_vector[0] = 1
                else:
                    label_vector[1] = 1
                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
        else:
            for text in texts:
                label_vector = torch.zeros(len(prompt_text))
                if text in label_map:
                    label_text = label_map[text]
                    label_vector[prompt_text.index(label_text)] = 1

                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
    else:
        for text in texts:
            label_vector = torch.zeros(len(prompt_text))
            labels = text.split('-')
            for label in labels:
                if label in label_map:
                    label_text = label_map[label]
                    label_vector[prompt_text.index(label_text)] = 1
            
            label_vector = label_vector.unsqueeze(0)
            label_vectors = torch.cat([label_vectors, label_vector], dim=0)

    return label_vectors

def get_prompt_text(label_map: dict):
    prompt_text = []
    for v in label_map.values():
        prompt_text.append(v)

    return prompt_text

def get_batch_mask(lengths, maxlen):
    batch_size = lengths.shape[0]
    mask = torch.empty(batch_size, maxlen)
    mask.fill_(0)
    for i in range(batch_size):
        if lengths[i] < maxlen:
            mask[i, lengths[i]:maxlen] = 1
    
    return mask.bool()

def random_extract(feat, t_max):
   r = np.random.randint(feat.shape[0] - t_max)
   return feat[r : r+t_max, :]

def uniform_extract(feat, t_max, avg: bool = True):
    new_feat = np.zeros((t_max, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), t_max+1, dtype=np.int32)
    if avg == True:
        for i in range(t_max):
            if r[i]!=r[i+1]:
                new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
            else:
                new_feat[i,:] = feat[r[i],:]
    else:
        r = np.linspace(0, feat.shape[0]-1, t_max, dtype=np.uint16)
        new_feat = feat[r, :]
            
    return new_feat

def pad(feat, min_len):
    clip_length = feat.shape[0]
    if clip_length <= min_len:
       return np.pad(feat, ((0, min_len - clip_length), (0, 0)), mode='constant', constant_values=0)
    else:
       return feat

def process_feat(feat, length, is_random=False):
    clip_length = feat.shape[0]
    if feat.shape[0] > length:
        if is_random:
            return random_extract(feat, length), length
        else:
            return uniform_extract(feat, length), length
    else:
        return pad(feat, length), clip_length

def process_feat_audio(visual_feat, audio_feat, length, is_random=False):
    """Process both visual and audio features to match length"""
    clip_length = visual_feat.shape[0]
    
    # Make sure audio features match visual features in sequence length
    if audio_feat.shape[0] != visual_feat.shape[0]:
        audio_feat = uniform_extract(audio_feat, visual_feat.shape[0], avg=True)
    
    if visual_feat.shape[0] > length:
        if is_random:
            start_idx = np.random.randint(visual_feat.shape[0] - length)
            visual_feat_processed = visual_feat[start_idx : start_idx+length, :]
            audio_feat_processed = audio_feat[start_idx : start_idx+length, :]
            return visual_feat_processed, audio_feat_processed, length
        else:
            visual_feat_processed = uniform_extract(visual_feat, length)
            audio_feat_processed = uniform_extract(audio_feat, length)
            return visual_feat_processed, audio_feat_processed, length
    else:
        visual_feat_processed = pad(visual_feat, length)
        audio_feat_processed = pad(audio_feat, length)
        return visual_feat_processed, audio_feat_processed, clip_length

def process_split(feat, length):
    clip_length = feat.shape[0]
    if clip_length < length:
        return pad(feat, length), clip_length
    else:
        split_num = int(clip_length / length) + 1
        for i in range(split_num):
            if i == 0:
                split_feat = feat[i*length:i*length+length, :].reshape(1, length, feat.shape[1])
            elif i < split_num - 1:
                split_feat = np.concatenate([split_feat, feat[i*length:i*length+length, :].reshape(1, length, feat.shape[1])], axis=0)
            else:
                split_feat = np.concatenate([split_feat, pad(feat[i*length:i*length+length, :], length).reshape(1, length, feat.shape[1])], axis=0)

        return split_feat, clip_length

def process_split_audio(visual_feat, audio_feat, length):
    """Process both visual and audio features for test time"""
    clip_length = visual_feat.shape[0]
    
    # Make sure audio features match visual features in sequence length
    if audio_feat.shape[0] != visual_feat.shape[0]:
        audio_feat = uniform_extract(audio_feat, visual_feat.shape[0], avg=True)
    
    if clip_length < length:
        return pad(visual_feat, length), pad(audio_feat, length), clip_length
    else:
        split_num = int(clip_length / length) + 1
        for i in range(split_num):
            visual_chunk = visual_feat[i*length:min(i*length+length, clip_length), :]
            audio_chunk = audio_feat[i*length:min(i*length+length, clip_length), :]
            
            # Pad if needed
            if visual_chunk.shape[0] < length:
                visual_chunk = pad(visual_chunk, length)
                audio_chunk = pad(audio_chunk, length)
                
            visual_chunk = visual_chunk.reshape(1, length, visual_feat.shape[1])
            audio_chunk = audio_chunk.reshape(1, length, audio_feat.shape[1])
            
            if i == 0:
                split_visual_feat = visual_chunk
                split_audio_feat = audio_chunk
            else:
                split_visual_feat = np.concatenate([split_visual_feat, visual_chunk], axis=0)
                split_audio_feat = np.concatenate([split_audio_feat, audio_chunk], axis=0)

        return split_visual_feat, split_audio_feat, clip_length

def cosine_scheduler(base_value, final_value, curr_epoch, epochs):
    """Cosine scheduler used for self-distillation mixing coefficient"""
    return final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * curr_epoch / epochs))