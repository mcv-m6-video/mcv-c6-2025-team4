#!/usr/bin/env python3

"""
File containing classes related to the frame datasets.
"""

#Standard imports
from util.io import load_json
import os
import random
import numpy as np
import copy
import torch
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm
import pickle
import math

#Constants

# Pad the start/end of videos with empty frames
DEFAULT_PAD_LEN = 5
FPS_SN = 25


class ActionSpotDataset(Dataset):
    def __init__(
            self,
            classes,                    # dict of class names to idx
            game_file,                 # path to label json
            frame_dir,                  # path to frames
            flow_dir,                   # path to optical flow frames
            store_dir,                  # path to store files (with frames path and labels per clip)
            store_mode,                 # 'store' or 'load'
            clip_len,                   # Number of frames per clip
            dataset_len,                # Number of clips
            stride=1,                   # Downsample frame rate
            overlap=1,                  # Overlap between clips (in proportion to clip_len)
            pad_len=DEFAULT_PAD_LEN,    # Number of frames to pad the start
                                        # and end of videos
            dataset = 'soccernetball',     # Dataset name
            labels_dir = None,          # Directory with labels for SoccerNetBall
            task = 'classification'     # Classification or localization
    ):
        self._src_file = game_file
        self._games = load_json(game_file)
        self._split = game_file.split('/')[-1].split('.')[0]
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._games)}
        self._dataset = dataset
        assert dataset == 'soccernetball'
        self._store_dir = store_dir
        self._store_mode = store_mode
        assert store_mode in ['store', 'load']
        self._clip_len = clip_len
        assert clip_len > 0
        self._stride = stride
        assert stride > 0
        assert overlap >= 0 and overlap <= 1
        self._clip_sampling_step = 1 if overlap == 1 else int((1 - overlap) * clip_len * stride)
        self._pad_len = pad_len
        assert pad_len >= 0     
        self._labels_dir = labels_dir
        self._task = task
        assert task in ['classification', 'spotting']

        #Frame reader class
        self._frame_reader = FrameReader(frame_dir, flow_dir, dataset = dataset)

        #Store or load clips
        if self._store_mode == 'store':
            self._store_clips()
        elif self._store_mode == 'load':
            self._load_clips()

        if dataset_len is None:
            self._dataset_len = len(self._frame_paths)
        else:
            self._dataset_len = dataset_len

        self._total_len = len(self._frame_paths)

    def _store_clips(self):
        #Initialize frame paths list
        self._frame_paths = []
        self._labels_store = []

        for video in tqdm(self._games):
            video_len = int(video['num_frames'])

            #Load labels
            video_half = 1
            labels_file = load_json(os.path.join(self._labels_dir, video['video'] + '/Labels-ball.json'))['annotations']

            for base_idx in range(-self._pad_len * self._stride, max(0, video_len - 1 + (2 * self._pad_len - self._clip_len) * self._stride), self._clip_sampling_step):

                frames_paths = self._frame_reader.load_paths(video['video'], base_idx, base_idx + self._clip_len * self._stride, stride=self._stride)

                labels = []

                for event in labels_file:
                    event_half = int(event['gameTime'][0])
                    if event_half == video_half:
                        event_frame = int(int(event['position']) / 1000 * FPS_SN) #miliseconds to frames
                        label_idx = (event_frame - base_idx) // self._stride #position of event in clip

                        if (label_idx >= 0 and label_idx < self._clip_len):
                            label = self._class_dict[event['label']]
                            labels.append({'label': label, 'label_idx': label_idx}) #add label and position to list (for both classification and location)

                if frames_paths[1] != -1: #in case no frames were available

                    self._frame_paths.append(frames_paths)
                    self._labels_store.append(labels)

        #Save to store
        store_path = os.path.join(self._store_dir, 'LEN' + str(self._clip_len) + 'SPLIT' + self._split) #store clips information of dataset with LEN and SPLIT information

        if not os.path.exists(store_path):
            os.makedirs(store_path)

        with open(store_path + '/frame_paths.pkl', 'wb') as f:
            pickle.dump(self._frame_paths, f)
        with open(store_path + '/labels.pkl', 'wb') as f:
            pickle.dump(self._labels_store, f)
        print('Stored clips to ' + store_path)
        return
    
    def _load_clips(self):
        store_path = os.path.join(self._store_dir, 'LEN' + str(self._clip_len) + 'SPLIT' + self._split)
        
        with open(store_path + '/frame_paths.pkl', 'rb') as f:
            self._frame_paths = pickle.load(f)
        with open(store_path + '/labels.pkl', 'rb') as f:
            self._labels_store = pickle.load(f)
        print('Loaded clips from ' + store_path)
        return
    # Modify _get_one to return both RGB and flow frames
    def _get_one(self):
        idx = random.randint(0, self._total_len - 1)
        frames_path = self._frame_paths[idx]
        dict_label = self._labels_store[idx]      

        # Load RGB and Flow frames
        rgb_frames, flow_frames = self._frame_reader.load_frames(frames_path, pad=True, stride=self._stride)

        # Process labels (spotting or classification)
        if self._task == 'spotting':
            labels = np.zeros(self._clip_len, np.int64)
            for label in dict_label:
                labels[label['label_idx']] = label['label']

        elif self._task == 'classification':
            labels = np.zeros(len(self._class_dict), np.int64)
            for label in dict_label:
                labels[label['label'] - 1] = 1  # Labels start at 1, adjust for zero-indexed

        return {'frame': rgb_frames, 'flow': flow_frames, 'contains_event': int(np.sum(labels) > 0), 'label': labels}

    def __getitem__(self, unused):
        return self._get_one()

    def __len__(self):
        return self._dataset_len

    def print_info(self):
        _print_info_helper(self._src_file, self._games)


class FrameReader:

    def __init__(self, frame_dir, flow_dir, dataset):
        self._frame_dir = frame_dir
        self._flow_dir = flow_dir
        self.dataset = dataset

    def read_frame(self, frame_path):
        img = torchvision.io.read_image(frame_path)
        return img
    
    def read_flow(self, flow_path):
        # Assume flow files are saved as 2-channel images (u, v flow components)
        npy_array=np.load(flow_path)
        flow = torch.from_numpy(npy_array)
        flow=flow.permute(2, 0, 1)
        flow=flow.float()
        return flow
    
    def load_paths(self, video_name, start, end, stride=1):

        path = os.path.join(self._frame_dir, video_name)
        flow_path = os.path.join(self._flow_dir, video_name+'_flow')

        found_start = -1
        pad_start = 0
        pad_end = 0
        for frame_num in range(start, end, stride):

            if frame_num < 0:
                pad_start += 1
                continue

            if pad_end > 0:
                pad_end += 1
                continue

            frame = frame_num
            frame_path = os.path.join(path, 'frame' + str(frame) + '.jpg')
            flow_frame_path = os.path.join(flow_path,str(frame_num) + '.npy')  # Assuming flow frames are PNG files
            base_path = path
            ndigits = -1
            
            exist_frame = os.path.exists(frame_path)
            exist_flow_frame = os.path.exists(flow_frame_path)

            if exist_frame & exist_flow_frame and found_start == -1:
                found_start = frame_num

            if not exist_frame or not exist_flow_frame:
                pad_end += 1

        ret = [base_path, found_start, pad_start, pad_end, ndigits, (end-start) // stride]
        
        return ret
    
    def load_frames(self, paths, pad=False, stride=1):
        base_path = paths[0]
        start = paths[1]
        pad_start = paths[2]
        pad_end = paths[3]
        ndigits = paths[4]
        length = paths[5]

        rgb_frames = []
        flow_frames = []

        for j in range(length - pad_start - pad_end):
            # Load RGB frames
            rgb_frame_path = os.path.join(base_path, 'frame' + str(start + j * stride) + '.jpg')
            flow_frame_path = os.path.join(base_path+'_flow', str(start + j * stride) + '.npy')  # Corresponding flow frame
            rgb_frames.append(self.read_frame(rgb_frame_path))
            flow_frames.append(self.read_flow(flow_frame_path))

        # Stack the frames into tensors
        rgb_frames = torch.stack(rgb_frames, dim=0)
        flow_frames = torch.stack(flow_frames, dim=0)

        # Pad the frames if necessary
        if pad_start > 0 or (pad and pad_end > 0):
            rgb_frames = torch.nn.functional.pad(rgb_frames, (0, 0, 0, 0, 0, 0, pad_start, pad_end if pad else 0))
            flow_frames = torch.nn.functional.pad(flow_frames, (0, 0, 0, 0, 0, 0, pad_start, pad_end if pad else 0))

        return rgb_frames, flow_frames
    
    def print_info(self):
        num_frames = sum([x['num_frames'] for x in self._games])
        print('{} : {} videos, {} frames ({} stride)'.format(
            self._src_file, len(self._games), num_frames, self._stride)
        )


def _print_info_helper(src_file, labels):
        num_frames = sum([x['num_frames'] for x in labels])
        print('{} : {} videos, {} frames'.format(
            src_file, len(labels), num_frames))