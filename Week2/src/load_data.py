import cv2
import numpy as np


def get_total_frames(video_path):
    """
    Returns the total number of frames in the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total

def load_video_frame(video_path, index):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise IOError("Cannot open video file")

    # Set the index for the next frame to retrieve
    video.set(cv2.CAP_PROP_POS_FRAMES, index)


    # Read that frame
    ret, frame = video.read()

    # Check whether it was read
    if not ret:
        raise IOError("Cannot read video file")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Release video file
    video.release()
    return frame


def load_frames_list(video_path, start, end,color='rgb'):
    """
    Loads and returns a list of frames from the video starting at frame 'start'
    and ending at frame 'end' (end is exclusive).
    """
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise IOError("Cannot open video file")

    # Set the starting frame position
    video.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = []
    for frame_index in range(start, end):
        ret, frame = video.read()
        if not ret:
            # Stop if we reach the end of the video or encounter a read error
            break
        
        if color=='rgb':
            # Convert frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            _,_,v = cv2.split(hsvImage)
    
            live_value = np.mean(v)
            brightness_factor = 128 / live_value 
            hsvImage[...,2] = cv2.multiply(hsvImage[...,2],brightness_factor)
            frame_rgb = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2RGB)
            
        frames.append(frame_rgb)# Convert frame from BGR to RGB

    video.release()
    return frames

import cv2
import numpy as np
import random

def load_frames_kfold(video_path, total_frames, fold=0, color='rgb', random_sample=False):
    """
    Loads frames for K-fold validation, using 25% of the dataset as training and 75% as test.
    
    Parameters:
        video_path (str): Path to the video file.
        total_frames (int): Total number of frames in the video.
        fold (int): The fold index (0 to 3) to use as training.
        color (str): Color mode ('rgb' or 'hsv').
        random_sample (bool): If True, selects a random 25% of frames for training instead of sequential folding.
    
    Returns:
        train_frames (list): Frames for training.
        test_frames (list): Frames for testing.
    """
    if fold not in range(4):
        raise ValueError("Fold index must be between 0 and 3")
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise IOError("Cannot open video file")
    
    all_indices = list(range(total_frames))
    
    if random_sample:
        random.shuffle(all_indices)
        train_indices = set(all_indices[:int(total_frames * 0.25)])
    else:
        fold_size = int(np.round(total_frames * 0.25))
        train_start = fold * fold_size
        train_end = train_start + fold_size
        train_indices = set(range(train_start, train_end))
        # print(total_frames)
    train_frames, test_frames = [], []
    
    for frame_index in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break
        
        if color == 'rgb':
            frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            _, _, v = cv2.split(hsvImage)
            brightness_factor = 128 / np.mean(v)
            hsvImage[..., 2] = cv2.multiply(hsvImage[..., 2], brightness_factor)
            frame_processed = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2RGB)
        
        if frame_index in train_indices:
            train_frames.append(frame_processed)
        else:
            test_frames.append(frame_processed)
    
    video.release()
    return train_frames, test_frames



def load_bbx(annot_path):
    bbxs = []
    with open(annot_path, 'r') as annot:
        for line in annot:
            # Split line by commas
            bbx = line.strip().split(",")
            # Extract values 0, 3, 4, 5, and 6
            # [left, top, width, height, height]
            values = [int(bbx[0])] + [int(bbx[i]) for i in range(2, 6)]
            bbxs.append(values)
    return bbxs

