import os

# Set environment variable to allow expandable segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys

sys.path.append('core')

import cv2
import numpy as np
import torch
from torchvision import models, transforms
from src.sort import Sort  # Ya tienes implementado el algoritmo SORT
import torchvision.ops as ops
import json
from tqdm import tqdm

# -------------------------------
# FlowFormer Imports and Setup
from FlowFormer.configs.things_eval import get_cfg as get_things_cfg
from FlowFormer.core.FlowFormer import build_flowformer
from FlowFormer.core.utils.utils import InputPadder

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch


# Device selection for optical flow
# Device selection for optical flow (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device for optical flow:", device)

# -------------------------------
# Initialize FlowFormer model (optical flow) on CUDA (default float32)
FLOWFORMER_MODEL_PATH = '/home/toukapy/Documentos/Master CV/C6/mcv-c6-2025-team4/Week3/sintel.pth'
cfg_flow = get_things_cfg()
cfg_flow.model = FLOWFORMER_MODEL_PATH
flow_model = torch.nn.DataParallel(build_flowformer(cfg_flow))
flow_model.load_state_dict(torch.load(cfg_flow.model, map_location=device))
flow_model = flow_model.to(device)
flow_model.eval()

# -------------------------------
# JSON I/O functions for detections and MOT format
def load_detections_from_json(file_path):
    try:
        with open(file_path, "r") as f:
            detections = json.load(f)
        return {str(k): np.array(v) for k, v in detections.items()}  # Mantener claves como str
    except FileNotFoundError:
        return {}

def save_detections_to_json(detections, file_path):
    with open(file_path, "w") as f:
        json.dump({k: [box.tolist() for box in v] for k, v in detections.items()}, f)

def save_mot_format(tracked_objects, output_mot_path):
    output_dir = os.path.dirname(output_mot_path)
    if output_dir:
        os.makedirs(os.path.dirname(output_mot_path), exist_ok=True)
    with open(output_mot_path, "w") as f:
        for frame_idx, objects in tracked_objects.items():
            if isinstance(objects, dict):
                for obj_id, (box, _) in objects.items():
                    x1, y1, x2, y2 = map(int, box)
                    width, height = x2 - x1, y2 - y1
                    f.write(f"{frame_idx}, {obj_id}, {x1}, {y1}, {width}, {height}, 1, 1, 1\n")
            elif isinstance(objects, tuple) and len(objects) == 2:
                box, _ = objects
                x1, y1, x2, y2 = map(int, box)
                width, height = x2 - x1, y2 - y1
                obj_id = frame_idx  # Si no se tiene ID, usar el frame como ID
                f.write(f"{frame_idx}, {obj_id}, {x1}, {y1}, {width}, {height}, 1, 1, 1\n")
            else:
                print(f"Warning: Unhandled data structure at frame {frame_idx}: {objects}")

# -------------------------------
# Compute optical flow at low resolution
def compute_optical_flow_lowres(pred_frame, next_frame, model, device, target_size=(640, 360)):
    """
    Resize the input frames to a fixed target resolution (e.g., 640x360),
    compute optical flow using FlowFormer (in float32 on CUDA),
    and return the flow in the target resolution coordinate space.
    """
    # Convert BGR to RGB and normalize to [0,1]
    pred_rgb = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    next_rgb = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Resize to target resolution
    target_w, target_h = target_size
    pred_resized = cv2.resize(pred_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    next_resized = cv2.resize(next_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Convert resized images to torch tensors with shape (1,3,H,W)
    pred_tensor = torch.from_numpy(pred_resized.transpose(2, 0, 1)).unsqueeze(0).to(device)
    next_tensor = torch.from_numpy(next_resized.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # Pad tensors as required by FlowFormer
    padder = InputPadder(pred_tensor.shape)
    pred_tensor, next_tensor = padder.pad(pred_tensor, next_tensor)

    # Run inference (float32)
    with torch.no_grad():
        flow_output = model(pred_tensor, next_tensor)

    # Unpad and convert to numpy (flow in target resolution)
    flow_tensor = padder.unpad(flow_output[0]).cpu()
    flow_lowres = flow_tensor[0].numpy().transpose(1, 2, 0)
    torch.cuda.empty_cache()
    return flow_lowres

# -------------------------------
# Predict positions for bounding boxes using the computed flow
def predict_positions_with_of(flow, boxes):
    predictions = []
    for box in boxes:
        x1, y1, x2, y2 = box
        # Compute the center of the box
        cx, cy = np.round([(x1 + x2) / 2, (y1 + y2) / 2]).astype(int)
        h, w = flow.shape[:2]
        cx = np.clip(cx, 0, w - 1)
        cy = np.clip(cy, 0, h - 1)
        # Use flow at the center to update the box position
        dx, dy = flow[cy, cx]
        new_x1 = x1 + dx
        new_y1 = y1 + dy
        new_x2 = x2 + dx
        new_y2 = y2 + dy
        predictions.append([new_x1, new_y1, new_x2, new_y2])
    return np.array(predictions)

def load_ground_truth(file_path):
    gt_data = {}
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            frame, track_id, x, y, w, h = map(int, parts[:6])
            if frame not in gt_data:
                gt_data[frame] = []
            gt_data[frame].append([track_id, x, y, w, h])
    return gt_data

# -------------------------------
# Unique color generator for track IDs
track_colors = {}
def get_color(track_id):
    if track_id not in track_colors:
        color = np.random.randint(0, 256, size=3).tolist()  # BGR
        track_colors[track_id] = color
    return track_colors[track_id]

# -------------------------------
# Main video processing configuration
video_path = "/home/toukapy/Documentos/Master CV/C6/mcv-c6-2025-team4/data/aic19-track1-mtmc-train/train/S01/c003/vdo.avi"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

gt_boxes = load_ground_truth('/home/toukapy/Documentos/Master CV/C6/mcv-c6-2025-team4/data/aic19-track1-mtmc-train/train/S01/c003/gt/gt.txt')

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer for tracking output
output_path = 'good_005_with.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Initialize video writer for flow visualization output
flow_output_path = 'flow_visualization.mp4'
flow_out = cv2.VideoWriter(flow_output_path, fourcc, fps, (frame_width, frame_height))

# Initialize SORT tracker
mot_tracker = Sort()

frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample_rate = 1
selected_frames = range(0, frame_total, sample_rate)

# Setup Detectron2 predictor for object detection (on CPU)
cfg_det = get_cfg()
cfg_det.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg_det.MODEL.WEIGHTS = '/home/toukapy/Documentos/Master CV/C6/mcv-c6-2025-team4/Week3/model_final_bad.pth'
cfg_det.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Adjust for your dataset
cfg_det.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg_det)

# Dictionary for storing tracking results
tracked_dict = {}

# Read the first frame
ret, prev_frame = cap.read()

# Process frames
for frame_idx in tqdm(selected_frames, desc="Processing video"):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break

    original_h, original_w = frame.shape[:2]
    target_w, target_h = (640, 360)

    # Compute scale factors for coordinate conversion
    scale_down_x = target_w / original_w
    scale_down_y = target_h / original_h
    scale_up_x = original_w / target_w
    scale_up_y = original_h / target_h

    # Detect objects using Detectron2 (running on CPU)
    outputs = predictor(frame)
    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

    # If no detections, skip update
    if boxes.shape[0] == 0:
        print(f"No detections at frame {frame_idx}, skipping tracking update.")
        prev_frame = frame
        out.write(frame)
        flow_blank = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        flow_out.write(flow_blank)
        continue

    if frame_idx > 0:
        # Compute low-res optical flow between prev_frame and current frame
        flow_lowres = compute_optical_flow_lowres(prev_frame, frame, flow_model, device, target_size=(target_w, target_h))

        # Generate flow visualization (low-res to HSV then upscaled)
        hsv = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        hsv[..., 1] = 255  # Full saturation
        mag, ang = cv2.cartToPolar(flow_lowres[..., 0], flow_lowres[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_vis_lowres = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        flow_vis = cv2.resize(flow_vis_lowres, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        flow_out.write(flow_vis)

        # Update detections using the low-res optical flow:
        # Scale detection centers to low-res space, get displacement, then scale displacement back.
        predicted_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = int(((x1 + x2) / 2) * scale_down_x)
            cy = int(((y1 + y2) / 2) * scale_down_y)
            cx = np.clip(cx, 0, flow_lowres.shape[1] - 1)
            cy = np.clip(cy, 0, flow_lowres.shape[0] - 1)
            dx, dy = flow_lowres[cy, cx]
            dx *= scale_up_x
            dy *= scale_up_y
            new_x1 = x1 + dx
            new_y1 = y1 + dy
            new_x2 = x2 + dx
            new_y2 = y2 + dy
            predicted_boxes.append([new_x1, new_y1, new_x2, new_y2])
        predicted_boxes = np.array(predicted_boxes)

        # Update SORT tracker with predicted boxes
        tracked_objects = mot_tracker.update(np.array([np.append(box, 1.0) for box in predicted_boxes]))
        frame_tracking = {}
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj.astype(int)
            frame_tracking[track_id] = ([x1, y1, x2, y2], frame_idx)
            # Draw box with unique color for each track_id
            color = get_color(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        tracked_dict[frame_idx] = frame_tracking

    # Optionally, overlay ground truth boxes (in blue)
    if frame_idx in gt_boxes:
        for gt in gt_boxes[frame_idx]:
            _, x, y, w, h = gt
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    prev_frame = frame
    out.write(frame)
    torch.cuda.empty_cache()

# Save tracking results in MOT format
output_mot_path = "good_c005_with.txt"
save_mot_format(tracked_dict, output_mot_path)

cap.release()
out.release()
flow_out.release()