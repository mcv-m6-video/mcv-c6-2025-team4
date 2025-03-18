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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device for optical flow:", device)

# -------------------------------pip
# Initialize FlowFormer model (optical flow) on CUDA with FP16
FLOWFORMER_MODEL_PATH = 'sintel.pth'
cfg = get_things_cfg()
cfg.model = FLOWFORMER_MODEL_PATH
flow_model = torch.nn.DataParallel(build_flowformer(cfg))
flow_model.load_state_dict(torch.load(cfg.model, map_location=device))
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
# Compute optical flow using FlowFormer (runs on CUDA, FP16)
def compute_optical_flow_lowres(pred_frame, next_frame, model, device, target_size=(640, 360)):
    """
    Resize the input frames to a fixed target resolution (e.g., 640x360),
    compute optical flow using FlowFormer (in float32 on CUDA),
    and return the flow in the target resolution coordinate space.
    """
    # Convert BGR frames to RGB and normalize to [0,1] in float32
    pred_rgb = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    next_rgb = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Resize images to target resolution
    target_w, target_h = target_size
    pred_resized = cv2.resize(pred_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    next_resized = cv2.resize(next_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Convert resized images to torch tensors with shape (1, 3, H, W)
    pred_tensor = torch.from_numpy(pred_resized.transpose(2, 0, 1)).unsqueeze(0).to(device)
    next_tensor = torch.from_numpy(next_resized.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # Pad the tensors (model requirement)
    padder = InputPadder(pred_tensor.shape)
    pred_tensor, next_tensor = padder.pad(pred_tensor, next_tensor)

    # Run inference with FlowFormer in default float32 precision
    with torch.no_grad():
        flow_output = model(pred_tensor, next_tensor)

    # Unpad the flow output and convert to numpy array (target resolution flow)
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
        # Calcular el centro de la caja
        cx, cy = np.round([(x1 + x2) / 2, (y1 + y2) / 2]).astype(int)
        h, w = flow.shape[:2]
        cx = np.clip(cx, 0, w - 1)
        cy = np.clip(cy, 0, h - 1)
        # Desplazamiento basado en el flujo en el centro de la caja
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

def get_color(track_id):
    # If track_id is not in the dictionary, assign a random color
    if track_id not in track_colors:
        # Generate a random BGR color (values 0-255)
        color = np.random.randint(0, 256, size=3).tolist()
        track_colors[track_id] = color
    return track_colors[track_id]

def in_roi(bbox, roi_mask, min_ratio=0.5):
    """
    Check if a bounding box has at least min_ratio overlap with the ROI mask.
    bbox: [x, y, w, h]
    roi_mask: Single-channel (grayscale) mask where 255 = ROI, 0 = outside.
    min_ratio: Minimum fraction of bbox area that must be inside the ROI to keep it.
    """
    x, y, w, h = map(int, bbox)
    x2, y2 = x + w, y + h

    # Clamp coords to image boundaries
    x = max(x, 0)
    y = max(y, 0)
    x2 = min(x2, roi_mask.shape[1] - 1)
    y2 = min(y2, roi_mask.shape[0] - 1)

    # If invalid region or no overlap
    if x2 <= x or y2 <= y:
        return False

    # Extract the corresponding patch from the ROI
    patch = roi_mask[y:y2, x:x2]
    # Count how many pixels are non-zero (inside ROI)
    inside = np.count_nonzero(patch)
    total = patch.size
    ratio = inside / float(total)
    return ratio >= min_ratio


def load_start_times(txt_file, fps=10):
    start_frames = {}
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                video_id, time_seconds = parts[0], float(parts[1])
                start_frames[video_id] = -int(time_seconds * fps)  # Convert time to frames
    return start_frames



seq='S04/'
videos=['c016','c017']

start_frames_dict = load_start_times("E:/aic19-track1-mtmc-train/cam_timestamp/"+seq.split('/')[0]+'.txt')

mot_tracker=Sort()
for vid in videos:

    # -------------------------------
    # Main video processing configuration
    # video_path = "E:/aic19-track1-mtmc-train/train/"+seq+vid+"/vdo.avi"
    # cap = cv2.VideoCapture(video_path)
    # if not cap.isOpened():
    #     print("Error: Unable to open video.")
    #     exit()

    gt_boxes = load_ground_truth('E:/aic19-track1-mtmc-train/train/'+seq+vid+'/gt/gt.txt')
    roi_mask = cv2.imread('E:/aic19-track1-mtmc-train/train/'+seq+vid+'/roi.jpg', cv2.IMREAD_GRAYSCALE)
    # print(roi_mask)
    frame_path="E:/aic19-track1-mtmc-train/train/"+seq+vid+"/frames"
    frame=cv2.imread(frame_path+"/frame_000000.jpg")
    fps = 10
    frame_width = int(np.shape(frame)[1])
    frame_height = int(np.shape(frame)[0])

    # Inicializar escritor de video
    output_path = vid+'.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Inicializar el algoritmo SORT (ya implementado con Kalman Filter)
    del mot_tracker
    mot_tracker = Sort()

    frame_total = int(len(os.listdir(frame_path)))
    sample_rate = 1
    
    selected_frames = range(0, frame_total, sample_rate)

    cfg1 = get_cfg()
    cfg1.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg1.MODEL.WEIGHTS = './output/model_final.pth'
    cfg1.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Set correct number of classes
    cfg1.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg1)

    # Diccionario para almacenar los resultados del seguimiento
    tracked_dict = {}

    # Leer el primer fotograma
    # prev_frame=cv2.imread(frame_path+"/frame_000000.jpg")


    track_colors = {}



    # Procesar los fotogramas
    for frame_idx in tqdm(selected_frames, desc="Processing video"):
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        # ret, frame = cap.read()
        # if not ret:
        #     break

        frame=cv2.imread(frame_path+"/frame_"+str(frame_idx).zfill(6)+".jpg")

        # original_h, original_w = frame.shape[:2]
        # target_w, target_h = (640, 360)

        # scale_down_x = target_w / original_w
        # scale_down_y = target_h / original_h
        # scale_up_x = original_w / target_w
        # scale_up_y = original_h / target_h

        # Detectar objetos usando Detectron2
        outputs = predictor(frame)
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

        # Skip update if no detections
        if boxes.shape[0] == 0:
            print(f"No detections at frame {frame_idx}, skipping tracking update.")
            # prev_frame = frame
            out.write(frame)
            continue

        if frame_idx > 0:
            # If you want to update boxes using optical flow, you can compute it here.
            # For now, we're directly using the boxes as predicted_boxes.
            predicted_boxes = []
            for box in boxes:
                if in_roi(box, roi_mask, min_ratio=0.5):
                    predicted_boxes.append(box)

            predicted_boxes = np.array(predicted_boxes)

            # Update SORT tracker with predicted boxes
            # print(np.shape(predicted_boxes))
            if len(predicted_boxes) > 0:

                tracked_objects = mot_tracker.update(np.array([np.append(box, 1.0) for box in predicted_boxes]))
            else :
                tracked_objects=[]

            frame_tracking = {}
            for obj in tracked_objects:
                x1, y1, x2, y2, track_id = obj.astype(int)
                frame_tracking[track_id] = ([x1, y1, x2, y2], frame_idx)
                # Get unique color for this track_id
                color = get_color(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            tracked_dict[frame_idx] = frame_tracking

        # Optionally, draw ground truth boxes if available
        if frame_idx in gt_boxes:
            for gt in gt_boxes[frame_idx]:
                _, x, y, w, h = gt
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # prev_frame = frame
        out.write(frame)
        torch.cuda.empty_cache()

    # Guardar resultados en formato MOTChallenge
    output_mot_path = vid+".txt"
    save_mot_format(tracked_dict, output_mot_path)

    out.release()
