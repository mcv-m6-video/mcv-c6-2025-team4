import os
import cv2
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
from src import load_data, metrics, read_data
import numpy as np

# Argument configuration
parser = argparse.ArgumentParser(description="Evaluation of YOLO models for vehicle detection.")
parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model to evaluate (e.g., ./modelYolo/yolov8n.pt)")
args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths for input data and output
path = "./data/AICity_data/train/S03/c010"
output_dir = './output_videos'
path_annotation = "./data/ai_challenge_s03_c010-full_annotation.xml"
video_path = os.path.join(path, "vdo.avi")

# Get the total number of frames and split them into training and testing sets
total_frames = load_data.get_total_frames(video_path)
training_end = int(total_frames * 0.25)  # 25% of frames are used for training

# Load ground truth annotations from XML file
gt_data, _ = read_data.parse_annotations_xml(path_annotation, isGT=True)
gt_dict = {}
for item in gt_data:
    frame_no = item["frame"]
    box = item["bbox"][0].tolist() if isinstance(item["bbox"], np.ndarray) else item["bbox"]
    gt_dict.setdefault(frame_no, []).append(box)

# List of YOLO models to evaluate
yolo_models = [args.model]

# Open the video for processing
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, training_end)  # Start from the testing phase
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

for yolo_model_path in yolo_models:
    print(f"\nProcessing with {yolo_model_path}")
    model = YOLO(yolo_model_path).to(device)  # Load the YOLO model onto the selected device

    all_pred_boxes, all_gt_boxes = [], []

    # Initialize the VideoWriter to save the processed video
    out_frames = cv2.VideoWriter(os.path.join(output_dir, f"YOLO_{os.path.basename(yolo_model_path)}.avi"),
                                 cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size, isColor=True)

    # Process each frame from the starting test frame
    frame_idx = training_end  # Current frame index
    while cap.isOpened():
        ret, frame_rgb = cap.read()
        if not ret or frame_idx >= total_frames:
            break  # Stop if the video ends or test frames are exhausted

        with torch.no_grad():  # Disable gradient computation for efficiency
            image = Image.fromarray(frame_rgb)
            results = model(image)  # Run the YOLO model on the frame

            # Extract predicted bounding boxes for 'car' objects
            pred_boxes = [box.cpu().numpy() for result in results for box, cls in zip(result.boxes.xyxy, result.boxes.cls) if model.names[int(cls)] == 'car']
            gt_boxes = gt_dict.get(frame_idx, [])  # Get ground truth boxes for the frame

            # Store predictions and ground truth
            all_pred_boxes.append(pred_boxes)
            all_gt_boxes.append(gt_boxes)

            # Draw predicted bounding boxes in green
            for box in pred_boxes:
                cv2.rectangle(frame_rgb, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

            # Draw ground truth bounding boxes in blue
            for gt_box in gt_boxes:
                cv2.rectangle(frame_rgb, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (255, 0, 0), 2)

            # Save the processed frame in the output video
            out_frames.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        frame_idx += 1  # Move to the next frame

    # Compute Mean Average Precision (mAP) for the model at different IoU thresholds
    video_ap = metrics.compute_video_average_precision(all_pred_boxes, all_gt_boxes, iou_threshold=0.5)
    print(f"mAP50 for {yolo_model_path}: {video_ap:.4f}")
    video_ap = metrics.compute_video_average_precision(all_pred_boxes, all_gt_boxes, iou_threshold=0.75)
    print(f"mAP75 for {yolo_model_path}: {video_ap:.4f}")

    # Release the output video writer
    out_frames.release()

cap.release()
cv2.destroyAllWindows()

