import cv2
import numpy as np
import tensorflow as tf
import torch
from torchvision import models, transforms
from src import metrics
from torchvision.models.detection import (
    retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights,
    ssd300_vgg16, SSD300_VGG16_Weights,
    ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
)
import torchvision.ops as ops
from tqdm import tqdm
import json
import os

# Device selection (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
next_id = 0  # ID counter for tracking objects


# Function to save tracking results in MOT format
def save_mot_format(tracked_objects, output_mot_path):
    os.makedirs(os.path.dirname(output_mot_path), exist_ok=True)
    with open(output_mot_path, "w") as f:
        for frame_idx, objects in tracked_objects.items():
            if isinstance(objects, dict):
                # Handling case when objects are stored as a dictionary
                for obj_id, (box, _) in objects.items():
                    x1, y1, x2, y2 = map(int, box)
                    width, height = x2 - x1, y2 - y1
                    f.write(f"{frame_idx}, {obj_id}, {x1}, {y1}, {width}, {height}, 1, 1, 1\n")

            elif isinstance(objects, tuple) and len(objects) == 2:
                # Handling case when objects are stored as tuples
                box, _ = objects  # Extract bounding box
                x1, y1, x2, y2 = map(int, box)
                width, height = x2 - x1, y2 - y1
                obj_id = frame_idx  # Use frame index as object ID if not available
                f.write(f"{frame_idx}, {obj_id}, {x1}, {y1}, {width}, {height}, 1, 1, 1\n")

            else:
                print(f"Warning: Unhandled data structure at frame {frame_idx}: {objects}")


# Function to save detections in JSON format
def save_detections_to_json(detections, file_path):
    with open(file_path, "w") as f:
        json.dump({k: [box.tolist() for box in v] for k, v in detections.items()}, f)


# Function to load detections from a JSON file
def load_detections_from_json(file_path):
    try:
        with open(file_path, "r") as f:
            detections = json.load(f)
        return {int(k): np.array(v) for k, v in detections.items()}
    except FileNotFoundError:
        return None


# Function to track objects between frames
def track_objects(boxes, frame_number):
    global next_id
    updated_objects = {}
    for obj_id, (prev_box, last_seen) in tracked_objects.items():
        if frame_number - last_seen > 10:  # Remove objects missing for over 10 frames
            continue

        best_iou = 0
        best_match = None
        for i, box in enumerate(boxes):
            current_iou = metrics.compute_iou(prev_box, box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_match = (i, box)

        if best_iou > 0.5:  # Assign objects with IoU > 0.5 to the same track
            updated_objects[obj_id] = (best_match[1], frame_number)
            del boxes[best_match[0]]

    # Assign new IDs to newly detected objects
    for box in boxes:
        updated_objects[next_id] = (box, frame_number)
        next_id += 1

    return updated_objects


# Function for object detection using a chosen model
def detect_objects(frame, model, framework='tensorflow'):
    """
    Detect objects using the specified model.

    Parameters:
    - frame: The video frame.
    - model: The pre-trained object detection model.
    - framework: 'tensorflow', 'torch', or 'opencv'.

    Returns:
    - boxes: List of detected bounding boxes.
    """
    if framework == 'torch':
        # Preprocess the frame for PyTorch models
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_frame = transform(frame).unsqueeze(0)
        model.eval()

        with torch.no_grad():
            predictions = model(input_frame)

        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        # Filter only car detections (assuming class 1 corresponds to cars)
        car_indices = labels == 1
        boxes = boxes[car_indices]
        scores = scores[car_indices]

        # Apply Non-Maximum Suppression (NMS) to reduce overlapping detections
        keep = ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold=0.15)
        boxes = boxes[keep.numpy()]
        scores = scores[keep.numpy()]

        return boxes

    elif framework == 'opencv':
        # Preprocess image for OpenCV's Mask R-CNN
        blob = cv2.dnn.blobFromImage(frame, 1.0, (800, 800), (0, 0, 0), swapRB=True, crop=False)
        model.setInput(blob)
        output = model.forward(["detection_out_final", "detection_masks"])

        # Extract bounding boxes
        boxes = []
        for i in range(output[0].shape[2]):
            confidence = output[0][0, 0, i, 2]
            if confidence > 0.5:
                x1 = int(output[0][0, 0, i, 3] * frame.shape[1])
                y1 = int(output[0][0, 0, i, 4] * frame.shape[0])
                x2 = int(output[0][0, 0, i, 5] * frame.shape[1])
                y2 = int(output[0][0, 0, i, 6] * frame.shape[0])
                boxes.append([x1, y1, x2, y2])

        return boxes

    else:
        raise ValueError(f"Framework '{framework}' not supported.")


# Load a pre-trained model based on the selected framework
def load_model(model_path, framework='tensorflow'):
    if framework == 'tensorflow':
        return tf.saved_model.load(model_path)
    elif framework == 'torch':
        model = fasterrcnn_resnet50_fpn(weights=None)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    elif framework == 'opencv':
        return cv2.dnn.readNetFromTensorflow(model_path)
    else:
        raise ValueError(f"Framework '{framework}' not supported.")


# Generate a consistent color for each track ID
def generate_color(track_id):
    np.random.seed(track_id)
    return tuple(np.random.randint(0, 255, 3).tolist())


# Load the video
video_path = "./data/AICity_data/train/S03/c010/vdo.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Video could not be opened")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter for output
output_path = './output_videos/task2_1rcnn.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Load the object detection model
model_path = 'faster_rcnn_resnet50_05_og.pth'
framework = 'torch'
model = load_model(model_path, framework)

# Process frames, detect and track objects
tracked_objects = {}
frame_count = 0

# Get the total number of frames in the video
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample_rate = 1  # Process every frame
selected_frames = range(0, frame_total, sample_rate)

# Load or initialize detections storage
detections_path = "./Week2/detections.json"
saved_detections = load_detections_from_json(detections_path)
if saved_detections:
    print("Using saved detections")
else:
    print("Generating new detections")
    saved_detections = {}

# Dictionary to store tracked objects
tracked_dict = {}

# Process video frames
for frame_idx in tqdm(selected_frames, desc="Processing video"):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Set video position
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if the frame cannot be read

    # Detect objects in the current frame
    if str(frame_idx) not in saved_detections:
        boxes = detect_objects(frame, model, framework)
        saved_detections[str(frame_idx)] = boxes
    else:
        boxes = saved_detections[str(frame_idx)]

    # Save detections every 50 frames
    if frame_idx % 50 == 0:
        save_detections_to_json(saved_detections, detections_path)

    # Perform object tracking
    tracked_objects = track_objects(list(boxes), frame_idx)
    tracked_dict[frame_idx] = tracked_objects  # Store tracked objects per frame

    # Draw bounding boxes and object IDs on the frame
    for obj_id, (box, _) in tracked_objects.items():
        x1, y1, x2, y2 = map(int, box)

        # Generate a unique color for each track_id
        color = generate_color(obj_id)

        # Draw the bounding box with the assigned color
        if x1 is None or y1 is None or x2 is None or y2 is None:
            print(f"Error: Invalid values ({x1}, {y1}, {x2}, {y2})")
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Add text label with the track ID above the rectangle
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Write the processed frame to the output video file
    out.write(frame)

    # Update frame count
    frame_count += 1

# Release video resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Verify the tracked dictionary structure before saving
if not isinstance(tracked_dict, dict):
    print("Error: tracked_objects is not a dictionary!", type(tracked_objects))
else:
    # Save tracking results in MOTChallenge format
    output_mot_path = "Week2/MOTS-train-1.txt"
    save_mot_format(tracked_dict, output_mot_path)
