import cv2
import numpy as np
import torch
from torchvision import models, transforms
from src.sort import Sort
import torchvision.ops as ops
import json
import os
from tqdm import tqdm

# Select the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to save tracking results in MOTChallenge format
def save_mot_format(tracked_objects, output_mot_path):
    os.makedirs(os.path.dirname(output_mot_path), exist_ok=True)
    with open(output_mot_path, "w") as f:
        for frame_idx, objects in tracked_objects.items():
            if isinstance(objects, dict):
                # Handle case when objects are stored as a dictionary
                for obj_id, (box, _) in objects.items():
                    x1, y1, x2, y2 = map(int, box)
                    width, height = x2 - x1, y2 - y1
                    f.write(f"{frame_idx}, {obj_id}, {x1}, {y1}, {width}, {height}, 1, 1, 1\n")

            elif isinstance(objects, tuple) and len(objects) == 2:
                # Handle case when objects are stored as tuples
                box, _ = objects
                x1, y1, x2, y2 = map(int, box)
                width, height = x2 - x1, y2 - y1
                obj_id = frame_idx  # Use frame index as object ID if none is available
                f.write(f"{frame_idx}, {obj_id}, {x1}, {y1}, {width}, {height}, 1, 1, 1\n")

            else:
                print(f"Warning: Unhandled data structure at frame {frame_idx}: {objects}")


# Function to generate unique colors per track ID
def generate_color(track_id):
    np.random.seed(track_id)
    return tuple(np.random.randint(0, 255, 3).tolist())


# Function to load detections from a JSON file
def load_detections_from_json(file_path):
    try:
        with open(file_path, "r") as f:
            detections = json.load(f)
        return {int(k): np.array(v) for k, v in detections.items()}
    except FileNotFoundError:
        return None


# Function to save detections in JSON format
def save_detections_to_json(detections, file_path):
    with open(file_path, "w") as f:
        json.dump({k: [box.tolist() for box in v] for k, v in detections.items()}, f)


# Object detection function
def detect_objects(frame, model, framework='torch'):
    """
    Detect objects using a pre-trained model.

    Parameters:
    - frame: The video frame.
    - model: The pre-trained object detection model.
    - framework: 'torch' (default), currently the only supported framework.

    Returns:
    - boxes: List of bounding boxes detected in the frame.
    """
    original_height, original_width = frame.shape[:2]

    if framework == 'torch':
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

        # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
        keep = ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold=0.15)
        boxes = boxes[keep.numpy()]
        scores = scores[keep.numpy()]

        # Rescale boxes back to the original frame size
        scale_x = original_width / 800
        scale_y = original_height / 800

        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        return boxes

    else:
        raise ValueError(f"Unsupported framework: {framework}")


# Function to load the pre-trained model
def load_model(model_path, framework='torch'):
    if framework == 'torch':
        model = models.detection.fasterrcnn_resnet50_fpn(weights=None)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.eval()
        return model
    else:
        raise ValueError(f"Unsupported framework: {framework}")


# Video configuration
video_path = "./data/AICity_data/train/S03/c010/vdo.avi"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer for saving the processed video
output_path = './output_videos/task2_2rcnn.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Load the object detection model
model_path = 'faster_rcnn_resnet50_05_og.pth'
framework = 'torch'
model = load_model(model_path, framework)

# Initialize the tracking algorithm (SORT)
mot_tracker = Sort()

# Define frame selection
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample_rate = 1
selected_frames = range(0, frame_total, sample_rate)

# Load or initialize detections storage
detections_path = "./detections.json"
saved_detections = load_detections_from_json(detections_path)
if saved_detections:
    print("Using saved detections")
else:
    print("Generating new detections")
    saved_detections = {}

# Dictionary to store tracking results
tracked_dict = {}

# Process the video frame by frame
for frame_idx in tqdm(selected_frames, desc="Processing video"):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    if str(frame_idx) not in saved_detections:
        detections = detect_objects(frame, model, framework)
        saved_detections[str(frame_idx)] = detections
    else:
        detections = saved_detections[str(frame_idx)]

    # Save detections every 50 frames
    if frame_idx % 50 == 0:
        save_detections_to_json(saved_detections, detections_path)

    # Format detections for the SORT tracker
    if len(detections) > 0:
        dets = np.array([np.append(box, 1.0) for box in detections if len(box) == 4])
    else:
        dets = np.empty((0, 5))

    # Perform object tracking
    tracked_objects = mot_tracker.update(dets)

    # Store tracking results in the appropriate format
    frame_tracking = {}
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj.astype(int)
        frame_tracking[track_id] = ([x1, y1, x2, y2], frame_idx)

        color = generate_color(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save tracking data
    tracked_dict[frame_idx] = frame_tracking

    # Write processed frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()

# Save tracking results in MOTChallenge format
output_mot_path = "Week2/MOTS-train.txt"
save_mot_format(tracked_dict, output_mot_path)


