import cv2
import numpy as np
import torch
from torchvision import transforms
import sys
import os
from tqdm import tqdm
import json

# --------------------------
# 1. Set up the vehicle detector using Detectron2
# --------------------------
sys.path.append("/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/detectron2")
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

d2_cfg = get_cfg()
d2_cfg.merge_from_file(
    "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
d2_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Assume one class: vehicle
d2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
d2_cfg.MODEL.WEIGHTS = "Week4/model_final.pth"  # Update with your Detectron2 weights path
detector = DefaultPredictor(d2_cfg)

# --------------------------
# 2. Set up the FastReID re-identification module
# --------------------------
from fastreid.config import get_cfg as get_fr_cfg
from fastreid.engine.defaults import DefaultPredictor as FRPredictor

fr_cfg = get_fr_cfg()
fr_cfg.merge_from_file(
    "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/Week4/models/bagtricks_R50-ibn_vehicleid.yml")  # Update with your FastReID config file
fr_cfg.MODEL.WEIGHTS = "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/Week4/models/vehicleid_bot_R50-ibn.pth"  # Update with your FastReID weights path
fr_cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
fr_predictor = FRPredictor(fr_cfg)

# Define the preprocessing transform expected by FastReID
fr_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),  # Adjust if needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def extract_feature(crop):
    """
    Given a vehicle crop (BGR image), extract a normalized feature vector using FastReID.
    """
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    input_tensor = fr_transform(crop_rgb).unsqueeze(0).to(fr_cfg.MODEL.DEVICE)
    with torch.no_grad():
        features = fr_predictor.model(input_tensor)
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features.cpu().numpy().squeeze()


# --------------------------
# 3. Define chipper and embedder functions with crop saving capability
# --------------------------
def chipper(frame, detections, save_dir=None, frame_idx=None, camera_id="c015"):
    """
    Given a frame and a list of detections, return:
       - a list of object chips (crops)
       - a list of detection info dictionaries.
    Each detection is expected to be a tuple: ([left, top, w, h], confidence, detection_class).
    If 'save_dir' and 'frame_idx' are provided, the function saves each crop as an image
    and creates a field "crop_filename" using the pattern:
         frame_{frame_idx:06d}_det_{i:02d}_{camera_id}.jpg
    """
    chips = []
    detection_info = []  # List to hold per-detection dictionaries
    for i, det in enumerate(detections):
        bbox, conf, det_class = det
        left, top, w, h = map(int, bbox)
        chip = frame[top:top+h, left:left+w]
        chips.append(chip)
        crop_filename = None
        if save_dir is not None and frame_idx is not None:
            os.makedirs(save_dir, exist_ok=True)
            crop_filename = f"frame_{frame_idx:06d}_det_{i:02d}_{camera_id}.jpg"
            chip_path = os.path.join(save_dir, crop_filename)
            cv2.imwrite(chip_path, chip)
        # Convert bbox values, score, and frame to native Python types.
        detection_info.append({
            "bbox": [float(x) for x in bbox],
            "score": float(conf),
            "det_class": int(det_class),
            "crop_filename": crop_filename,
            "frame": int(frame_idx)
        })
    return chips, detection_info



def embedder(chips):
    """
    Given a list of object chips, compute and return a list of feature embeddings using extract_feature().
    """
    embeddings = []
    for chip in chips:
        if chip.size == 0:
            embeddings.append(np.zeros(512))  # Adjust dimension as needed
        else:
            embeddings.append(extract_feature(chip))
    return embeddings


# --------------------------
# 4. Define a function to filter detections by ROI overlap
# --------------------------
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

    if x2 <= x or y2 <= y:
        return False

    patch = roi_mask[y:y2, x:x2]
    inside = np.count_nonzero(patch)
    total = patch.size
    ratio = inside / float(total)
    return ratio >= min_ratio


# --------------------------
# 5. Set up DeepSORT using deep_sort_realtime (using external embeddings)
# --------------------------
from deep_sort_realtime.deepsort_tracker import DeepSort
tracker = DeepSort(max_age=5)  # Other parameters can be tuned as needed


# --------------------------
# 6. Function to save tracking results in MOT format (.txt)
# --------------------------
def save_mot_format(tracked_objects, output_mot_path):
    """
    Write tracking results in MOTChallenge format.
    tracked_objects is a dict: frame_idx -> {track_id: ([x1,y1,x2,y2], frame_idx)}
    """
    with open(output_mot_path, "w") as f:
        for frame_idx, objects in tracked_objects.items():
            for obj_id, (bbox, _) in objects.items():
                x1, y1, x2, y2 = map(int, bbox)
                width, height = x2 - x1, y2 - y1
                f.write(f"{frame_idx}, {obj_id}, {x1}, {y1}, {width}, {height}, 1, 1, 1\n")


# --------------------------
# 7. (Optional) Unique color generator for visualization
# --------------------------
track_colors = {}
def get_color(track_id):
    if track_id not in track_colors:
        color = np.random.randint(0, 256, size=3).tolist()  # BGR color
        track_colors[track_id] = color
    return track_colors[track_id]


# --------------------------
# 8. Main tracking loop with ROI filtering, crop saving, and detection metadata collection
# --------------------------
if __name__ == "__main__":
    video_path = "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/data/aic19-track1-mtmc-train/train/S04/c040/vdo.avi"
    roi_path = "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/data/aic19-track1-mtmc-train/train/S04/c040/roi.jpg"

    # Directory to save original detection crops (raw crops without drawn boxes)
    crops_save_dir = "Week4/detection_crops"

    # File to save detection metadata (per detection info including crop_filename)
    detection_metadata_file = "Week4/results/s04_c040_detection_metadata.json"
    all_detection_metadata = {}  # Dictionary keyed by frame index

    # Load ROI as a single-channel mask
    roi_mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
    if roi_mask is None:
        print("Error: Could not load ROI mask from:", roi_path)
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_folder = "Week4/video_frames/s04_c040_roi"
    os.makedirs(output_folder, exist_ok=True)

    tracked_dict = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        outputs = detector(frame)
        instances = outputs["instances"]
        detections = []
        if instances.has("pred_boxes"):
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], score, 0))

        filtered_detections = []
        for det in detections:
            bbox, conf, cls_ = det
            if in_roi(bbox, roi_mask, min_ratio=0.5):
                filtered_detections.append(det)

        # If no valid detections, save the frame as-is and continue.
        if len(filtered_detections) == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_idx += 1
            print(f"Frame {frame_idx} - No detections after ROI filtering.")
            continue

        # Extract chips and also collect detection metadata (including crop filenames)
        object_chips, detection_info = chipper(frame, filtered_detections, save_dir=crops_save_dir, frame_idx=frame_idx, camera_id="c015")
        # Save the detection info for this frame in our global dictionary.
        all_detection_metadata[frame_idx] = detection_info

        embeds = embedder(object_chips)

        tracks = tracker.update_tracks(filtered_detections, embeds=embeds)
        frame_tracking = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            frame_tracking[track_id] = ([x1, y1, x2, y2], frame_idx)
            color = get_color(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        tracked_dict[frame_idx] = frame_tracking

        frame_filename = os.path.join(output_folder, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_idx += 1
        print(f"Procesado frame {frame_idx} con {len(filtered_detections)} detecciones válidas.")

    cap.release()
    saved_frames = os.listdir(output_folder)
    print(f"Se han guardado {len(saved_frames)} frames en la carpeta '{output_folder}'.")

    mot_output_path = "Week4/results/s04_c040_roi.txt"
    save_mot_format(tracked_dict, mot_output_path)
    print("Tracking completado con filtrado por ROI. Frames, resultados MOT y crops guardados.")

    # Save the detection metadata to a JSON file for later use in multi-camera matching.
    os.makedirs(os.path.dirname(detection_metadata_file), exist_ok=True)
    with open(detection_metadata_file, "w") as f:
        json.dump(all_detection_metadata, f, indent=4)
    print(f"Detection metadata saved to {detection_metadata_file}.")



