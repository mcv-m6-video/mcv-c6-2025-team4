import os
import sys
import cv2
import numpy as np
import torch
from torchvision import transforms
from itertools import combinations
from numpy.linalg import norm
from tqdm import tqdm

# =============================================================================
# 1. Detector and Re‑ID Setup (DeepSORT’s custom re‑ID is used inside the tracker)
# =============================================================================

# Setup Detectron2 for vehicle detection
sys.path.append("/home/toukapy/Dokumentuak/Master CV/C6/detectron2")
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

d2_cfg = get_cfg()
d2_cfg.merge_from_file("/home/toukapy/Dokumentuak/Master CV/C6/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
d2_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # one class: vehicle
d2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
d2_cfg.MODEL.WEIGHTS = "Week4/model_final.pth"  # Update with your weights
detector = DefaultPredictor(d2_cfg)

# Setup FastReID for re-identification (used for feature extraction)
from fastreid.config import get_cfg as get_fr_cfg
from fastreid.engine.defaults import DefaultPredictor as FRPredictor

fr_cfg = get_fr_cfg()
fr_cfg.merge_from_file("/home/toukapy/Dokumentuak/Master CV/C6/Week4/models/bagtricks_R50-ibn_vehicleid.yml")
fr_cfg.MODEL.WEIGHTS = "/home/toukapy/Dokumentuak/Master CV/C6/Week4/models/vehicleid_bot_R50-ibn.pth"
fr_cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
fr_predictor = FRPredictor(fr_cfg)

fr_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_feature(crop):
    """
    Extracts a normalized feature vector from a vehicle crop (BGR) using FastReID.
    """
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    input_tensor = fr_transform(crop_rgb).unsqueeze(0).to(fr_cfg.MODEL.DEVICE)
    with torch.no_grad():
        features = fr_predictor.model(input_tensor)
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features.cpu().numpy().squeeze()


# =============================================================================
# 2. DeepSORT Tracker Setup
# =============================================================================
from deep_sort_realtime.deepsort_tracker import DeepSort
# We assume the deep_sort_realtime tracker calls your custom re‑ID (via your detector+extract_feature)
tracker = DeepSort(max_age=5)

# =============================================================================
# 3. Functions for Detection, ROI Filtering, and Embedding
# =============================================================================
def chipper(frame, detections):
    """
    Given a frame and a list of detections (each a tuple: ([x, y, w, h], score, class)),
    returns the cropped chips.
    """
    chips = []
    for det in detections:
        bbox, score, cls = det
        left, top, w, h = map(int, bbox)
        chip = frame[top:top+h, left:left+w]
        chips.append(chip)
    return chips

def embedder(chips):
    """
    For each chip, extract the feature using extract_feature.
    Returns a list of embeddings.
    """
    embeddings = []
    for chip in chips:
        if chip.size == 0:
            embeddings.append(np.zeros(512))
        else:
            embeddings.append(extract_feature(chip))
    return embeddings

def in_roi(bbox, roi_mask, min_ratio=0.5):
    """
    Checks if bbox ([x,y,w,h]) has at least min_ratio overlap with the ROI mask.
    """
    x, y, w, h = map(int, bbox)
    x2, y2 = x+w, y+h
    x = max(x, 0)
    y = max(y, 0)
    x2 = min(x2, roi_mask.shape[1]-1)
    y2 = min(y2, roi_mask.shape[0]-1)
    if x2 <= x or y2 <= y:
        return False
    patch = roi_mask[y:y2, x:x2]
    inside = np.count_nonzero(patch)
    total = patch.size
    return (inside / float(total)) >= min_ratio

# =============================================================================
# 4. Functions to load tracking results from MOT files and aggregate tracklets
# =============================================================================
def load_tracking_results(mot_file):
    """
    Reads a MOT file (frame, id, x, y, w, h, ...)
    Returns a dict: frame -> list of detections.
    Each detection is {'track_id': id, 'bbox': [x, y, w, h]}.
    """
    tracking = {}
    with open(mot_file, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            frame = int(parts[0].strip())
            tid = int(parts[1].strip())
            x = float(parts[2].strip())
            y = float(parts[3].strip())
            w = float(parts[4].strip())
            h = float(parts[5].strip())
            bbox = [x, y, w, h]
            if frame not in tracking:
                tracking[frame] = []
            tracking[frame].append({'track_id': tid, 'bbox': bbox})
    return tracking

def aggregate_tracklets(tracking, fps, start_time):
    """
    Groups detections by track_id and computes for each tracklet:
      - frames list, centers list, start_time, end_time, and average center.
    Returns a dict: track_id -> { 'frames': [...], 'centers': [...],
                                  'start_time': t_ini, 'end_time': t_fin,
                                  'avg_center': np.array([x, y]) }
    """
    tracklets = {}
    for frame, dets in tracking.items():
        for det in dets:
            tid = det['track_id']
            bbox = det['bbox']
            center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
            if tid not in tracklets:
                tracklets[tid] = {'frames': [], 'centers': []}
            tracklets[tid]['frames'].append(frame)
            tracklets[tid]['centers'].append(center)
    for tid, data in tracklets.items():
        frames_arr = np.array(data['frames'])
        data['start_time'] = start_time + frames_arr.min() / fps
        data['end_time'] = start_time + frames_arr.max() / fps
        data['avg_center'] = np.mean(data['centers'], axis=0)
    return tracklets

# =============================================================================
# 5. Functions for Timestamps, Calibration, and Projection
# =============================================================================
def load_timestamps(timestamp_file):
    timestamps = {}
    with open(timestamp_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                cam_id = parts[0]
                ts = float(parts[1])
                timestamps[cam_id] = ts
    return timestamps

def load_calibration(calib_file):
    with open(calib_file, "r") as f:
        line = f.readline().strip()
    rows = [r.strip() for r in line.split(";")]
    H = []
    for row in rows:
        vals = [float(v) for v in row.split()]
        H.append(vals)
    return np.array(H)

def project_to_local(point, H, ref_gps):
    """
    Transforms an image point to local coordinates by applying the inverse homography
    (which maps from GPS to image) and subtracting the reference GPS.
    """
    H_inv = np.linalg.inv(H)
    p = np.array([point[0], point[1], 1.0])
    gps = H_inv.dot(p)
    gps /= gps[2]
    local = gps[:2] - np.array(ref_gps)
    return local

# =============================================================================
# 6. Association Using Appearance Features from DeepSORT
# =============================================================================
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2) + 1e-6)

def associate_tracklets_with_appearance(tracklets_cam1, tracklets_cam2, H1, H2, ref_gps,
                                        appearance_feats1, appearance_feats2,
                                        min_time_gap=1, time_tol=60, spatial_tol=50, appearance_thresh=0.5):
    """
    Associates tracklets between two cameras based on:
      - Temporal gap between the end of a tracklet in cam1 and the start in cam2.
      - Spatial distance in local coordinates.
      - Appearance similarity (cosine similarity of deepSORT features).
    appearance_featsX is a dict: track_id -> feature vector.
    Returns a list of associations: (tid_cam1, tid_cam2)
    """
    associations = []
    for tid1, data1 in tracklets_cam1.items():
        center1_local = project_to_local(data1['avg_center'], H1, ref_gps)
        feat1 = appearance_feats1.get(tid1, None)
        for tid2, data2 in tracklets_cam2.items():
            dt = data2['start_time'] - data1['end_time']
            if dt < min_time_gap or dt > time_tol:
                continue
            center2_local = project_to_local(data2['avg_center'], H2, ref_gps)
            spatial_dist = np.linalg.norm(center1_local - center2_local)
            if spatial_dist >= spatial_tol:
                continue
            feat2 = appearance_feats2.get(tid2, None)
            if feat1 is not None and feat2 is not None:
                cos_sim = cosine_similarity(feat1, feat2)
                if cos_sim < appearance_thresh:
                    continue
            associations.append((tid1, tid2))
    return associations

# =============================================================================
# 7. Union-Find for Global Fusion
# =============================================================================
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.cameras = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.cameras[x] = set([x.split('_')[0]])
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        self.parent[y_root] = x_root
        self.cameras[x_root].update(self.cameras[y_root])

    def get_groups(self):
        groups = {}
        for item in self.parent:
            root = self.find(item)
            if root not in groups:
                groups[root] = []
            groups[root].append(item)
        return groups

    def get_group_cameras(self, group_id):
        return self.cameras[self.find(group_id)]

# =============================================================================
# 8. Save MOT Format with Features
# =============================================================================
def save_mot_and_features(tracked_objects, features, output_mot_path):
    """
    Saves tracking results in MOT format, appending the feature vector at the end.
    tracked_objects: dict, frame_idx -> { track_id: ([x1, y1, x2, y2], frame_idx) }
    features: dict, track_id -> feature vector (numpy array)
    """
    with open(output_mot_path, "w") as f:
        for frame_idx, objects in tracked_objects.items():
            for obj_id, (bbox, _) in objects.items():
                x1, y1, x2, y2 = map(int, bbox)
                width, height = x2 - x1, y2 - y1
                feat_str = ""
                if obj_id in features:
                    feat = features[obj_id]
                    feat_str = " ".join(f"{v:.4f}" for v in feat)
                line = f"{frame_idx}, {obj_id}, {x1}, {y1}, {width}, {height}, 1, 1, 1, {feat_str}"
                f.write(line + "\n")

# =============================================================================
# 9. Helper for Unique Colors for Visualization
# =============================================================================
track_colors = {}
def get_color(track_id):
    if track_id not in track_colors:
        color = np.random.randint(0, 256, size=3).tolist()  # BGR
        track_colors[track_id] = color
    return track_colors[track_id]

# =============================================================================
# 10. Single-Camera Tracking Function (process one video, save MOT and deepSORT features)
# =============================================================================
def process_camera_video(video_path, roi_path, output_folder):
    """
    Process a single camera video using deepSORT to generate MOT results and obtain appearance features.
    Returns:
       tracked_dict: dict mapping frame index to detections (for MOT file)
       global_features: dict mapping track id to representative deepSORT feature
    """
    roi_mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
    if roi_mask is None:
        print("Error: Cannot load ROI mask from:", roi_path)
        sys.exit(1)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video", video_path)
        sys.exit(1)
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(output_folder, exist_ok=True)
    tracked_dict = {}
    global_features = {}  # For each confirmed track id
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
        filtered_detections = [det for det in detections if in_roi(det[0], roi_mask, min_ratio=0.5)]
        if len(filtered_detections) == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_idx += 1
            print(f"Frame {frame_idx}: No detections after ROI filtering.")
            continue
        object_chips = chipper(frame, filtered_detections)
        embeds = embedder(object_chips)
        # Update deepSORT tracker with external embeddings
        tracks = tracker.update_tracks(filtered_detections, embeds=embeds, frame=frame)
        frame_tracking = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            ltrb = track.to_ltrb()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, ltrb)
            frame_tracking[tid] = ([x1, y1, x2, y2], frame_idx)
            color = get_color(tid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Use the last deepSORT predicted feature as representative
            if hasattr(track, "features") and len(track.features) > 0:
                # You might also average the features if you prefer
                global_features[tid] = track.features[-1]
            else:
                global_features[tid] = np.zeros(512)
        tracked_dict[frame_idx] = frame_tracking
        frame_filename = os.path.join(output_folder, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1
        print(f"Processed frame {frame_idx} with {len(filtered_detections)} valid detections.")
    cap.release()
    return tracked_dict, global_features

# =============================================================================
# 11. Main Global Multi-Camera Fusion Process
# =============================================================================

if __name__ == "__main__":
    # Define camera IDs and file paths for MOT results (already generated) and videos
    cam_ids = ["c001", "c002", "c003", "c004", "c005"]
    tracking_files = {
        "c001": "/home/toukapy/Dokumentuak/Master CV/C6/Week4/s01_c001_roi.txt",
        "c002": "/home/toukapy/Dokumentuak/Master CV/C6/Week4/s01_c002_roi.txt",
        "c003": "/home/toukapy/Dokumentuak/Master CV/C6/Week4/s01_c003_roi.txt",
        "c004": "/home/toukapy/Dokumentuak/Master CV/C6/Week4/s01_c004_roi.txt",
        "c005": "/home/toukapy/Dokumentuak/Master CV/C6/Week4/s01_c005_roi.txt",
    }
    video_files = {
        "c001": "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/train/S01/c001/vdo.avi",
        "c002": "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/train/S01/c002/vdo.avi",
        "c003": "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/train/S01/c003/vdo.avi",
        "c004": "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/train/S01/c004/vdo.avi",
        "c005": "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/train/S01/c005/vdo.avi",
    }
    roi_files = {
        "c001": "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/train/S01/c001/roi.jpg",
        "c002": "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/train/S01/c002/roi.jpg",
        "c003": "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/train/S01/c003/roi.jpg",
        "c004": "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/train/S01/c004/roi.jpg",
        "c005": "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/train/S01/c005/roi.jpg",
    }
    calib_files = {
        "c001": "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/train/S01/c001/calibration.txt",
        "c002": "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/train/S01/c002/calibration.txt",
        "c003": "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/train/S01/c003/calibration.txt",
        "c004": "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/train/S01/c004/calibration.txt",
        "c005": "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/train/S01/c005/calibration.txt",
    }
    timestamp_file = "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/cam_timestamp/S01.txt"
    fps = 10

    # Reference GPS (center of S01 zone)
    ref_gps = [42.525678, -90.723601]

    # Load timestamps and calibrations
    timestamps = load_timestamps(timestamp_file)
    H_cam = {cam: load_calibration(calib_files[cam]) for cam in cam_ids}

    # For each camera, run single-camera tracking to get appearance features from deepSORT.
    global_features_by_cam = {}
    for cam in cam_ids:
        print(f"Processing video for camera {cam}...")
        video_path = video_files[cam]
        roi_path = roi_files[cam]
        out_folder = f"Week4/{cam}_roi_global"
        tracked_dict, global_feats = process_camera_video(video_path, roi_path, out_folder)
        # Save MOT+features for this camera
        mot_out = os.path.join("Global_Tracking_Miren_appearance", f"{cam}_global.txt")
        os.makedirs("Global_Tracking_Miren_appearance", exist_ok=True)
        save_mot_and_features(tracked_dict, global_feats, mot_out)
        global_features_by_cam[cam] = global_feats

    # Load tracking results from MOT files for each camera and aggregate tracklets.
    tracklets = {}
    appearance_feats = {}  # Here we now use the deepSORT features obtained above.
    for cam in cam_ids:
        tracking = load_tracking_results(tracking_files[cam])
        start_time = timestamps.get(cam, 0)
        tracklets[cam] = aggregate_tracklets(tracking, fps, start_time)
        # For each tracklet (local id), look up its corresponding deepSORT feature.
        # You might average over multiple frames if available; here we assume one representative feature per local track id.
        appearance_feats[cam] = global_features_by_cam[cam]

    # Perform pairwise association using appearance and spatio-temporal constraints.
    all_associations = []
    for cam1, cam2 in combinations(cam_ids, 2):
        assoc = associate_tracklets_with_appearance(tracklets[cam1], tracklets[cam2],
                                                    H_cam[cam1], H_cam[cam2], ref_gps,
                                                    appearance_feats[cam1], appearance_feats[cam2],
                                                    min_time_gap=1, time_tol=60, spatial_tol=50, appearance_thresh=0.5)
        for tid1, tid2 in assoc:
            all_associations.append((f"{cam1}_{tid1}", f"{cam2}_{tid2}"))
            print(f"Asociado {cam1}_{tid1} con {cam2}_{tid2}")

    # Fuse associations using Union-Find to create global groups.
    uf = UnionFind()
    for a, b in all_associations:
        uf.union(a, b)
    groups = uf.get_groups()

    # Filter groups that appear in at least 2 cameras and assign a global ID.
    global_id_mapping = {}
    global_id = 1
    for group_id, items in groups.items():
        group_cams = uf.get_group_cameras(group_id)
        if len(group_cams) < 2:
            continue
        for item in items:
            global_id_mapping[item] = global_id
        global_id += 1

    # Re-write tracking files for each camera with the global ID instead of the local track id.
    out_dir = "Global_Tracking_Miren"
    os.makedirs(out_dir, exist_ok=True)
    for cam in cam_ids:
        input_file = tracking_files[cam]
        output_file = os.path.join(out_dir, f"{cam}_global.txt")
        with open(input_file, "r") as fin, open(output_file, "w") as fout:
            for line in fin:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                frame = parts[0].strip()
                local_tid = parts[1].strip()
                key = f"{cam}_{local_tid}"
                if key in global_id_mapping:
                    new_tid = global_id_mapping[key]
                    new_line = f"{frame}, {new_tid}, {parts[2].strip()}, {parts[3].strip()}, {parts[4].strip()}, {parts[5].strip()}"
                    if len(parts) > 6:
                        extras = ", ".join(part.strip() for part in parts[6:])
                        new_line += ", " + extras
                    fout.write(new_line + "\n")
        print(f"Archivo global para {cam} guardado en {output_file}")

    print("Grupos globales (global_id: [tracklets]):")
    for root, items in groups.items():
        group_cams = uf.get_group_cameras(root)
        if len(group_cams) < 2:
            continue
        print(f"Global ID {global_id_mapping.get(root, 'N/A')}: {items}")

