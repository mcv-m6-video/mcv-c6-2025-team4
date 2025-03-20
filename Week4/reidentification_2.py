import os
import sys
import json
import cv2
import numpy as np
from itertools import combinations, product
from math import radians, sin, cos, sqrt, atan2
from multi_matching_global_og import load_tracking_results, aggregate_tracklets, UnionFind, project_to_local, \
    load_timestamps, load_calibration
from deep_sort_multi_track_ROI import extract_feature

def load_detection_metadata(metadata_file):
    with open(metadata_file, "r") as f:
        detection_metadata = json.load(f)
    # Expect keys to be strings (frame numbers) mapping to lists of detection dictionaries.
    return detection_metadata

sys.path.append('core')
from src.sort import Sort  # Ya tienes implementado el algoritmo SORT
from torchvision import models, transforms
import torchvision.ops as ops
from tqdm import tqdm


# =============================================================================
# Helper Functions: Load Start Times, Ground Truth, Homography, Predictions
# =============================================================================
def load_start_times(txt_file, fps=10):
    """Load video start times and convert them to frames."""
    start_frames = {}
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                video_id, time_seconds = parts[0], float(parts[1])
                if video_id == 'c015':
                    fps = 8
                start_frames[video_id] = int(time_seconds * fps)
    print("Start frames:", start_frames)
    return start_frames


def load_ground_truth(file_path, sequence=None):
    if sequence is None:
        gt_data = {}
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                frame, track_id, x, y, w, h = map(int, parts[:6])
                if frame not in gt_data:
                    gt_data[frame] = []
                gt_data[frame].append([track_id, x, y, w, h])
        return gt_data
    else:
        gt_dict = {}
        for vid in sequence:
            gt_data = {}
            with open(file_path + vid + "/gt/gt.txt", "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    frame, track_id, x, y, w, h = map(int, parts[:6])
                    if frame not in gt_data:
                        gt_data[frame] = []
                    gt_data[frame].append([track_id, x, y, w, h])
            gt_dict[vid] = gt_data
        return gt_dict


def load_homography(base_dir, vid_sequence):
    """Load homography matrices from calibration.txt for each camera."""
    homographies = {}
    distortions = {}
    for vid in vid_sequence:
        with open(f"{base_dir}{vid}/calibration.txt", 'r') as f:
            lines = f.readlines()
        values = np.array([float(x) for x in " ".join(lines).replace(";", " ").split()])
        H = values[:9].reshape(3, 3)
        homographies[vid] = np.linalg.inv(H)  # Inverse to project from image to world
        distortions[vid] = values[9:]
    return homographies, distortions


def load_predictions(file_path, sequence):
    """
    Loads predictions (detections) from text files.
    Expected format (per line): frame, track_id, x, y, w, h
    """
    pred_dict = {}
    for vid in sequence:
        data = {}
        with open(file_path + "s03_" + vid + "_roi.txt", "r") as f:
            for line in f:
                parts = line.strip().split(",")
                frame, track_id, x, y, w, h = map(int, parts[:6])
                if frame not in data:
                    data[frame] = []
                data[frame].append([track_id, x, y, w, h])
        pred_dict[vid] = data
    return pred_dict


def get_max_frame(gt_dict):
    last = []
    for vid in gt_dict:
        last.append(list(gt_dict[vid].keys())[-1])
    return max(last)


def project_to_world(homography, x, y):
    """Project image pixel coordinates (x, y) to world coordinates."""
    point = np.array([x, y, 1])
    projected = np.dot(homography, point)
    return projected[:2] / projected[2]


def get_world_coordinates(detections, homographies, start_frames, distortions):
    """Convert bounding box coordinates to world coordinates using homographies."""
    world_positions = {}
    # Intrinsic camera matrix (adjust if needed)
    K = np.array([[1000, 0, 640],
                  [0, 1000, 480],
                  [0, 0, 1]])
    for vid, dets in detections.items():
        print("Processing world coordinates for", vid)
        world_positions[vid] = {}
        H_inv = homographies[vid]
        for frame, info in dets.items():
            if frame not in world_positions[vid]:
                world_positions[vid][frame] = []
            for item in info:
                track_id, x, y, w, h = item
                x_center = x + w / 2
                y_center = y + h / 2
                if np.shape(distortions[vid]) != ():
                    undistorted = cv2.undistortPoints(np.array([[x_center, y_center]], dtype=np.float32), K,
                                                      distortions[vid], P=K)
                    undistorted = undistorted.reshape(-1, 2)
                    x_center, y_center = undistorted[0, 0], undistorted[0, 1]
                world_x, world_y = project_to_world(H_inv, x_center, y_center)
                world_positions[vid][frame].append([track_id, world_x, world_y, x, y, w, h])
    return world_positions


def find_corresponding_frame(vid1, vid2, frame1, start_frames, fps=None):
    """Compute corresponding frame in vid2 given a frame in vid1 using start times and fps."""
    if fps is not None:
        f1 = start_frames[vid1]
        f2 = start_frames[vid2]
        time_elapsed = (frame1 - f1) / fps[vid1]
        frame2 = f2 + time_elapsed * fps[vid2]
    else:
        f1 = start_frames[vid1]
        f2 = start_frames[vid2]
        frame2 = frame1 - (f1 - f2)
    return round(frame2)


def find_order(start_frames):
    """Returns cameras ordered descending by start time."""
    return dict(sorted(start_frames.items(), key=lambda item: item[1], reverse=True))


def haversine_distance_meters(coord1, coord2):
    """Compute Haversine distance in meters between two (lat, lon) pairs."""
    R = 6371000
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def find_potential_matches(coords1, coords2, track1s, track2s, bboxes1, bboxes2, threshold=2.0):
    """Find matching objects based on Haversine distance."""
    matches = []
    for (track1, coord1, box1), (track2, coord2, box2) in product(zip(track1s, coords1, bboxes1),
                                                                  zip(track2s, coords2, bboxes2)):
        dist = haversine_distance_meters(np.array(coord1), np.array(coord2))
        if dist <= threshold:
            matches.append((track1, track2, box1, box2))
    return matches


def match_across_cameras(world_positions, start_frames, threshold=2.0, fps=None):
    """
    Matches detections across cameras using aligned frames and world coordinates.
    Returns a list of matches: each match is a pair of keys [ [vid, track_id, frame, bbox], [vid, track_id, frame, bbox] ].
    """
    final_tracks = []
    vid_order = find_order(start_frames)
    for vid1 in vid_order:
        for frame1, info1 in world_positions[vid1].items():
            for vid2 in world_positions:
                if vid1 == vid2:
                    continue
                frame2 = find_corresponding_frame(vid1, vid2, frame1, start_frames, fps)
                if frame2 < 0 or frame2 not in world_positions[vid2]:
                    continue
                info2 = world_positions[vid2][frame2]
                # Extract track IDs, world coordinates and bounding boxes
                track1s, coords1, bboxes1 = zip(
                    *[(obj[0], [obj[1], obj[2]], [obj[3], obj[4], obj[5], obj[6]]) for obj in info1])
                track2s, coords2, bboxes2 = zip(
                    *[(obj[0], [obj[1], obj[2]], [obj[3], obj[4], obj[5], obj[6]]) for obj in info2])
                matches = find_potential_matches(coords1, coords2, track1s, track2s, bboxes1, bboxes2,
                                                 threshold=threshold)
                for track1, track2, bbox1, bbox2 in matches:
                    key1 = [vid1, track1, frame1, bbox1]
                    key2 = [vid2, track2, frame2, bbox2]
                    final_tracks.append([key1, key2])
    return final_tracks


# =============================================================================
# Appearance (DeepSORT) Functions from Our Code
# =============================================================================
def compute_average_embeddings_from_metadata(tracklets, detection_metadata, crops_folder, distance_threshold=50):
    """
    For each tracklet (keys: "frames", "avg_center"), select in each frame the detection
    whose bounding-box center is closest to the tracklet’s average center (if within threshold).
    Load the crop image (using "crop_filename") and compute its embedding (via extract_feature).
    Average embeddings are stored in "avg_embedding".
    """
    for tid, data in tracklets.items():
        embeddings = []
        selected_detections = []
        tracklet_frames = data.get("frames", [])
        tracklet_center = np.array(data["avg_center"])
        for frame in tracklet_frames:
            key = str(frame)
            if key not in detection_metadata:
                continue
            detections_in_frame = detection_metadata[key]
            best_det = None
            best_dist = float('inf')
            for det in detections_in_frame:
                bbox = det.get("bbox")
                if bbox is None:
                    continue
                x, y, w, h = bbox
                center_det = np.array([x + w / 2, y + h / 2])
                dist = np.linalg.norm(center_det - tracklet_center)
                if dist < best_dist:
                    best_dist = dist
                    best_det = det
            if best_det is not None and best_dist < distance_threshold:
                selected_detections.append(best_det)
                crop_filename = best_det.get("crop_filename")
                if crop_filename:
                    crop_path = os.path.join(crops_folder, crop_filename)
                    if os.path.exists(crop_path):
                        crop_img = cv2.imread(crop_path)
                        if crop_img is not None:
                            emb = extract_feature(crop_img)
                            embeddings.append(emb)
                        else:
                            print(f"Warning: Could not read crop image {crop_path}")
                    else:
                        print(f"Warning: Crop file {crop_path} not found.")
                else:
                    print("Warning: 'crop_filename' not found in detection:", best_det)
        data["detections"] = selected_detections
        if embeddings:
            data["avg_embedding"] = np.mean(embeddings, axis=0)
        else:
            data["avg_embedding"] = None
    return tracklets


def cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0.0
    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-6)
    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-6)
    return np.dot(emb1_norm, emb2_norm)


def associate_tracklets_with_embeddings(tracklets_cam1, tracklets_cam2, H1, H2, ref_gps,
                                        min_time_gap=1, time_tol=60, spatial_tol=50, emb_threshold=0.5):
    """
    For each pair of tracklets (one from each camera), checks:
      - Temporal gap: data2.start_time - data1.end_time within [min_time_gap, time_tol]
      - Spatial distance (projected centers) < spatial_tol
      - Cosine similarity between "avg_embedding" >= emb_threshold
    If all conditions are met, the tracklets are associated.
    """
    associations = []
    for tid1, data1 in tracklets_cam1.items():
        center1_local = project_to_local(data1["avg_center"], H1, ref_gps)
        for tid2, data2 in tracklets_cam2.items():
            dt = data2["start_time"] - data1["end_time"]
            if dt < min_time_gap or dt > time_tol:
                continue
            center2_local = project_to_local(data2["avg_center"], H2, ref_gps)
            spatial_dist = np.linalg.norm(center1_local - center2_local)
            if spatial_dist < spatial_tol:
                sim = cosine_similarity(data1.get("avg_embedding"), data2.get("avg_embedding"))
                if sim >= emb_threshold:
                    associations.append((tid1, tid2))
    return associations


# =============================================================================
# Re-Identification Update: Update Track IDs (without UnionFind)
# =============================================================================
def filter_and_update_track_id(data, old_id, new_id):
    """
    Creates a new dictionary with only detections having old_id and updates them to new_id.
    Deep copies the matched detections.
    """
    updated_data = {}
    for frame, objects in data.items():
        updated_objects = [obj[:] for obj in objects if obj[0] == old_id]
        for obj in updated_objects:
            obj[0] = new_id
        if updated_objects:
            updated_data[frame] = updated_objects
    return updated_data


# =============================================================================
# Saving Functions
# =============================================================================
def save_tracks_to_files(final_tracks, output_folder):
    """
    Saves final per-camera tracks to files.
    Format: frame, track_id, x, y, w, h
    """
    os.makedirs(output_folder, exist_ok=True)
    for vid, frames in final_tracks.items():
        output_file = os.path.join(output_folder, f"{vid}_final_tracks.txt")
        with open(output_file, "w") as f:
            for frame, objects in sorted(frames.items()):
                for obj in objects:
                    track_id, x, y, w, h = obj
                    f.write(f"{frame},{track_id},{x},{y},{w},{h}\n")
        print(f"Saved: {output_file}")


# =============================================================================
# Main Multi-Camera Pipeline with Appearance-based Re-ID
# =============================================================================
if __name__ == "__main__":
    # Settings and paths (adjust as needed)
    seq = 'S03/'
    videos = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    output_dir = "./final_tracks/"
    video_dir = "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/data/aic19-track1-mtmc-train/train/S03"
    start_frames = load_start_times(
        "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/data/aic19-track1-mtmc-train/cam_timestamp/" +
        seq.split('/')[0] + '.txt')

    homographies, distortions = load_homography(
        "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/data/aic19-track1-mtmc-train/train/" + seq, videos)
    gt_dict = load_ground_truth(
        "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/data/aic19-track1-mtmc-train/train/" + seq, videos)
    detections = load_predictions("/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/Week4/results/", videos)
    max_frame = get_max_frame(gt_dict)

    # Convert bounding boxes to world coordinates
    world_positions = get_world_coordinates(detections, homographies, start_frames, distortions)

    fps = {
        'c010': 10,
        'c011': 10,
        'c012': 10,
        'c013': 10,
        'c014': 10,
        'c015':8

    }
    ref_gps = [42.525678, -90.723601]

    # --- Appearance: Load detection metadata and compute appearance embeddings ---
    detection_metadata_by_cam = {}
    base_tracking_path = "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/Week4/results/"
    crops_folder = "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/Week4/detection_crops"
    for cam in videos:
        metadata_file = os.path.join(base_tracking_path, f"s03_{cam}_detection_metadata.json")
        detection_metadata_by_cam[cam] = load_detection_metadata(metadata_file)

    # --- Compute tracklets per camera using tracking results ---
    tracklets = {}
    for cam in videos:
        tracking = load_tracking_results(os.path.join(base_tracking_path, f"s03_{cam}_roi.txt"))
        # (Assume default fps for all; adjust if needed)
        tracklets[cam] = aggregate_tracklets(tracking, fps[cam], start_frames[cam])

    # --- Compute average appearance embeddings for each tracklet ---
    tracklets_by_cam = {}
    for cam in videos:
        tracklets_by_cam[cam] = compute_average_embeddings_from_metadata(
            tracklets[cam],
            detection_metadata_by_cam[cam],
            crops_folder,
            distance_threshold=1000
        )

    # --- Associate tracklets between cameras using spatial, temporal, and appearance cues ---
    all_associations = []
    for cam1, cam2 in combinations(videos, 2):
        assoc = associate_tracklets_with_embeddings(
            tracklets_by_cam[cam1],
            tracklets_by_cam[cam2],
            homographies[cam1],
            homographies[cam2],
            ref_gps,
            min_time_gap=1,
            time_tol=60,
            spatial_tol=50,
            emb_threshold=0.5
        )
        for tid1, tid2 in assoc:
            all_associations.append((f"{cam1}_{tid1}", f"{cam2}_{tid2}"))
    print("Pairwise associations found:")
    print(all_associations)

    # --- Re-Identification Update: Update track IDs using the associations ---
    changed_ids = {}  # To track which original IDs have been processed per video
    new_id = 1  # Global new track ID counter
    detections_reid = {}  # Final dictionary: video -> {frame: [updated detections]}

    for element in all_associations:
        a, b = element[0], element[1]  # Each is a string like "c001_3"
        vid_a, id_a = a.split('_')[0], int(a.split('_')[1])
        vid_b, id_b = b.split('_')[0], int(b.split('_')[1])

        if vid_a not in detections_reid:
            detections_reid[vid_a] = {}
        if vid_b not in detections_reid:
            detections_reid[vid_b] = {}
        if vid_a not in changed_ids:
            changed_ids[vid_a] = []
        if vid_b not in changed_ids:
            changed_ids[vid_b] = []

        if id_a not in changed_ids[vid_a] and id_b not in changed_ids[vid_b]:
            # (Optional) Debug print for a particular camera
            if vid_a == 'c015' or vid_b == 'c015':
                print("Match:", a, b)
            updated_a = filter_and_update_track_id(detections[vid_a], id_a, new_id)
            updated_b = filter_and_update_track_id(detections[vid_b], id_b, new_id)
            for frame, objs in updated_a.items():
                detections_reid[vid_a].setdefault(frame, []).extend(objs)
            for frame, objs in updated_b.items():
                detections_reid[vid_b].setdefault(frame, []).extend(objs)
            changed_ids[vid_a].append(id_a)
            changed_ids[vid_b].append(id_b)
            new_id += 1

    print("Updated detections for c015:")
    print(detections_reid.get('c015', {}))

    # --- Save final updated tracks to files ---
    save_tracks_to_files(detections_reid, output_dir)
