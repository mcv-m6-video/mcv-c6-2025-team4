import os
import json
import numpy as np
import cv2
from itertools import combinations

# Import spatial/temporal functions from your global matching script
from multi_matching_global_og import load_tracking_results, aggregate_tracklets, UnionFind, project_to_local, \
    load_timestamps, load_calibration

# Import the feature extraction function from your DeepSORT embedder script
from deep_sort_multi_track_ROI import extract_feature


# --------------------------------------------------
# Function to load detection metadata from a JSON file
# --------------------------------------------------
def load_detection_metadata(metadata_file):
    with open(metadata_file, "r") as f:
        detection_metadata = json.load(f)
    # Expect keys to be strings (frame numbers) mapping to lists of detection dictionaries.
    return detection_metadata


# --------------------------------------------------
# Compute average embeddings for tracklets using detection metadata
# --------------------------------------------------
def compute_average_embeddings_from_metadata(tracklets, detection_metadata, crops_folder, distance_threshold=50):
    """
    For each tracklet (which has keys "frames" and "avg_center"), for each frame in the tracklet,
    select the detection (from the provided detection_metadata, which is a dict keyed by frame number)
    whose bounding-box center is closest to the tracklet's avg_center (if within distance_threshold).
    Then load the corresponding crop image (using the "crop_filename" field) from crops_folder and compute
    its embedding via extract_feature. The tracklet's "avg_embedding" is set as the average of these embeddings.

    The selected detections are also stored in the tracklet under a new key "detections".
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
                # Expecting each detection dict to have "bbox": [x, y, w, h]
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


# --------------------------------------------------
# Cosine similarity between two embeddings
# --------------------------------------------------
def cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0.0
    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-6)
    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-6)
    return np.dot(emb1_norm, emb2_norm)


# --------------------------------------------------
# Association function using spatial, temporal, and appearance cues
# --------------------------------------------------
def associate_tracklets_with_embeddings(tracklets_cam1, tracklets_cam2, H1, H2, ref_gps,
                                        min_time_gap=1, time_tol=60, spatial_tol=50, emb_threshold=0.8):
    """
    For each pair of tracklets (one from each camera), check:
      - Time gap between tracklets (dt = second.start_time - first.end_time)
      - Spatial distance between projected average centers
      - Cosine similarity between their "avg_embedding" fields.
    If conditions are met, the two tracklets are considered associated.
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


# --------------------------------------------------
# Function to detect missing detections in global groups (unchanged)
# --------------------------------------------------
def detect_missing_detections(groups, cam_ids):
    """
    For each global group (keys like "c015_23"), determine which cameras are missing.
    """
    missing = {}
    for group_id, items in groups.items():
        present_cams = set(item.split('_')[0] for item in items)
        missing_cams = set(cam_ids) - present_cams
        if missing_cams:
            missing[group_id] = list(missing_cams)
    return missing

def create_filtered_mot_files(tracking_files, groups, output_dir):
    """
    Reads original MOT files (one per camera) and creates new MOT files that only keep detections
    that appear in at least two cameras. Detections not in multi-camera groups are eliminated.

    Parameters:
      tracking_files : dict
          Dictionary mapping camera IDs to the file paths of their original MOT files.
      groups : dict
          Dictionary of union–find groups. Each value is a list of strings formatted as "cam_localid".
      output_dir : str
          Directory where the new MOT files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build a global ID mapping only for groups that span at least two different cameras.
    global_id_mapping = {}
    global_id = 1
    for group_rep, items in groups.items():
        # Determine the set of cameras present in this group.
        cameras_in_group = set(item.split("_")[0] for item in items)
        if len(cameras_in_group) >= 2:
            for item in items:
                global_id_mapping[item] = global_id
            global_id += 1
        # Otherwise, ignore this group (detections not appearing in other cameras will be dropped).

    # Process each camera file.
    for cam, file_path in tracking_files.items():
        new_lines = []
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                frame = parts[0].strip()
                local_id = parts[1].strip()
                key = f"{cam}_{local_id}"
                if key in global_id_mapping:
                    new_id = global_id_mapping[key]
                    new_line = f"{frame}, {new_id}, {', '.join(parts[2:])}\n"
                    new_lines.append(new_line)
                # If the detection is not in the mapping, we simply eliminate it.

        # Sort the new lines by frame number.
        new_lines.sort(key=lambda l: int(l.split(',')[0].strip()))

        output_file = os.path.join(output_dir, f"{cam}_global.txt")
        with open(output_file, "w") as fout:
            fout.writelines(new_lines)
        print(f"Created filtered MOT file for camera {cam}: {output_file}")


# --------------------------------------------------
# Main multi-camera matching pipeline using per-camera detection metadata
# --------------------------------------------------
if __name__ == "__main__":
    # Define camera IDs and paths (update these paths as needed)
    cam_ids = ['c011','c012','c013','c010','c014','c015']
    base_tracking_path = "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/Week4/results"
    tracking_files = {cam: os.path.join(base_tracking_path, f"s03_{cam}_roi.txt") for cam in cam_ids}
    calib_files = {
        cam: f"/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/data/aic19-track1-mtmc-train/train/S03/{cam}/calibration.txt"
        for cam in cam_ids}
    timestamp_file = "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/data/aic19-track1-mtmc-train/cam_timestamp/S03.txt"

    # For each camera, we assume a separate detection metadata JSON file exists.
    # For example, files like "s03_c010_detection_metadata.json", etc.
    detection_metadata_by_cam = {}
    for cam in cam_ids:
        metadata_file = os.path.join(base_tracking_path, f"s03_{cam}_detection_metadata.json")
        detection_metadata_by_cam[cam] = load_detection_metadata(metadata_file)

    # Load timestamps
    timestamps = {}
    with open(timestamp_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                cam = parts[0]
                timestamps[cam] = float(parts[1])
    H_cam = {cam: load_calibration(calib_files[cam]) for cam in cam_ids}

    fps = 10
    ref_gps = [42.525678, -90.723601]

    # Path to the folder with the saved detection crops
    crops_folder = "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/Week4/detection_crops"

    # Aggregate tracklets for each camera.
    # The aggregate_tracklets function produces tracklets with keys:
    # 'frames', 'centers', 'start_time', 'end_time', and 'avg_center'.
    # (These tracklets do not include per-detection info.)
    tracklets = {}
    for cam in cam_ids:
        tracking = load_tracking_results(tracking_files[cam])
        if cam == "c015":
            tracklets[cam] = aggregate_tracklets(tracking, 8, timestamps[cam])
        else:
            tracklets[cam] = aggregate_tracklets(tracking, fps, timestamps[cam])

    # Now, for each camera, attach detection data and compute average embeddings.
    tracklets_by_cam = {}
    for cam in cam_ids:
        tracklets_by_cam[cam] = compute_average_embeddings_from_metadata(
            tracklets[cam],
            detection_metadata_by_cam[cam],
            crops_folder,
            distance_threshold=50
        )

    # Perform pairwise associations between cameras using spatial, temporal, and appearance cues.
    all_associations = []
    for cam1, cam2 in combinations(cam_ids, 2):
        assoc = associate_tracklets_with_embeddings(
            tracklets_by_cam[cam1],
            tracklets_by_cam[cam2],
            H_cam[cam1],
            H_cam[cam2],
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

    # Use Union-Find (from your global matching script) to merge associations across cameras.
    uf = UnionFind()
    for a, b in all_associations:
        uf.union(a, b)
    groups = uf.get_groups()

    # Identify which cameras are missing detections in each global group.
    missing_detections = detect_missing_detections(groups, cam_ids)

    print("Missing detections per group:")
    print(missing_detections)

    # Output the global associations.
    print("\nGlobal Associations (groups with matching tracklets across cameras):")
    for group_id, items in groups.items():
        if len(items) > 1:
            print(f"Group {group_id}: {items}")

    print("\nMissing Detections per Global Group (cameras that did not detect the car):")
    for group_id, missing_cams in missing_detections.items():
        print(f"Group {group_id} missing in cameras: {missing_cams}")

    # Set output directory for new MOT files:
    output_dir = "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/Week4/Global_Tracking_New_s03"

    # Call the function to create new MOT files.
    create_filtered_mot_files(tracking_files, groups, output_dir)






