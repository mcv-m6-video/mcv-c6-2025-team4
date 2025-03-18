import os

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


# Device selection for optical flow

def load_start_times(txt_file, fps=10):
    """Load video start times and convert them to frames."""
    start_frames = {}
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                video_id, time_seconds = parts[0], float(parts[1])
                start_frames[video_id] = int(time_seconds * fps)  # Convert time to frames
    return start_frames

def align_frame(vid, frame, start_frames):
    """Align frame number across cameras using start times."""
    return frame - start_frames.get(vid, 0)

def load_ground_truth(file_path,sequence=None):
    if sequence is None:
        gt_data = {}
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                frame, track_id, x, y, w, h = map(int, parts[:6])
                if frame not in gt_data:
                    gt_data[frame] = []
                gt_data[frame].append([track_id, x, y, w, h])
    else:
        gt_dict={}
        for vid in sequence:
            gt_data = {}
            with open(file_path+vid+"/gt/gt.txt", "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    frame, track_id, x, y, w, h = map(int, parts[:6])
                    if frame not in gt_data:
                        gt_data[frame] = []
                    gt_data[frame].append([track_id, x, y, w, h])
            gt_dict[vid]=gt_data
            
    return gt_dict

def load_homography(base_dir, vid_sequence):
    """ Load homography matrices from calibration.txt """
    homographies = {}
    distortions={}
    for vid in vid_sequence:
        with open(f"{base_dir}{vid}/calibration.txt", 'r') as f:
            lines = f.readlines()
        values = np.array([float(x) for x in " ".join(lines).replace(";", " ").split()])
        H = values[:9].reshape(3, 3)  # 3x3 homography matrix
        homographies[vid] = np.linalg.inv(H)  # Inverse to project from image to real-world
        distortions[vid]=values[9:]
    return homographies,distortions

def load_predictions(file_path,sequence):
    gt_dict={}
    for vid in sequence:
        gt_data = {}
        with open(file_path+vid+".txt", "r") as f:
            for line in f:
                parts = line.strip().split(",")
                frame, track_id, x, y, w, h = map(int, parts[:6])
                if frame not in gt_data:
                    gt_data[frame] = []
                gt_data[frame].append([track_id, x, y, w, h])
        gt_dict[vid]=gt_data
    return gt_dict


def get_max_frame(gt_dict):
    last=[]
    for vid in gt_dict:
        last.append(list(gt_dict[vid].keys())[-1])
    return max(last)

def project_to_world(homography, x, y):
    """ Project image pixel coordinates (x, y) to real-world GPS coordinates """
    point = np.array([x, y, 1])
    projected = np.dot(homography, point)
    return projected[:2] / projected[2]  # Normalize

# def get_world_coordinates(detections, homographies):
#     """ Convert bounding box coordinates to world coordinates using homographies """
#     world_positions = {}
#     for vid, dets in detections.items():
#         world_positions[vid] = []
#         H_inv = homographies[vid]
#         for frame, track_id, x, y, w, h in dets:
#             x_center = x + w / 2  # Bottom center of bounding box
#             y_bottom = y + h
#             world_x, world_y = project_to_world(H_inv, x_center, y_bottom)
#             world_positions[vid].append((frame, track_id, world_x, world_y))
#     return world_positions


def get_world_coordinates(detections, homographies, start_frames,distortions):
    """Convert bounding box coordinates to world coordinates using homographies."""
    world_positions = {}
    K = np.array([[1000, 0, 640],   # fx,  0, cx
              [0, 1000, 480],   #  0, fy, cy
              [0, 0, 1]])
    
    for vid, dets in detections.items():
        print(vid)
        world_positions[vid] = []
        H_inv = homographies[vid]
        
        detect_dict={}
        for frame,info in dets.items():
            if frame not in detect_dict:
                detect_dict[frame] = []
            for item in info:
                track_id=item[0]
                x=item[1]
                y=item[2]
                w=item[3]
                h=item[4]
                x_center = x + w / 2  # Bottom center of bounding box
                y_center = y + h / 2
                # print(x_center,y_center)
                if np.shape(distortions[vid])!=0:
                    # print(vid,np.shape(distortions[vid]))
                    undistorted_points = cv2.undistortPoints(np.array([x_center,y_center]), K, distortions[vid], P=K)
                    undistorted_points = undistorted_points.reshape(-1, 2)
                    # print(undistorted_points)
                    x_center=undistorted_points[:,0][0]
                    y_center=undistorted_points[:,1][0]
                world_x, world_y = project_to_world(H_inv, x_center, y_center)

                detect_dict[frame].append([track_id, world_x, world_y, x, y, w, h])
            
            world_positions[vid]=detect_dict 
    return world_positions

def find_corresponding_frame(vid1,vid2,frame1,start_frames):
    f1=start_frames[vid1]
    f2=start_frames[vid2]   
    dif=f1-f2
    return frame1-dif

def find_order(start_frames):
    new_order=dict(sorted(start_frames.items(), key=lambda item: item[1],reverse=True))
    return new_order
from math import radians, sin, cos, sqrt, atan2
from itertools import product

def decimal_to_dms(decimal_degree):
    """Convert decimal degrees to degrees, minutes, and seconds (DMS)."""
    degrees = int(decimal_degree)
    minutes_float = abs(decimal_degree - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60
    return degrees, minutes, seconds

def euclidean_distance(coord1, coord2):
    """Compute Euclidean distance between two GPS coordinates (in degrees)."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    return sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

def haversine_distance_meters(coord1, coord2):
    """Compute Haversine distance between two GPS coordinates in meters."""
    R = 6371000  # Earth's radius in meters
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c  # Distance in meters

def find_potential_matches(coords1, coords2, track1s, track2s,bboxes1,bboxes2, threshold=2.0):
    """Find matching objects based on Euclidean distance in world coordinates."""
    matches = []
    for (track1, coord1,box1), (track2, coord2,box2) in product(zip(track1s, coords1,bboxes1), zip(track2s, coords2,bboxes2)):
        dist = haversine_distance_meters(np.array(coord1),np.array(coord2))
        
        if dist <= threshold:
            matches.append((track1, track2,box1,box2))  # Store track IDs and distance
    return matches


def compare_lists(list1, list2, metric='haversine'):
    """Compute similarity between two lists of GPS coordinates."""
    distance_function = haversine_distance_meters if metric == 'haversine' else euclidean_distance
    
    # Compute all pairwise distances
    distances = [(coord1, coord2, distance_function(coord1, coord2)) 
                 for coord1, coord2 in product(list1, list2)]
    
    # Sort pairs by similarity (lower distance is better)
    distances.sort(key=lambda x: x[2])

    return distances

def match_across_cameras(world_positions, start_frames, threshold=2.0):
    """
    Match objects across cameras using aligned frames and world coordinates.
    Uses the start times to align frames across cameras.
    """
    global_tracks = {}
    track_mapping = {}  # Maps local track_id -> global track_id
    global_track_id = 1
    track_occurrences = {}
    final_tracks_per_camera = []

    # print(world_positions)
    # Iterate through all pairs of videos

    vid_order=find_order(start_frames)

    for vid1 in vid_order:

        for frame1, info1 in world_positions[vid1].items():
            # print(vid1)
            # print(frame1,info1)
            # # Align frame numbers across cameras
            # aligned_frame1 = align_frame_across_cameras(vid1, frame1, start_frames)
            matched = False
            matched_tracks = set()

            for vid2 in world_positions:
                if vid1 == vid2:
                    continue  # Skip same video comparison

                frame2=find_corresponding_frame(vid1,vid2,frame1,start_frames)
                # print(vid2, frame2)
                if frame2<0 or frame2 not in world_positions[vid2].keys():
                    continue
                
                # print(vid1,frame1)
                # print(vid2,frame2)

                info2=world_positions[vid2][frame2]
                
                # Extract track IDs and coordinates
                # Extract track IDs, coordinates, and bounding boxes
                track1s, coords1, bboxes1 = zip(*[(obj[0], [obj[1], obj[2]], [obj[3],obj[4],obj[5],obj[6]]) for obj in info1])
                track2s, coords2, bboxes2 = zip(*[(obj[0], [obj[1], obj[2]], [obj[3],obj[4],obj[5],obj[6]]) for obj in info2])
                
                # Find matches within 5 meters
                matches = find_potential_matches(coords1, coords2, track1s, track2s,bboxes1,bboxes2, threshold=threshold)


                for track1, track2, bbox1,bbox2 in matches:
                    key1, key2 = [vid1, track1,frame1,bbox1], [vid2, track2,frame2,bbox2]
                    print(key1)
                    print(key2)
                    final_tracks_per_camera.append([key1,key2])
    #                 if key1 in track_mapping and key2 in track_mapping:
    #                     # Merge track IDs if they belong to different global IDs
    #                     global_id1, global_id2 = track_mapping[key1], track_mapping[key2]
    #                     if global_id1 != global_id2:
    #                         # Merge two global track IDs into one
    #                         for k, v in track_mapping.items():
    #                             if v == global_id2:
    #                                 track_mapping[k] = global_id1
    #                 elif key1 in track_mapping:
    #                     global_id = track_mapping[key1]
    #                 elif key2 in track_mapping:
    #                     global_id = track_mapping[key2]
    #                 else:
    #                     global_id = global_track_id
    #                     global_track_id += 1

    #                 # Update track mappings properly
    #                 track_mapping[key1] = global_id
    #                 track_mapping[key2] = global_id

    #                 # Track occurrences
    #                 track_occurrences.setdefault(global_id, set()).update([vid1, vid2])

    #                 matched_tracks.add(global_id)
           
    #         # Store matched objects for this camera
    #         for obj in info1:
    #             track_id = track_mapping.get((vid1, obj[0]), None)
    #             if track_id:
    #                 global_tracks.setdefault(track_id, []).append((vid1, frame1, obj[1], obj[2],obj[3],obj[4],obj[5],obj[6]))

    #                 # Store in per-camera structure
    #                 if frame1 not in final_tracks_per_camera[vid1]:
    #                     final_tracks_per_camera[vid1][frame1] = []
    #                 final_tracks_per_camera[vid1][frame1].append((track_id, obj[1], obj[2],obj[3],obj[4],obj[5],obj[6]))
    #     a
    # # Remove tracks that appear in fewer than 2 cameras
    # final_tracks_per_camera = {
    #     vid: {
    #         frame: [(tid,x_gps,y_gps, x, y,w,h) for tid, x_gps,y_gps,x, y,w,h in objects if len(track_occurrences[tid]) >= 2]
    #         for frame, objects in frames.items()
    #     }
    #     for vid, frames in final_tracks_per_camera.items()
    # }

    return final_tracks_per_camera

def get_color(track_id):
    """Generate a unique color for each track ID."""
    np.random.seed(track_id)
    return tuple(int(c) for c in np.random.randint(0, 255, 3))

def create_videos_with_tracks(final_tracks_per_camera, video_folder, output_folder, ground_truth):
    """
    Creates a video for each camera with the predicted bounding boxes and ground truth.
    
    :param final_tracks_per_camera: Dictionary containing final tracks per camera.
    :param video_folder: Path to the folder containing the original videos.
    :param output_folder: Path to the folder where the output videos will be saved.
    :param ground_truth: Dictionary containing ground truth bounding boxes per camera.
    """
    os.makedirs(output_folder, exist_ok=True)

    for cam_id, frames in final_tracks_per_camera.items():
        video_path = os.path.join(video_folder,cam_id, "vdo.avi")
        if not os.path.exists(video_path):
            print(f"Warning: Video file {video_path} not found.")
            continue

        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        output_video_path = os.path.join(output_folder, f"{cam_id}_tracked.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index in frames:
                for obj in frames[frame_index]:
                    # print(obj)
                    track_id, _,_, x, y, w, h = obj  # Extract bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if cam_id in ground_truth and frame_index in ground_truth[cam_id]:
                for obj in ground_truth[cam_id][frame_index]:
                    track_id, x, y, w, h = obj
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, f"GT: {track_id}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            out.write(frame)
            frame_index += 1

        cap.release()
        out.release()
        print(f"Saved tracked video: {output_video_path}")



def save_tracks_to_files(final_tracks_per_camera, output_folder):
    """
    Saves the final per-camera tracks to separate files in the same format as input.
    
    :param final_tracks_per_camera: Dictionary containing final tracks per camera.
    :param output_folder: Path to the folder where the output files will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    for cam_id, frames in final_tracks_per_camera.items():
        output_file = os.path.join(output_folder, f"{cam_id}_final_tracks.txt")

        with open(output_file, "w") as f:
            print(final_tracks_per_camera[cam_id])
            for frame, objects in sorted(final_tracks_per_camera[cam_id].items()):
                for obj in objects:
                    track_id, x, y ,w,h= obj
                    f.write(f"{frame},{track_id},{x},{y},{w},{h}\n")  # Same format as input
        print(f"Saved: {output_file}")  # Log output file path



# seq='S01/'
# videos=['c001','c002','c003','c004','c005']
seq='S03/'
videos=['c010','c011','c012','c013','c014','c015']
output_dir = "./final_tracks/"
video_dir = "E:/aic19-track1-mtmc-train/train/S01"
start_frames = load_start_times("E:/aic19-track1-mtmc-train/cam_timestamp/"+seq.split('/')[0]+'.txt')

homographies,distortions=load_homography("E:/aic19-track1-mtmc-train/train/"+seq,videos)

gt_dict=load_ground_truth("E:/aic19-track1-mtmc-train/train/"+seq,videos)
detections=load_predictions("C:/Users/User/Documents/GitHub/mcv-c6-2025-team4/Week4/",videos)
max_frame=get_max_frame(gt_dict)

# Convert bounding boxes to world coordinates
world_positions = get_world_coordinates(detections, homographies, start_frames,distortions)

# print(world_positions['c001'].keys())
# print(world_positions['c002'].keys())
# print(world_positions['c003'].keys())
# print(world_positions['c004'].keys())
# print(world_positions['c005'].keys())


# Match detections across cameras
matches = match_across_cameras(world_positions,start_frames,20)


# Save final tracks with updated track IDs in the original format
# save_tracks_to_files(track_mapping,output_dir)
# # generate_tracking_videos(video_dir, output_dir, detections, track_mapping, gt_dict)
# create_videos_with_tracks(track_mapping, video_dir, output_dir, gt_dict)

# Example usage:
# matches = [('c005', 341, 2, [296, 222, 141, 103]), 
#            ('c004', 238, 1, [1335, 380, 414, 210])]


def update_track_id(data, old_id, new_id):
    for frame, objects in data.items():
        for obj in objects:
            if obj[0] == old_id:
                obj[0] = new_id
    return data

# def filter_and_update_track_id(data, old_id, new_id):
#     updated_data = {}
#     for frame, objects in data.items():
#         updated_objects = []
#         for obj in objects:
#             if obj[0] == old_id:
#                 obj[0] = new_id
#                 updated_objects.append(obj)
#         if updated_objects:
#             updated_data[frame] = updated_objects  # Only add the frame if it has been updated
#     return updated_data

# changed_ids={}
# new_id=1
# detections_reid=detections.copy()

# # print(matches[0])
# for element in matches:
#     a=element[0]
#     b=element[1]
#     if a[0] not in changed_ids:
#         changed_ids[a[0]]=[]
#     if b[0] not in changed_ids:
#         changed_ids[b[0]]=[]

#     for i in a:
#         if a[1] not in changed_ids[a[0]] and b[1] not in changed_ids[b[0]] :
#             detections_reid[a[0]]=filter_and_update_track_id(detections_reid[a[0]],a[1],new_id)
#             detections_reid[b[0]]=filter_and_update_track_id(detections_reid[b[0]],b[1],new_id)
           
#             changed_ids[a[0]].append(a[1])
#             changed_ids[b[0]].append(b[1])
#             # print(changed_ids)
#             new_id=new_id+1
    

def filter_and_update_track_id(data, old_id, new_id):
    """Creates a new dictionary with only the updated track IDs."""
    updated_data = {}
    for frame, objects in data.items():
        updated_objects = [obj[:] for obj in objects if obj[0] == old_id]  # Copy matched objects
        for obj in updated_objects:
            obj[0] = new_id  # Update track_id
        if updated_objects:
            updated_data[frame] = updated_objects  # Store only updated detections
    return updated_data

changed_ids = {}
new_id = 1
detections_reid = {}  # New dictionary structured by video ID -> frames

for element in matches:
    a, b = element[0], element[1]  # (video_id, track_id)

    if a[0] not in detections_reid:
        detections_reid[a[0]] = {}
    if b[0] not in detections_reid:
        detections_reid[b[0]] = {}

    # Initialize changed_ids tracking
    if a[0] not in changed_ids:
        changed_ids[a[0]] = []
    if b[0] not in changed_ids:
        changed_ids[b[0]] = []

    # Track processed IDs within each video
    if a[1] not in changed_ids[a[0]] and b[1] not in changed_ids[b[0]]:
        updated_a = filter_and_update_track_id(detections[a[0]], a[1], new_id)
        updated_b = filter_and_update_track_id(detections[b[0]], b[1], new_id)

        # Merge updates into detections_reid (organized by video -> frame)
        for frame, objs in updated_a.items():
            detections_reid[a[0]].setdefault(frame, []).extend(objs)
        for frame, objs in updated_b.items():
            detections_reid[b[0]].setdefault(frame, []).extend(objs)

        # Mark IDs as processed
        changed_ids[a[0]].append(a[1])
        changed_ids[b[0]].append(b[1])

        new_id += 1  # Increment new track_id

print(detections_reid['c015'])

# print(detections_reid['c001'])
save_tracks_to_files(detections_reid,output_dir)