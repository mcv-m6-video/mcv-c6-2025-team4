import cv2
import numpy as np
import os
from tqdm import tqdm

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
    
def load_predictions(file_path):

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
    if track_id not in pred_boxes:
        # Generate a random BGR color (values 0-255)
        color = np.random.randint(0, 256, size=3).tolist()
        pred_boxes[track_id] = color
    return pred_boxes[track_id]


seq='S03/'
vid='c015'
gt_boxes = load_ground_truth('E:/aic19-track1-mtmc-train/train/'+seq+vid+'/gt/gt.txt')
pred_boxes=load_predictions("E:/aic19-track1-mtmc-train/train/"+seq+vid+"/pred/predictions.txt")
# pred_boxes=load_predictions("C:/Users/User/Documents/GitHub/mcv-c6-2025-team4/Week4/final_tracks/"+vid+"_final_tracks.txt")

frame_path="E:/aic19-track1-mtmc-train/train/"+seq+vid+"/frames"
frame=cv2.imread(frame_path+"/frame_000000.jpg")
fps = 8
frame_width = int(np.shape(frame)[1])
frame_height = int(np.shape(frame)[0])


# Inicializar escritor de video
output_path = "C:/Users/User/Documents/GitHub/mcv-c6-2025-team4/Week4/final_tracks/"+vid+'.mp4'
# 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_total = int(len(os.listdir(frame_path)))
sample_rate = 1

selected_frames = range(0, frame_total, sample_rate)
for frame_idx in tqdm(selected_frames, desc="Processing video"):

    frame=cv2.imread(frame_path+"/frame_"+str(frame_idx).zfill(6)+".jpg")

    if frame_idx in pred_boxes.keys():
        for obj in pred_boxes[frame_idx]:
            # print(obj)
            track_id,x1, y1, x2, y2 =  map(int, obj)
            
            
            # print(color)
            cv2.rectangle(frame, (x1, y1), (x1+x2, y1+y2), (0,255,0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    # Optionally, draw ground truth boxes if available
    if frame_idx in gt_boxes:
        for gt in gt_boxes[frame_idx]:
            _, x, y, w, h = gt
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    out.write(frame)

out.release()