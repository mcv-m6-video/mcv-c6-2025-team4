import cv2
import os
from detectron2.structures import BoxMode
import numpy as np

path_seq="C:/Users/User/Downloads/aic19-track1-mtmc-train/train/S01/"
vid_folder=os.listdir(path_seq)
dataset_dicts = []

for vid in vid_folder:
    gt_file=path_seq+vid+"/gt/gt.txt"
    gt_dict = {}

    # Read the ground truth file
    with open(gt_file, "r") as f:
        for line in f:
            values = list(map(int, line.strip().split(',')))
            frame, ID, left, top, width, height, *_ = values
            
            # Convert [left, top, width, height] -> [x_min, y_min, x_max, y_max]
            bbox = [left, top, left + width, top + height]
            
            # Store in dictionary grouped by frame
            if frame not in gt_dict:
                gt_dict[frame] = []
            gt_dict[frame].append(bbox)

    # Process each frame
    for frame in sorted(gt_dict.keys()):
        record = {}

        # Image path
        record["file_name"] = path_seq+vid+"/frames/"+ f"frame_{str(frame).zfill(6)}.jpg"
        print(record["file_name"])
        # Read image to get dimensions
        image = cv2.imread(record["file_name"])
        if image is None:
            print(f"Warning: Image {record['file_name']} not found.")
            continue
        
        h, w, _ = np.shape(image)

        record["image_id"] = frame
        record["height"] = h
        record["width"] = w

        # Annotations
        objs = []
        for bbox in gt_dict[frame]:
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
            }
            objs.append(obj)
        print(objs)
        record["annotations"] = objs
        dataset_dicts.append(record)