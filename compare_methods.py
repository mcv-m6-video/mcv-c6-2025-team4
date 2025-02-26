import cv2
import os
import numpy as np
from src import load_data, read_data, metrics

output_dir = "./output_videos"
os.makedirs(output_dir, exist_ok=True)

# Video path
video_path = "./data/AICity_data/train/S03/c010/vdo.avi"
path_annotation = "./data/ai_challenge_s03_c010-full_annotation.xml"

# Load total frames
total_frames = load_data.get_total_frames(video_path)
training_end = int(total_frames * 0.25)

# Load ground truth annotations
gt_data, _ = read_data.parse_annotations_xml(path_annotation, isGT=True)
gt_dict = {}
for item in gt_data:
    frame_no = item["frame"]
    box = item["bbox"][0].tolist() if isinstance(item["bbox"], np.ndarray) else item["bbox"]
    gt_dict.setdefault(frame_no, []).append(box)

# Initialize methods
mog = cv2.bgsegm.createBackgroundSubtractorMOG()
mog2 = cv2.createBackgroundSubtractorMOG2()
lsbp = cv2.bgsegm.createBackgroundSubtractorLSBP()

# Open video
cap = cv2.VideoCapture(video_path)

# Obtain video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)

# Initialize VideoWriters for each method
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec AVI

out_mog = cv2.VideoWriter(os.path.join(output_dir, "MOG.avi"), fourcc, fps, frame_size, isColor=False)
out_mog2 = cv2.VideoWriter(os.path.join(output_dir, "MOG2.avi"), fourcc, fps, frame_size, isColor=False)
out_lsbp = cv2.VideoWriter(os.path.join(output_dir, "LSBP.avi"), fourcc, fps, frame_size, isColor=False)

print("Processing frames and calculating mAP@50...")

all_pred_boxes = {"MOG": [], "MOG2": [], "LSBP": []}
all_gt_boxes = {"MOG": [], "MOG2": [], "LSBP": []}

frame_idx = 0
while cap.isOpened():
    ret, frame_rgb = cap.read()
    if not ret:
        break  # end of the video

    gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction methods
    fg_mask_mog = mog.apply(gray_frame)
    fg_mask_mog2 = mog2.apply(gray_frame)
    fg_mask_lsbp = lsbp.apply(gray_frame)

    # Write frame in the output video
    out_mog.write(fg_mask_mog)
    out_mog2.write(fg_mask_mog2)
    out_lsbp.write(fg_mask_lsbp)

    gt_boxes = gt_dict.get(frame_idx, [])

    for method_name, fg_mask in [
        ("MOG", fg_mask_mog),
        ("MOG2", fg_mask_mog2),
        ("LSBP", fg_mask_lsbp),
    ]:
        # Extract bounding boxes
        pred_boxes = metrics.extract_bounding_boxes(fg_mask, min_area=1500)
        
        all_pred_boxes[method_name].append(pred_boxes)
        all_gt_boxes[method_name].append(gt_boxes)

    frame_idx += 1  

cap.release()
out_mog.release()
out_mog2.release()
out_lsbp.release()

# Obtain mAP for each method
final_map50 = {}
for method in ["MOG", "MOG2", "LSBP"]:
    map50 = metrics.compute_video_average_precision(all_pred_boxes[method], all_gt_boxes[method], iou_threshold=0.5)
    final_map50[method] = map50

# print results
print("\nFinal mAP@50 Results:")
for method, map50 in final_map50.items():
    print(f"{method}: {map50:.4f}")

print(f"\nProcessing completed! Videos saved in {output_dir}")
