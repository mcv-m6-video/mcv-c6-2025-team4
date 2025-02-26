import os
import cv2
import numpy as np
import imageio

from src import gaussian_modelling, load_data, metrics, read_data
from src.load_data import load_video_frame, load_frames_list

# Path to the AI City dataset video
path = "./data/AICity_data/train/S03/c010"

# Path to the annotations of the video
path_annotation = "./data/ai_challenge_s03_c010-full_annotation.xml"
path_detection = [
    "./data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt",
    "./data/AICity_data/train/S03/c010/det/det_ssd512.txt",
    "./data/AICity_data/train/S03/c010/det/det_yolo3.txt"
]

video_path = os.path.join(path, "vdo.avi")

# Determine the total number of frames in the video
total_frames = load_data.get_total_frames(video_path)

# Use 25% of the frames to initialize the background model
training_end = int(total_frames * 0.25)
training_frames = load_frames_list(video_path, start=0, end=training_end)
print("Adaptive background parameters initialized successfully!!!")

# Load ground truth annotations (assuming XML format)
gt_data, _ = read_data.parse_annotations_xml(path_annotation, isGT=True)

# Organize ground truth data by frame number
gt_dict = {}
for item in gt_data:
    # Only consider moving vehicles (ignore parked cars)
    if not item.get("parked", False):
        frame_no = item["frame"]
        # Convert bounding box to a list if it is a NumPy array
        box = item["bbox"][0].tolist() if isinstance(item["bbox"], np.ndarray) else item["bbox"]
        if frame_no in gt_dict:
            gt_dict[frame_no].append(box)
        else:
            gt_dict[frame_no] = [box]

# Lists to store predicted and ground truth bounding boxes
ap_list = []
all_pred_boxes_gmm = []
all_gt_boxes_gmm = []

# Initialize Gaussian Mixture Model (GMM) background subtractor
gmm_model = gaussian_modelling.GMMBackgroundSubtractor(history=500, varThreshold=40, detectShadows=True)

# Define morphological operation kernels
kernel_open = np.ones((5, 5), np.uint8)  # Remove small noise
kernel_close = np.ones((9, 9), np.uint8)  # Merge regions
kernel_dilate = np.ones((3, 3), np.uint8)  # Expand foreground slightly

# Load test frames (remaining 75% of the video)
test_frames = load_frames_list(video_path, start=training_end, end=total_frames)
for idx, frame_rgb in enumerate(test_frames, start=training_end):
    # Apply GMM background subtraction to detect moving objects
    fg_mask = gmm_model.apply(frame_rgb)

    # Remove shadows (gray pixels with value 127)
    fg_mask[fg_mask == 127] = 0

    # Convert to binary mask: retain high-confidence foreground pixels
    _, fg_mask_binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to refine mask
    fg_mask_binary = cv2.morphologyEx(fg_mask_binary, cv2.MORPH_OPEN, kernel_open)  # Remove noise
    fg_mask_binary = cv2.morphologyEx(fg_mask_binary, cv2.MORPH_CLOSE, kernel_close)  # Merge small regions
    fg_mask_binary = cv2.dilate(fg_mask_binary, kernel_dilate, iterations=5)  # Expand detected objects

    # Apply Gaussian blur to smooth mask edges
    fg_mask_binary = cv2.GaussianBlur(fg_mask_binary, (9, 9), 0)

    # Convert mask to 8-bit format for visualization
    mask_8bit = fg_mask_binary.astype(np.uint8)

    # Convert grayscale mask to color for visualization
    mask_colored = cv2.cvtColor(mask_8bit, cv2.COLOR_GRAY2BGR)

    # Extract predicted bounding boxes from the foreground mask
    pred_boxes = metrics.extract_bounding_boxes(fg_mask_binary, min_area=1000)
    gt_boxes = gt_dict.get(idx, [])

    # Store predictions and ground truth for later evaluation
    all_pred_boxes_gmm.append(pred_boxes)
    all_gt_boxes_gmm.append(gt_boxes)

    # Evaluate detection performance (IoU, precision, recall)
    if gt_boxes:
        precision, recall, iou_list, tp, fp, fn = metrics.evaluate_detections(pred_boxes, gt_boxes, iou_threshold=0.5)
        avg_iou = np.mean(iou_list) if iou_list else 0.0
        print(f"Frame {idx}: Avg IoU={avg_iou:.2f}")
    else:
        print(f"Frame {idx}: No ground truth available.")

    # Draw predicted bounding boxes in green
    for box in pred_boxes:
        cv2.rectangle(mask_colored, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)  # Green

    # Draw ground truth bounding boxes in red
    for gt_box in gt_boxes:
        cv2.rectangle(mask_colored, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (255, 0, 0), 2)  # Red

    # Display the mask with detected objects
    cv2.imshow("Mask with Detections", mask_colored)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # Press ESC to exit
        break

# Release all OpenCV windows
cv2.destroyAllWindows()

# Compute mean Average Precision (mAP) for object detection using the GMM method
video_ap_gmm = metrics.compute_video_average_precision(all_pred_boxes_gmm, all_gt_boxes_gmm, iou_threshold=0.5)
print(f"GMM Method Video mAP (Cars): {video_ap_gmm:.4f}")

