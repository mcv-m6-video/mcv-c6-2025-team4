import os
import cv2
import numpy as np
from tqdm import tqdm
import imageio

from src import gaussian_modelling, load_data, metrics, read_data
from src.load_data import load_video_frame, load_frames_list


def improved_classify_frame(frame, background_mean, background_variance, threshold_factor=6, min_area=500):
    """
    Classifies each pixel in an image as background or foreground using a Gaussian model.

    Parameters:
    - frame: np.ndarray
        The input image frame in RGB format.
    - background_mean: np.ndarray
        The precomputed mean background model.
    - background_variance: np.ndarray
        The precomputed variance background model.
    - threshold_factor: float, optional
        The threshold factor for classification (default is 6).
    - min_area: int, optional
        The minimum area for connected components to be retained (default is 500).

    Returns:
    - refined_mask: np.ndarray
        A binary mask where foreground objects are white (255) and background is black (0).
    """
    # Convert inputs to float for precision
    frame_float = frame.astype(np.float32)
    bg_mean_float = background_mean.astype(np.float32)
    sigma = np.sqrt(background_variance.astype(np.float32) + 1e-6)

    # Compute the absolute difference for each channel
    diff = np.abs(frame_float - bg_mean_float)

    # Classify a pixel as background if all channels are within threshold_factor * sigma
    within_threshold = diff <= (threshold_factor * sigma)
    background_mask = np.all(within_threshold, axis=2).astype(np.uint8) * 255

    # Compute the foreground mask (where objects such as cars should be)
    foreground_mask = cv2.bitwise_not(background_mask)

    # Apply morphological opening to remove small noisy regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)

    # Remove small blobs using connected component analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    refined_mask = np.zeros_like(cleaned_mask)

    for i in range(1, num_labels):  # Skip the background label 0
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            refined_mask[labels == i] = 255

    return refined_mask


# Paths to dataset and annotations
path = "./data/AICity_data/train/S03/c010"
path_annotation = "./data/ai_challenge_s03_c010-full_annotation.xml"
path_detection = [
    "./data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt",
    "./data/AICity_data/train/S03/c010/det/det_ssd512.txt",
    "./data/AICity_data/train/S03/c010/det/det_yolo3.txt"
]

video_path = os.path.join(path, "vdo.avi")

# Initialize the Gaussian background model
gaussian_model = gaussian_modelling.NonRecursiveGaussianModel()

# Get the total number of frames in the video
total_frames = load_data.get_total_frames(video_path)

# Use 25% of the video frames for training the background model
training_end = int(total_frames * 0.25)
training_frames = load_data.load_frames_list(video_path, start=0, end=training_end)
bg_mean, bg_variance = gaussian_model.compute_gaussian_background(training_frames)

print("Gaussian Background parameters calculated successfully!")

# Load the remaining frames for testing
test_frames = load_data.load_frames_list(video_path, start=training_end, end=total_frames)

# Load ground truth annotations from XML file
gt_data, _ = read_data.parse_annotations_xml(path_annotation, isGT=True)

# Organize ground truth data by frame number
gt_dict = {}
for item in gt_data:
    # Only consider moving vehicles (ignore parked cars)
    if not item.get("parked", False):
        frame_no = item["frame"]
        # Convert bounding box to list if it is a NumPy array
        box = item["bbox"][0].tolist() if isinstance(item["bbox"], np.ndarray) else item["bbox"]
        if frame_no in gt_dict:
            gt_dict[frame_no].append(box)
        else:
            gt_dict[frame_no] = [box]

# Lists to store all predicted and ground truth bounding boxes
all_pred_boxes = []
all_gt_boxes = []

# Process each test frame
for idx, frame_rgb in enumerate(test_frames, start=training_end):

    # Compute the binary foreground mask using the Gaussian model
    background_mask = improved_classify_frame(frame_rgb, bg_mean, bg_variance, threshold_factor=4)
    mask_8bit = background_mask.astype(np.uint8)
    mask_colored = cv2.cvtColor(mask_8bit, cv2.COLOR_GRAY2BGR)

    # Extract predicted bounding boxes from the foreground mask
    pred_boxes = metrics.extract_bounding_boxes(background_mask, min_area=500)
    gt_boxes = gt_dict.get(idx, [])

    # Store predictions and ground truth for later AP computation
    all_pred_boxes.append(pred_boxes)
    all_gt_boxes.append(gt_boxes)

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
        cv2.rectangle(mask_colored, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (255, 0, 0),
                      2)  # Red

    # Display results
    cv2.imshow("Frame with Detections", cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR))
    cv2.imshow("Foreground Mask", background_mask)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # Press ESC to exit
        break

# Compute mean Average Precision (mAP) for object detection
video_ap = metrics.compute_video_average_precision(all_pred_boxes, all_gt_boxes, iou_threshold=0.5)
print(f"Video mAP (AP for class 'car'): {video_ap:.4f}")

cv2.destroyAllWindows()
