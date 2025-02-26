from src import gaussian_modelling, load_data, metrics, read_data
import os
import cv2
import numpy as np
from tqdm import tqdm

from src.load_data import load_video_frame, load_frames_list

def improved_classify_frame(frame, background_mean, background_variance, threshold_factor=6, min_area=500):
    # Convert inputs to float for precision
    frame_float = frame.astype(np.float32)
    bg_mean_float = background_mean.astype(np.float32)
    sigma = np.sqrt(background_variance.astype(np.float32) + 1e-6)

    # Compute the absolute difference for each channel
    diff = np.abs(frame_float - bg_mean_float)

    # Classify pixel as background if all channels are within threshold_factor * sigma
    within_threshold = diff <= (threshold_factor * sigma)
    background_mask = np.all(within_threshold, axis=2).astype(np.uint8) * 255

    # Derive the foreground mask (where cars should be)
    foreground_mask = cv2.bitwise_not(background_mask)

    # Apply morphological opening to remove small noisy regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)

    # Optionally, remove small blobs using connected component analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    refined_mask = np.zeros_like(cleaned_mask)
    for i in range(1, num_labels):  # Skip the background label 0
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            refined_mask[labels == i] = 255

    return refined_mask

class RealTimeTemporalMedianFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = []

    def update(self, new_mask):
        # Add the new mask to the buffer
        self.buffer.append(new_mask)
        # Keep only the most recent 'window_size' masks
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        # Compute the median across the buffered masks
        median_mask = np.median(np.stack(self.buffer, axis=0), axis=0)
        # Threshold the median result to convert it back to a binary mask
        refined_mask = (median_mask > 127).astype(np.uint8) * 255
        return refined_mask

# Path to the AI City data video
path = "./data/AICity_data/train/S03/c010"

# Path to the annotations of the video
path_annotation = "./data/ai_challenge_s03_c010-full_annotation.xml"
path_detection = ["./data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt",
                  "./data/AICity_data/train/S03/c010/det/det_ssd512.txt",
                  "./data/AICity_data/train/S03/c010/det/det_yolo3.txt"]


video_path = os.path.join(path, "vdo.avi")

gaussian_model = gaussian_modelling.NonRecursiveGaussianModel()

# Determine the total number of frames in the video.
total_frames = load_data.get_total_frames(video_path)

training_end = int(total_frames * 0.25)
training_frames = load_data.load_frames_list(video_path, start=0, end=training_end)
bg_mean, bg_variance = gaussian_model.compute_gaussian_background(training_frames)

print("Gaussian Background parameters calculated succesfully!!!")

test_frames = load_data.load_frames_list(video_path, start=training_end, end=total_frames)
# Load ground truth annotations (assuming XML annotations)
gt_data, _ = read_data.parse_annotations_xml(path_annotation, isGT=True)
# Organize GT per frame; here we assume gt_data is a list of dictionaries with keys "frame" and "bbox"
gt_dict = {}


# Video path
video_path = "./data/AICity_data/train/S03/c010/vdo.avi"
output_dir = "./output_videos"

# Abrir el video con OpenCV VideoCapture
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, training_end)  # Empezar en la zona de test

# Obtener propiedades del video
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)

# Inicializar VideoWriters para cada método
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec AVI
out_mask = cv2.VideoWriter(os.path.join(output_dir, "nonadaptive10_mask.avi"), fourcc, fps, frame_size, isColor=False)
out_frames = cv2.VideoWriter(os.path.join(output_dir, "nonadaptive10_frames.avi"), fourcc, fps, frame_size, isColor=True)



for item in gt_data:
    # Solo consideramos vehículos en movimiento (parked == False)
    if not item.get("parked", False):
        frame_no = item["frame"]
        # Convertir la caja a lista si es un numpy array
        box = item["bbox"][0].tolist() if isinstance(item["bbox"], np.ndarray) else item["bbox"]
        if frame_no in gt_dict:
            gt_dict[frame_no].append(box)
        else:
            gt_dict[frame_no] = [box]

for thrs in [10]:

    temporal_filter = RealTimeTemporalMedianFilter(window_size=5)
    all_pred_boxes = []
    all_gt_boxes = []

    # Loop over test frames (keeping track of frame index)
    for idx, frame_rgb in tqdm(enumerate(test_frames, start=training_end)):

        # Compute the binary background mask using the non-recursive Gaussian model
        background_mask = improved_classify_frame(frame_rgb, bg_mean, bg_variance, threshold_factor=thrs)
        # Apply the causal temporal median filter to refine the mask in real time
        #refined_mask = temporal_filter.update(background_mask)

        # Extract predicted bounding boxes from the foreground mask
        pred_boxes = metrics.extract_bounding_boxes(background_mask, min_area=500)
        gt_boxes = gt_dict.get(idx, [])

        # Append boxes for video-level AP calculation
        all_pred_boxes.append(pred_boxes)
        all_gt_boxes.append(gt_boxes)

        # Evaluate detection at the bounding box level (IoU, precision, recall)
        if gt_boxes:
            precision, recall, iou_list, tp, fp, fn = metrics.evaluate_detections(pred_boxes, gt_boxes, iou_threshold=0.5)
            avg_iou = np.mean(iou_list) if iou_list else 0.0
            #print(f"Frame {idx}: Avg IoU={avg_iou:.2f}")
        #else:
            #print(f"Frame {idx}: No hay GT disponible.")

        # Optionally: Display the frame, mask, and draw bounding boxes on the frame for visualization.
        for box in pred_boxes:
            cv2.rectangle(frame_rgb, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        for box in gt_boxes:
            cv2.rectangle(frame_rgb, (int(np.round(box[0])),int(np.round(box[1]))), (int(np.round(box[2])),int(np.round(box[3]))), (0, 255, 0), 2)

        # cv2.imshow("Frame con Detecciones", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        # cv2.imshow("Máscara de Primer Plano", background_mask)
        out_mask.write(background_mask)
        out_frames.write(frame_rgb)

        #cv2.imshow("Frame with Detections", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        #cv2.imshow("Foreground Mask", background_mask)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC key to exit
            break

    video_ap = metrics.compute_video_average_precision(all_pred_boxes, all_gt_boxes, iou_threshold=0.5)
    print(f"Video mAP (AP for class 'car'): {video_ap:.4f}")

    cap.release()
    out_frames.release()
    out_mask.release()
    cv2.destroyAllWindows()