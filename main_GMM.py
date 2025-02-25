from src import gaussian_modelling, load_data, metrics, read_data
import os
import cv2
import numpy as np

from src.load_data import load_video_frame, load_frames_list

def improved_classify_frame(frame, background_mean, background_variance, threshold_factor=8, min_area=500):
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

# Determine the total number of frames in the video.
total_frames = load_data.get_total_frames(video_path)

# Creamos la instancia del modelo gaussiano adaptativo

# Determinamos el número total de frames del video
total_frames = load_data.get_total_frames(video_path)

# Usamos el 25% de los frames para inicializar el modelo de fondo
training_end = int(total_frames * 0.25)
training_frames = load_frames_list(video_path, start=0, end=training_end)
print("Parámetros de fondo adaptativo inicializados correctamente!!!")

# Cargamos las anotaciones de ground truth (suponiendo formato XML)
gt_data, _ = read_data.parse_annotations_xml(path_annotation, isGT=True)
# Organizamos el GT por frame: se crea un diccionario con la lista de bounding boxes para cada frame.
gt_dict = {}
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

# Opcional: instanciar el filtro temporal (si se desea refinar la máscara en tiempo real)
temporal_filter = RealTimeTemporalMedianFilter(window_size=5)

ap_list = []
all_pred_boxes_gmm = []
all_gt_boxes_gmm = []

f = open("GMM.txt", "a")

for thrs in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:

    gmm_model = gaussian_modelling.GMMBackgroundSubtractor(history=500, varThreshold=16, detectShadows=True)

    # Procesamos los frames de prueba (el 75% restante)
    test_frames = load_frames_list(video_path, start=training_end, end=total_frames)
    for idx, frame_rgb in enumerate(test_frames, start=training_end):
        # Apply GMM background subtraction
        fg_mask = gmm_model.apply(frame_rgb)

        # Optionally remove shadows (gray pixels with value 127)
        _, fg_mask_binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Extract bounding boxes
        pred_boxes = metrics.extract_bounding_boxes(fg_mask_binary, min_area=500)
        gt_boxes = gt_dict.get(idx, [])

        # Append for later evaluation
        all_pred_boxes_gmm.append(pred_boxes)
        all_gt_boxes_gmm.append(gt_boxes)
        if gt_boxes:
            precision, recall, iou_list, tp, fp, fn = metrics.evaluate_detections(pred_boxes, gt_boxes, iou_threshold=0.5)
            avg_iou = np.mean(iou_list) if iou_list else 0.0
            #print(f"Frame {idx}: Avg IoU={avg_iou:.2f}")
            #f.write(f"{idx}: Avg IoU={avg_iou:.2f}\n")
        #else:
            #print(f"Frame {idx}: No hay GT disponible.")

        # Visualization (optional)
        for box in pred_boxes:
            cv2.rectangle(frame_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        #cv2.imshow("GMM Detection", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

    # Compute mAP after all frames processed
    video_ap_gmm = metrics.compute_video_average_precision(all_pred_boxes_gmm, all_gt_boxes_gmm, iou_threshold=0.5)
    print(f"GMM Method Video mAP (Cars): {video_ap_gmm:.4f}")