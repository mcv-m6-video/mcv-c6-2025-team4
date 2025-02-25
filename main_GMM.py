from src import gaussian_modelling, load_data, metrics, read_data
import os
import cv2
import numpy as np
import imageio

from src.load_data import load_video_frame, load_frames_list

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

ap_list = []
all_pred_boxes_gmm = []
all_gt_boxes_gmm = []

# Define the output GIF path
gif_path = "plots/gmm/foreground_mask_sequence_gmm.gif"

# List to store frames for the GIF
gif_frames = []

output_video_path = "plots/gmm/foreground_mask_sequence_gmm_nonearby.avi"
frame_width, frame_height = 800, 512  # Adjust as needed
fps = 30  # Adjust based on your needs

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Use 'MP4V' for .mp4 output
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255,)  # White text for grayscale image
thickness = 2
position = (20, 40)  # Text position


#f = open("GMM_40_4000.txt", "w+")

#for thrs in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:

gmm_model = gaussian_modelling.GMMBackgroundSubtractor(history=500, varThreshold=40, detectShadows=True)

# Procesamos los frames de prueba (el 75% restante)
test_frames = load_frames_list(video_path, start=training_end, end=total_frames)
for idx, frame_rgb in enumerate(test_frames, start=training_end):
    # Apply GMM background subtraction
    fg_mask = gmm_model.apply(frame_rgb)

    # Optionally remove shadows (gray pixels with value 127)
    _, fg_mask_binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    mask_8bit = (fg_mask_binary).astype(np.uint8)

    mask_colored = cv2.cvtColor(mask_8bit, cv2.COLOR_GRAY2BGR)

    # Extract bounding boxes
    pred_boxes = metrics.extract_bounding_boxes(fg_mask_binary, min_area=500)
    gt_boxes = gt_dict.get(idx, [])

    # Append for later evaluation
    all_pred_boxes_gmm.append(pred_boxes)
    all_gt_boxes_gmm.append(gt_boxes)
    if gt_boxes:
        precision, recall, iou_list, tp, fp, fn = metrics.evaluate_detections(pred_boxes, gt_boxes, iou_threshold=0.5)
        avg_iou = np.mean(iou_list) if iou_list else 0.0
        print(f"Frame {idx}: Avg IoU={avg_iou:.2f}")
        #f.write(f"{idx}: Avg IoU={avg_iou:.2f}\n")
        cv2.putText(mask_colored, f"Frame {idx}", position, font, font_scale, font_color,
                    thickness,
                    cv2.LINE_AA)
    else:
        print(f"Frame {idx}: No hay GT disponible.")
        cv2.putText(mask_colored, f"Frame {idx}. No Ground Truth", position, font, font_scale,
                    font_color, thickness,
                    cv2.LINE_AA)

    # Dibujar las bounding boxes de las predicciones en verde
    for box in pred_boxes:
        #print("Drawing prediction box")
        cv2.rectangle(mask_colored, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)  # Verde

    # Dibujar las bounding boxes del ground truth en rojo
    for gt_box in gt_boxes:
        #print("Drawing ground truth box")
        cv2.rectangle(mask_colored, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (255, 0, 0), 2)  # Rojo

    resized_mask = cv2.resize(mask_colored, (800, 512))  # Adjust size as needed

    out.write(resized_mask)

    # Mostrar la máscara con los bounding boxes dibujados
    #cv2.imshow("Mascara con Detecciones", fg_mask_colored)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
out.release()


# Compute mAP after all frames processed
video_ap_gmm = metrics.compute_video_average_precision(all_pred_boxes_gmm, all_gt_boxes_gmm, iou_threshold=0.5)
print(f"GMM Method Video mAP (Cars): {video_ap_gmm:.4f}")

#imageio.mimsave(gif_path, gif_frames, duration=0.05)  # Adjust duration for speed