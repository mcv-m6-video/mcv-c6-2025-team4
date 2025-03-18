import cv2
import numpy as np
import torch
from torchvision import transforms
import sys
import os
from tqdm import tqdm

# --------------------------
# 1. Set up the vehicle detector using Detectron2
# --------------------------
sys.path.append("/home/toukapy/Dokumentuak/Master CV/C6/detectron2")
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

d2_cfg = get_cfg()
d2_cfg.merge_from_file(
    "/home/toukapy/Dokumentuak/Master CV/C6/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
d2_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Assume one class: vehicle
d2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
d2_cfg.MODEL.WEIGHTS = "model_final.pth"  # Update with your Detectron2 weights path
detector = DefaultPredictor(d2_cfg)

# --------------------------
# 2. Set up the FastReID re-identification module
# --------------------------
from fastreid.config import get_cfg as get_fr_cfg
from fastreid.engine.defaults import DefaultPredictor as FRPredictor

fr_cfg = get_fr_cfg()
fr_cfg.merge_from_file(
    "/home/toukapy/Dokumentuak/Master CV/C6/Week4/models/bagtricks_R50-ibn_vehicleid.yml")  # Update with your FastReID config file
fr_cfg.MODEL.WEIGHTS = "/home/toukapy/Dokumentuak/Master CV/C6/Week4/models/vehicleid_bot_R50-ibn.pth"  # Update with your FastReID weights path
fr_cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
fr_predictor = FRPredictor(fr_cfg)

# Define the preprocessing transform expected by FastReID
fr_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),  # Adjust if needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def extract_feature(crop):
    """
    Given a vehicle crop (BGR image), extract a normalized feature vector using FastReID.
    """
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    input_tensor = fr_transform(crop_rgb).unsqueeze(0).to(fr_cfg.MODEL.DEVICE)
    with torch.no_grad():
        features = fr_predictor.model(input_tensor)
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features.cpu().numpy().squeeze()


# --------------------------
# 3. Define chipper and embedder functions
# --------------------------
def chipper(frame, detections):
    """
    Given a frame and a list of detections, return a list of object chips (crops).
    Each detection is expected to be a tuple: ([left, top, w, h], confidence, detection_class)
    """
    chips = []
    for det in detections:
        bbox, conf, det_class = det
        left, top, w, h = bbox
        left, top, w, h = int(left), int(top), int(w), int(h)
        chip = frame[top:top + h, left:left + w]
        chips.append(chip)
    return chips


def embedder(chips):
    """
    Given a list of object chips, compute and return a list of feature embeddings using extract_feature().
    """
    embeddings = []
    for chip in chips:
        if chip.size == 0:
            embeddings.append(np.zeros(512))  # Adjust dimension as needed
        else:
            embeddings.append(extract_feature(chip))
    return embeddings


# --------------------------
# 4. Set up DeepSORT using deep_sort_realtime (using external embeddings)
# --------------------------
from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=5)  # Other parameters can be tuned as needed


# --------------------------
# 5. Function to save tracking results in MOT format (.txt)
# --------------------------
def save_mot_format(tracked_objects, output_mot_path):
    """
    Write tracking results in MOTChallenge format.
    tracked_objects is a dictionary with keys as frame indices and values as dictionaries,
    where each inner dictionary maps track_id to (bbox, frame_index).
    bbox is expected as [left, top, right, bottom].
    """
    with open(output_mot_path, "w") as f:
        for frame_idx, objects in tracked_objects.items():
            for obj_id, (bbox, _) in objects.items():
                x1, y1, x2, y2 = map(int, bbox)
                width, height = x2 - x1, y2 - y1
                # MOT format: frame, id, x, y, w, h, score, class, visibility
                f.write(f"{frame_idx}, {obj_id}, {x1}, {y1}, {width}, {height}, 1, 1, 1\n")


# --------------------------
# 6. (Optional) Unique color generator for visualization
# --------------------------
track_colors = {}


def get_color(track_id):
    if track_id not in track_colors:
        color = np.random.randint(0, 256, size=3).tolist()  # BGR color
        track_colors[track_id] = color
    return track_colors[track_id]


# --------------------------
# 7. Main tracking loop: Detection -> Chipping -> Embedding -> DeepSORT update -> Write results
# --------------------------
if __name__ == "__main__":
    video_path = "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/train/S01/c001/vdo.avi"  # Actualiza la ruta de tu video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Carpeta donde se guardarán los frames
    output_folder = "output_frames"
    import os

    os.makedirs(output_folder, exist_ok=True)

    # Diccionario para almacenar resultados de tracking en formato MOT
    tracked_dict = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detección de objetos con Detectron2 (se esperan cajas en formato [x1, y1, x2, y2])
        outputs = detector(frame)
        instances = outputs["instances"]
        detections = []
        if instances.has("pred_boxes"):
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                # Formato: ([left, top, w, h], confidence, detection_class)
                detections.append(([x1, y1, w, h], score, 0))

        # Extraer chips de objetos con la función chipper
        object_chips = chipper(frame, detections)
        # Calcular embeddings usando la función embedder
        embeds = embedder(object_chips)

        # Actualizar el tracker DeepSORT con detecciones y embeddings
        tracks = tracker.update_tracks(detections, embeds=embeds)
        frame_tracking = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # Formato: [left, top, right, bottom]
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            frame_tracking[track_id] = ([x1, y1, x2, y2], frame_idx)
            # Dibujar la caja y el ID sobre el frame
            color = get_color(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        tracked_dict[frame_idx] = frame_tracking

        # Guardar el frame procesado en la carpeta de salida
        frame_filename = os.path.join(output_folder, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_idx += 1
        print(f"Procesado frame {frame_idx}")

    cap.release()

    # Verificar que se hayan creado los frames correctamente
    saved_frames = os.listdir(output_folder)
    print(f"Se han guardado {len(saved_frames)} frames en la carpeta '{output_folder}'.")

    # Guardar los resultados de tracking en formato MOT (.txt)
    mot_output_path = "output_tracking.txt"
    save_mot_format(tracked_dict, mot_output_path)
    print("Tracking completado. Frames y resultados MOT guardados.")

