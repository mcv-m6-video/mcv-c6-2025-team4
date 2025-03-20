import os
import numpy as np
import cv2
from itertools import product
from math import radians, sin, cos, sqrt, atan2


# =============================================================================
# Data Loading and Calibration Functions
# =============================================================================
def load_timestamps(timestamp_file):
    """ Carga los timestamps de inicio de cada cámara desde el archivo. """
    timestamps = {}
    with open(timestamp_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                cam_id = parts[0]
                timestamps[cam_id] = float(parts[1])
    return timestamps


def load_calibration(calib_file):
    """ Carga la matriz de homografía desde el archivo de calibración. """
    with open(calib_file, "r") as f:
        line = f.readline().strip()
    H = np.array([list(map(float, row.split())) for row in line.split(";")])
    return H


def load_predictions(file_path, sequence):
    """
    Carga las predicciones (detecciones) desde archivos de texto.
    Formato: frame, track_id, x, y, w, h
    """
    pred_dict = {}
    for vid in sequence:
        data = {}
        with open(file_path + vid + ".txt", "r") as f:
            for line in f:
                parts = line.strip().split(",")
                frame, track_id, x, y, w, h = map(int, parts[:6])
                if frame not in data:
                    data[frame] = []
                data[frame].append([track_id, x, y, w, h])
        pred_dict[vid] = data
    return pred_dict


def get_max_frame(gt_dict):
    last = []
    for vid in gt_dict:
        last.append(list(gt_dict[vid].keys())[-1])
    return max(last)


# =============================================================================
# World Coordinate Conversion Functions
# =============================================================================
def project_to_world(homography, x, y):
    """ Project image pixel coordinates (x, y) to world coordinates """
    point = np.array([x, y, 1])
    projected = np.dot(homography, point)
    return projected[:2] / projected[2]


def get_world_coordinates(detections, homographies, start_frames, distortions):
    """
    Convierte las coordenadas de las cajas (detecciones) a coordenadas del mundo
    utilizando la homografía y parámetros de distorsión.
    """
    world_positions = {}
    # Define la matriz intrínseca (ajusta según tus datos)
    K = np.array([[1000, 0, 640],
                  [0, 1000, 480],
                  [0, 0, 1]])
    for vid, dets in detections.items():
        print("Procesando world coordinates para", vid)
        world_positions[vid] = {}
        H_inv = homographies[vid]
        for frame, info in dets.items():
            if frame not in world_positions[vid]:
                world_positions[vid][frame] = []
            for item in info:
                track_id, x, y, w, h = item
                x_center = x + w / 2
                y_center = y + h / 2
                # Corrige la distorsión si es aplicable
                if np.shape(distortions[vid]) != ():
                    undistorted = cv2.undistortPoints(np.array([[x_center, y_center]], dtype=np.float32), K,
                                                      distortions[vid], P=K)
                    undistorted = undistorted.reshape(-1, 2)
                    x_center, y_center = undistorted[0, 0], undistorted[0, 1]
                world_x, world_y = project_to_world(H_inv, x_center, y_center)
                world_positions[vid][frame].append([track_id, world_x, world_y, x, y, w, h])
    return world_positions


# =============================================================================
# Frame Alignment and Matching Functions
# =============================================================================
def find_corresponding_frame(vid1, vid2, frame1, start_frames, fps):
    """
    Calcula el frame correspondiente en vid2 a partir del frame en vid1 usando los start_times y FPS.
    """
    f1 = start_frames[vid1]
    f2 = start_frames[vid2]
    time_elapsed = (frame1 - f1) / fps[vid1]
    frame2 = f2 + time_elapsed * fps[vid2]
    return round(frame2)


def find_order(start_frames):
    """ Ordena las cámaras según su start_time en orden descendente. """
    return dict(sorted(start_frames.items(), key=lambda item: item[1], reverse=True))


def haversine_distance_meters(coord1, coord2):
    """ Calcula la distancia Haversine en metros entre dos pares (lat, lon) """
    R = 6371000
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def find_potential_matches(coords1, coords2, track1s, track2s, bboxes1, bboxes2, threshold=2.0):
    """
    Encuentra coincidencias entre detecciones basándose en la distancia Haversine.
    """
    matches = []
    for (track1, coord1, box1), (track2, coord2, box2) in product(zip(track1s, coords1, bboxes1),
                                                                  zip(track2s, coords2, bboxes2)):
        dist = haversine_distance_meters(np.array(coord1), np.array(coord2))
        if dist <= threshold:
            matches.append((track1, track2, box1, box2))
    return matches


def match_across_cameras(world_positions, start_frames, threshold=2.0, fps=None):
    """
    Realiza matching de detecciones entre cámaras usando frames alineados y coordenadas del mundo.
    Devuelve una lista de coincidencias donde cada elemento es:
      [[vid, track_id, frame, bbox], [vid, track_id, frame, bbox]]
    """
    final_tracks = []
    vid_order = find_order(start_frames)
    for vid1 in vid_order:
        for frame1, info1 in world_positions[vid1].items():
            for vid2 in world_positions:
                if vid1 == vid2:
                    continue
                frame2 = find_corresponding_frame(vid1, vid2, frame1, start_frames, fps)
                if frame2 < 0 or frame2 not in world_positions[vid2]:
                    continue
                info2 = world_positions[vid2][frame2]
                # Extrae IDs, coordenadas y cajas
                track1s, coords1, bboxes1 = zip(
                    *[(obj[0], [obj[1], obj[2]], [obj[3], obj[4], obj[5], obj[6]]) for obj in info1])
                track2s, coords2, bboxes2 = zip(
                    *[(obj[0], [obj[1], obj[2]], [obj[3], obj[4], obj[5], obj[6]]) for obj in info2])
                matches = find_potential_matches(coords1, coords2, track1s, track2s, bboxes1, bboxes2,
                                                 threshold=threshold)
                for track1, track2, bbox1, bbox2 in matches:
                    key1 = [vid1, track1, frame1, bbox1]
                    key2 = [vid2, track2, frame2, bbox2]
                    final_tracks.append([key1, key2])
    return final_tracks


# =============================================================================
# Re-Identification: Update Track IDs Across Cameras (without UnionFind)
# =============================================================================
def filter_and_update_track_id(data, old_id, new_id):
    """
    Crea un nuevo diccionario con solo las detecciones con el old_id y actualiza a new_id.
    """
    updated_data = {}
    for frame, objects in data.items():
        # Realiza una copia profunda de los objetos que coinciden con el old_id
        updated_objects = [obj[:] for obj in objects if obj[0] == old_id]
        for obj in updated_objects:
            obj[0] = new_id
        if updated_objects:
            updated_data[frame] = updated_objects
    return updated_data


# =============================================================================
# Saving Functions
# =============================================================================
def save_tracks_to_files(final_tracks, output_folder):
    """
    Guarda los tracks finales por cámara en archivos.
    Formato: frame, track_id, x, y, w, h
    """
    os.makedirs(output_folder, exist_ok=True)
    for vid, frames in final_tracks.items():
        output_file = os.path.join(output_folder, f"{vid}_final_tracks.txt")
        with open(output_file, "w") as f:
            for frame, objects in sorted(frames.items()):
                for obj in objects:
                    track_id, x, y, w, h = obj
                    f.write(f"{frame},{track_id},{x},{y},{w},{h}\n")
        print(f"Saved: {output_file}")


# =============================================================================
# Main: Multi-Camera Re-Identification without UnionFind
# =============================================================================
if __name__ == "__main__":
    # Parámetros y secuencia
    seq = 'S03/'
    videos = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    fps = {'c010': 10, 'c011': 10, 'c012': 10, 'c013': 10, 'c014': 10, 'c015': 8}
    output_dir = "./final_tracks/"

    # Rutas de archivos (ajusta según tu estructura)
    timestamp_file = "E:/aic19-track1-mtmc-train/cam_timestamp/" + seq.split('/')[0] + '.txt'
    calib_base = "E:/aic19-track1-mtmc-train/train/" + seq
    predictions_path = "C:/Users/User/Documents/GitHub/mcv-c6-2025-team4/Week4/"  # Ruta base para predicciones
    # Cargar start times, calibraciones y predicciones
    start_frames = load_timestamps(timestamp_file)
    homographies, distortions = load_homography(calib_base, videos)
    detections = load_predictions(predictions_path, videos)
    # (Opcional) Cargar ground truth si se necesita
    # gt_dict = load_ground_truth("E:/aic19-track1-mtmc-train/train/" + seq, videos)
    # max_frame = get_max_frame(gt_dict)

    # Convertir bounding boxes a coordenadas del mundo
    world_positions = get_world_coordinates(detections, homographies, start_frames, distortions)

    # Realizar matching entre cámaras (por ejemplo, usando un threshold de 30 metros)
    matches = match_across_cameras(world_positions, start_frames, threshold=30, fps=fps)

    # =============================================================================
    # Re-Identification Update: Usar los matches para actualizar los IDs de track globales
    # =============================================================================
    changed_ids = {}  # Para seguir qué IDs ya han sido procesados por video
    new_id = 1  # Contador para el nuevo ID global
    detections_reid = {}  # Diccionario final: video -> {frame: [detecciones actualizadas]}

    for element in matches:
        a, b = element[0], element[1]  # Cada elemento: ([vid, track_id, frame, bbox], [vid, track_id, frame, bbox])
        # Asegurar que existe la estructura en detections_reid
        if a[0] not in detections_reid:
            detections_reid[a[0]] = {}
        if b[0] not in detections_reid:
            detections_reid[b[0]] = {}
        # Inicializar changed_ids para cada video
        if a[0] not in changed_ids:
            changed_ids[a[0]] = []
        if b[0] not in changed_ids:
            changed_ids[b[0]] = []
        # Procesar solo si ninguno de los IDs ya ha sido actualizado
        if a[1] not in changed_ids[a[0]] and b[1] not in changed_ids[b[0]]:
            # (Opcional) Para depuración: imprimir coincidencias en c015
            if a[0] == 'c015' or b[0] == 'c015':
                print("Match:", a, b)
            updated_a = filter_and_update_track_id(detections[a[0]], a[1], new_id)
            updated_b = filter_and_update_track_id(detections[b[0]], b[1], new_id)
            # Fusionar las detecciones actualizadas en detections_reid
            for frame, objs in updated_a.items():
                detections_reid[a[0]].setdefault(frame, []).extend(objs)
            for frame, objs in updated_b.items():
                detections_reid[b[0]].setdefault(frame, []).extend(objs)
            changed_ids[a[0]].append(a[1])
            changed_ids[b[0]].append(b[1])
            new_id += 1

    print("Updated detections for c015:", detections_reid.get('c015', {}))

    # Guardar los tracks finales actualizados a archivos
    save_tracks_to_files(detections_reid, output_dir)


