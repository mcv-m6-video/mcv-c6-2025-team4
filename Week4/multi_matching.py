import numpy as np
import cv2
import os


# =============================================================================
# Funciones de carga y procesamiento
# =============================================================================

def load_tracking_results(mot_file):
    """
    Lee un archivo MOT (formato: frame, id, x, y, w, h, score, class, visibility)
    y retorna un diccionario con: frame -> list de detecciones.
    Cada detección es un diccionario con { 'track_id': id, 'bbox': [x, y, w, h] }.
    """
    tracking = {}
    with open(mot_file, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            frame = int(parts[0].strip())
            tid = int(parts[1].strip())
            x = float(parts[2].strip())
            y = float(parts[3].strip())
            w = float(parts[4].strip())
            h = float(parts[5].strip())
            bbox = [x, y, w, h]
            if frame not in tracking:
                tracking[frame] = []
            tracking[frame].append({'track_id': tid, 'bbox': bbox})
    return tracking


def aggregate_tracklets(tracking, fps, start_time):
    """
    Agrupa los resultados de tracking por track_id.
    Calcula para cada tracklet:
      - Lista de frames en los que aparece.
      - Lista de centros (a partir de cada bbox).
      - Tiempo de inicio y fin en segundos (usando fps y el timestamp de inicio).
      - Centro promedio (avg_center).

    Retorna un diccionario: track_id -> { 'frames': [...],
                                           'centers': [...],
                                           'start_time': t_ini,
                                           'end_time': t_fin,
                                           'avg_center': np.array([x, y]) }
    """
    tracklets = {}
    for frame, detections in tracking.items():
        for det in detections:
            tid = det['track_id']
            bbox = det['bbox']  # [x, y, w, h]
            center = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
            if tid not in tracklets:
                tracklets[tid] = {'frames': [], 'centers': []}
            tracklets[tid]['frames'].append(frame)
            tracklets[tid]['centers'].append(center)
    for tid, data in tracklets.items():
        frames_arr = np.array(data['frames'])
        data['start_time'] = start_time + frames_arr.min() / fps
        data['end_time'] = start_time + frames_arr.max() / fps
        data['avg_center'] = np.mean(data['centers'], axis=0)
    return tracklets


def load_timestamps(timestamp_file):
    """
    Lee el archivo de timestamps, que tiene líneas del tipo:
         c001 0
         c002 1.640
    Retorna un diccionario: cam_id -> timestamp (float, en segundos)
    """
    timestamps = {}
    with open(timestamp_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                cam_id = parts[0]
                ts = float(parts[1])
                timestamps[cam_id] = ts
    return timestamps


def load_calibration(calib_file):
    """
    Lee el archivo de calibración y retorna la matriz de homografía 3x3.
    Se asume que la primera línea contiene la matriz, donde los valores están separados por espacios y/o punto y coma.
    """
    with open(calib_file, "r") as f:
        line = f.readline().strip()
    rows = [row.strip() for row in line.split(";")]
    H = []
    for row in rows:
        vals = [float(v) for v in row.split()]
        H.append(vals)
    return np.array(H)


def project_to_world(point, H):
    # Calcular la inversa de la homografía, ya que H mapea de mundo a imagen.
    H_inv = np.linalg.inv(H)
    p = np.array([point[0], point[1], 1.0])
    p_world = H_inv.dot(p)
    p_world /= p_world[2]
    return p_world[:2]



# =============================================================================
# Función de asociación entre tracklets de dos cámaras
# =============================================================================
def associate_tracklets(tracklets_cam1, tracklets_cam2, H1, H2, min_time_gap=1, time_tol=60, spatial_tol=50):
    associations = []
    for tid1, data1 in tracklets_cam1.items():
        center1_world = project_to_world(data1['avg_center'], H1)
        for tid2, data2 in tracklets_cam2.items():
            dt = data2['start_time'] - data1['end_time']
            # Imprimir valores para depuración
            print(f"Comparando track {tid1} (cam1) y track {tid2} (cam2): dt = {dt:.2f} s")
            if dt < min_time_gap or dt > time_tol:
                continue
            center2_world = project_to_world(data2['avg_center'], H2)
            dist = np.linalg.norm(center1_world - center2_world)
            print(f"Distancia entre centros proyectados: {dist:.2f}")
            if dist < spatial_tol:
                associations.append((tid1, tid2))
    return associations


# =============================================================================
# Ejemplo de uso: Asociación entre dos cámaras (c001 y c002)
# =============================================================================
if __name__ == "__main__":
    # Definir rutas (ajusta según tu estructura)
    tracking_files = {
        "c001": "/home/toukapy/Dokumentuak/Master CV/C6/Week4/s01_c001_roi.txt",  # Archivo MOT de c001
        "c002": "/home/toukapy/Dokumentuak/Master CV/C6/Week4/s01_c002_roi.txt"  # Archivo MOT de c002
    }
    calib_files = {
        "c001": "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/train/S01/c001/calibration.txt",
        "c002": "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/train/S01/c002/calibration.txt"
    }
    timestamp_file = "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/cam_timestamp/S01.txt"  # Archivo con: "c001 0", "c002 1.640", etc.
    fps = 10  # Ajusta el FPS de tus videos

    # Cargar timestamps
    timestamps = load_timestamps(timestamp_file)
    # Cargar calibraciones
    H_cam = {}
    for cam_id, calib_path in calib_files.items():
        H_cam[cam_id] = load_calibration(calib_path)

    # Cargar tracking para cada cámara
    tracking_results = {}
    for cam_id, mot_file in tracking_files.items():
        tracking_results[cam_id] = load_tracking_results(mot_file)

    # Agregar los tracklets por cámara (convertir frames a tiempo usando FPS y timestamp de inicio)
    tracklets = {}
    for cam_id, tracking in tracking_results.items():
        start_time = timestamps.get(cam_id, 0)
        tracklets[cam_id] = aggregate_tracklets(tracking, fps, start_time)

    # Realizar la asociación entre c001 y c002 (puedes extenderlo a más cámaras)
    associations = associate_tracklets(tracklets["c001"], tracklets["c002"],
                                       H_cam["c001"], H_cam["c002"],
                                       min_time_gap=1, time_tol=120, spatial_tol=50)
    print("Asociaciones entre c001 y c002:", associations)

    # Aquí podrías asignar IDs globales basados en las asociaciones.
    # Por ejemplo, si (tid1, tid2) están asociados, se les asigna un mismo ID global.
    # Este ejemplo solo imprime las asociaciones.
