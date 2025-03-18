import os
import numpy as np
import cv2
from itertools import combinations

# =============================================================================
# Funciones de carga
# =============================================================================

def load_tracking_results(mot_file):
    """ Carga el archivo MOT y devuelve un diccionario con detecciones por frame. """
    tracking = {}
    with open(mot_file, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            frame = int(parts[0].strip())
            tid = int(parts[1].strip())
            x, y, w, h = map(float, parts[2:6])
            if frame not in tracking:
                tracking[frame] = []
            tracking[frame].append({'track_id': tid, 'bbox': [x, y, w, h]})
    return tracking

def aggregate_tracklets(tracking, fps, start_time):
    """ Agrupa detecciones por track_id y calcula información relevante. """
    tracklets = {}
    for frame, detections in tracking.items():
        for det in detections:
            tid = det['track_id']
            bbox = det['bbox']
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

def project_to_local(point, H, ref_gps):
    """ Proyecta un punto de imagen a coordenadas locales (GPS - ref_gps). """
    H_inv = np.linalg.inv(H)
    p = np.array([point[0], point[1], 1.0])
    gps = H_inv.dot(p)
    gps /= gps[2]
    local = gps[:2] - np.array(ref_gps)
    return local

# =============================================================================
# Clase Union-Find para fusionar asociaciones
# =============================================================================
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.cameras = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.cameras[x] = set([x.split('_')[0]])
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        x_root, y_root = self.find(x), self.find(y)
        if x_root == y_root:
            return
        self.parent[y_root] = x_root
        self.cameras[x_root].update(self.cameras[y_root])

    def get_groups(self):
        groups = {}
        for item in self.parent:
            root = self.find(item)
            if root not in groups:
                groups[root] = []
            groups[root].append(item)
        return groups

    def get_group_cameras(self, group_id):
        return self.cameras[self.find(group_id)]

# =============================================================================
# Función de asociación entre tracklets de dos cámaras
# =============================================================================
def associate_tracklets(tracklets_cam1, tracklets_cam2, H1, H2, ref_gps, min_time_gap=1, time_tol=60, spatial_tol=50):
    """ Asocia tracklets de dos cámaras considerando tiempo y espacio. """
    associations = []
    for tid1, data1 in tracklets_cam1.items():
        center1_local = project_to_local(data1['avg_center'], H1, ref_gps)
        for tid2, data2 in tracklets_cam2.items():
            dt = data2['start_time'] - data1['end_time']
            if dt < min_time_gap or dt > time_tol:
                continue
            center2_local = project_to_local(data2['avg_center'], H2, ref_gps)
            if np.linalg.norm(center1_local - center2_local) < spatial_tol:
                associations.append((tid1, tid2))
    return associations

# =============================================================================
# Proceso Global Multi-Cámara con Corrección de ID
# =============================================================================
if __name__ == "__main__":
    cam_ids = ["c001", "c002", "c003", "c004", "c005"]
    tracking_files = {cam: f"/home/toukapy/Dokumentuak/Master CV/C6/Week4/{cam}.txt" for cam in cam_ids}
    calib_files = {cam: f"/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/train/S01/{cam}/calibration.txt" for cam in cam_ids}
    timestamp_file = "/home/toukapy/Dokumentuak/Master CV/C6/data/aic19-track1-mtmc-train/cam_timestamp/S01.txt"
    fps = 10
    ref_gps = [42.525678, -90.723601]

    timestamps = load_timestamps(timestamp_file)
    H_cam = {cam: load_calibration(calib_files[cam]) for cam in cam_ids}

    tracklets = {}
    for cam in cam_ids:
        tracking = load_tracking_results(tracking_files[cam])
        tracklets[cam] = aggregate_tracklets(tracking, fps, timestamps[cam])

    all_associations = []
    for cam1, cam2 in combinations(cam_ids, 2):
        assoc = associate_tracklets(tracklets[cam1], tracklets[cam2], H_cam[cam1], H_cam[cam2], ref_gps)
        for tid1, tid2 in assoc:
            all_associations.append((f"{cam1}_{tid1}", f"{cam2}_{tid2}"))

    uf = UnionFind()
    for a, b in all_associations:
        uf.union(a, b)
    groups = uf.get_groups()

    # 🔹 CORRECT ID OFFSET BASED ON TIMESTAMP ORDER
    cam_start_times = {cam: timestamps[cam] for cam in cam_ids}
    sorted_cameras = sorted(cam_start_times.keys(), key=lambda cam: cam_start_times[cam])

    id_offset = {}
    offset_value = 1
    for cam in sorted_cameras:
        id_offset[cam] = offset_value
        max_local_id = max(tracklets[cam].keys()) if tracklets[cam] else 0
        offset_value += max_local_id

    global_id_mapping = {}
    for group_id, items in groups.items():
        group_cams = uf.get_group_cameras(group_id)
        if len(group_cams) < 2:
            continue
        for item in items:
            cam_id, local_tid = item.split("_")
            local_tid = int(local_tid)
            global_id_mapping[item] = id_offset[cam_id] + local_tid - 1

    out_dir = "Global_Tracking"
    os.makedirs(out_dir, exist_ok=True)
    for cam in cam_ids:
        input_file, output_file = tracking_files[cam], os.path.join(out_dir, f"{cam}_global.txt")
        with open(input_file, "r") as fin, open(output_file, "w") as fout:
            for line in fin:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                key = f"{cam}_{parts[1].strip()}"
                if key in global_id_mapping:
                    fout.write(f"{parts[0]}, {global_id_mapping[key]}, {', '.join(parts[2:])}\n")


