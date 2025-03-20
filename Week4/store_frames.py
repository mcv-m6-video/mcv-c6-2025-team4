
# # Load data
# def load_tracking_data_gt(file_path):
#     data = {}
#     with open(file_path, "r") as f:
#         for line in f:
#             frame, track_id, x, y, w, h, _, _, _,_ = map(int, line.strip().split(","))
#             if frame not in data:
#                 data[frame] = []
#             data[frame].append([track_id, x, y, w, h])
#     return data

import cv2
import os

# # Ruta del video

path_seq="E:/aic19-track1-mtmc-train/train/S01/"
vid_folder=os.listdir(path_seq)
print(vid_folder)
frames_dir = "/frames"  # Carpeta donde se guardarán los frames

for vid in vid_folder:
    # Crear la carpeta si no existe
    os.makedirs(path_seq+vid+frames_dir, exist_ok=True)
    video_path = path_seq+vid+"/vdo.avi"


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        exit()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = os.path.join(path_seq+vid+frames_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        frame_idx += 1

    cap.release()
    print(f"Se guardaron {frame_idx} fotogramas en '{path_seq+vid+frames_dir}'")