import cv2
import os
import numpy as np
import threading
from src import read_data, metrics

# -------------------------
# 1. Initial configuration
# -------------------------
video_paths = [
    "./output_videos/MOG2.avi",
    "./output_videos/BGS.avi",
    "./output_videos/LSBP.avi",
    "./output_videos/LOBSTER.avi",
    "./output_videos/MOG.avi"
]

path_annotation = "./data/ai_challenge_s03_c010-full_annotation.xml"

# load ground truth
gt_data, _ = read_data.parse_annotations_xml(path_annotation, isGT=True)
gt_dict = {}
for item in gt_data:
    frame_no = item["frame"]
    box = item["bbox"][0].tolist() if isinstance(item["bbox"], np.ndarray) else item["bbox"]
    gt_dict.setdefault(frame_no, []).append(box)

# different min area values to test
min_area_values = [500, 750, 1000, 1500, 2000]

results = {video: {} for video in video_paths}


# -------------------------
# 2. funtion to process video
# -------------------------
def process_video(video_path):
    global results

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Procesando {video_path} - FPS: {fps}, Tamaño: {frame_width}x{frame_height}")

    for min_area in min_area_values:
        print(f"  -> Evaluando min_area = {min_area}...")

        # initialize prediction and gt boxes
        all_pred_boxes = []
        all_gt_boxes = []

        frame_idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # restart video

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # end of video

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gt_boxes = gt_dict.get(frame_idx, [])

            # extract bounding boxes
            pred_boxes = metrics.extract_bounding_boxes(gray_frame, min_area=min_area)

            all_pred_boxes.append(pred_boxes)
            all_gt_boxes.append(gt_boxes)
            frame_idx += 1

        # Obtain mAP@50 for min_area
        final_map50 = metrics.compute_video_average_precision(all_pred_boxes, all_gt_boxes, iou_threshold=0.5)

        results[video_path][min_area] = final_map50

    cap.release()


# -------------------------
# 3. process 5 videos
# -------------------------
threads = []
for video in video_paths:
    thread = threading.Thread(target=process_video, args=(video,))
    threads.append(thread)
    thread.start()

# wait for the threads to end
for thread in threads:
    thread.join()

# -------------------------
# 4. print results
# -------------------------
print("\nResultados Finales (mAP@50):")
for video, min_area_results in results.items():
    print(f"\n📌 Video: {video}")
    for min_area, score in min_area_results.items():
        print(f"  - min_area {min_area}: mAP@50 = {score:.4f}")

print("\n✅ Procesamiento completado!")
