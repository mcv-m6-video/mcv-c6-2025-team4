import os
import numpy as np
from scipy.optimize import linear_sum_assignment

all_hota=[]
all_idf=[]
# for vid in ['c011','c012','c013','c010','c014','c015']:
# for vid in ['c016','c017','c018','c019','c020','c021','c022','c023','c024','c025','c026','c027','c028']:
seq='S03/'
for vid in ['c010','c011','c012','c013','c014', 'c015']:
    # Paths to input files
    PREDICTIONS_FILE = "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/Global_Tracking_Miren_S03/"+vid+"_global.txt"  # Replace with actual path
    GROUND_TRUTH_FILE = "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/data/aic19-track1-mtmc-train/train/"+seq+vid+"/gt/gt.txt"
    OUTPUT_DIR = "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/data/aic19-track1-mtmc-train/train/"+seq+vid+"/pred/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "predictions.txt")
    from trackeval.metrics.hota import HOTA
    from trackeval.metrics.identity import Identity
    # Load data
    def load_tracking_data_gt(file_path):
        data = {}
        with open(file_path, "r") as f:
            for line in f:
                frame, track_id, x, y, w, h, _, _, _,_ = map(int, line.strip().split(","))
                if frame not in data:
                    data[frame] = []
                data[frame].append([track_id, x, y, w, h])
        return data

    def load_tracking_data_pred(file_path):
        data = {}
        with open(file_path, "r") as f:
            for line in f:
                frame, track_id, x, y, w, h, _, _, _= map(int, line.strip().split(","))
                frame=frame+1
                if frame not in data:
                    data[frame] = []
                data[frame].append([track_id, x, y, w, h])
        return data

    def load_tracking_data(file_path):
        data = {}
        with open(file_path, "r") as f:
            for line in f:
                frame, track_id, x, y, w, h,_,_,_ = map(int, line.strip().split(","))
                if frame not in data:
                    data[frame] = []
                data[frame].append([track_id, x, y, w, h])
        return data


    gt_data = load_tracking_data_gt(GROUND_TRUTH_FILE)
    pred_data = load_tracking_data_pred(PREDICTIONS_FILE)

    # Frame alignment
    aligned_predictions = {}
    for frame in gt_data.keys():
        if frame in pred_data:
            aligned_predictions[frame] = pred_data[frame]

    def iou(bb1, bb2):
        x1, y1, w1, h1 = bb1
        x2, y2, w2, h2 = bb2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter_area = max(0, xb - xa) * max(0, yb - ya)
        box1_area = w1 * h1
        box2_area = w2 * h2
        return inter_area / float(box1_area + box2_area - inter_area)


    def match_tracks(gt_tracks, pred_tracks):
        cost_matrix = np.zeros((len(gt_tracks), len(pred_tracks)))
        for i, gt in enumerate(gt_tracks):
            for j, pred in enumerate(pred_tracks):
                cost_matrix[i, j] = 1 - iou(gt[1:], pred[1:])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return {pred_tracks[j][0]: gt_tracks[i][0] for i, j in zip(row_ind, col_ind)}

    final_data = []
    for frame, gt_tracks in gt_data.items():
        if frame in aligned_predictions:
            pred_tracks = aligned_predictions[frame]
            id_mapping = match_tracks(gt_tracks, pred_tracks)
            for pred in pred_tracks:
                track_id = id_mapping.get(pred[0], pred[0])  # Default to the same ID if no match
                final_data.append([frame, track_id, *pred[1:], 1, 1, 1])  # Add confidence, class, visibility

    # Save formatted results
    with open(OUTPUT_FILE, "w") as f:
        for row in final_data:
            f.write(",".join(map(str, row)) + "\n")

    # print(f"Processed tracking results saved to {OUTPUT_FILE}")

    def prepare_data_for_eval(gt_file, pred_file):
        gt_data = load_tracking_data_gt(gt_file)
        pred_data = load_tracking_data(pred_file)
        
        frames = sorted(set(gt_data.keys()) | set(pred_data.keys()))
        all_gt_ids = set()
        all_pred_ids = set()
        for frame in frames:
            all_gt_ids.update([t[0] for t in gt_data.get(frame, [])])
            all_pred_ids.update([t[0] for t in pred_data.get(frame, [])])

        
        eval_data = {
            'num_gt_dets': sum(len(gt_data.get(frame, [])) for frame in frames),
            'num_tracker_dets': sum(len(pred_data.get(frame, [])) for frame in frames),
            'num_gt_ids': len(all_gt_ids),
            'num_tracker_ids': len(all_pred_ids),
            'gt_ids': [],
            'tracker_ids': [],
            'similarity_scores': []
        }
        
        id_map_gt = {id_val: idx for idx, id_val in enumerate(sorted(all_gt_ids))}
        id_map_pred = {id_val: idx for idx, id_val in enumerate(sorted(all_pred_ids))}
        
        for frame in frames:
            # print(frame)
            gt_tracks = gt_data.get(frame, [])
            pred_tracks = pred_data.get(frame, [])
            gt_ids = np.array([id_map_gt[t[0]] for t in gt_tracks], dtype=int)
            pred_ids = np.array([id_map_pred[t[0]] for t in pred_tracks], dtype=int)
            
            similarity = np.zeros((len(gt_tracks), len(pred_tracks)))
            for i, gt in enumerate(gt_tracks):
                for j, pred in enumerate(pred_tracks):
                    similarity[i, j] = iou(gt[1:], pred[1:])
            
            eval_data['gt_ids'].append(gt_ids)
            eval_data['tracker_ids'].append(pred_ids)
            eval_data['similarity_scores'].append(similarity)
        
        return eval_data

    data = prepare_data_for_eval(GROUND_TRUTH_FILE, OUTPUT_FILE)

    hota_metric = HOTA()
    id_metric = Identity()
    hota_results = hota_metric.eval_sequence(data)
    id_results = id_metric.eval_sequence(data)
    all_hota.append(hota_results['HOTA(0)'])
    all_idf.append(id_results['IDF1'])
    print("Sequence S01 "+vid)
    print("IDF1/ HOTA results:")
    print("{0:.2f}".format(id_results['IDF1']*100),'/',"{0:.2f}".format(hota_results['HOTA(0)']*100))

print('average')
print(np.average(all_idf)*100,' ',np.average(all_hota)*100)

