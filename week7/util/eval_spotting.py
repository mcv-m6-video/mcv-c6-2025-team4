"""
File containing main evaluation functions
"""

#Standard imports
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
import json
import os
import sys
#Local imports
from dataset.frame import FPS_SN


sys.path.append('/export/home/c5mcv02/CVMasterActionSpotting/util/')
from ActionSpotting import average_mAP

#Constants
INFERENCE_BATCH_SIZE = 4


def evaluate(model, dataset, batch_size=INFERENCE_BATCH_SIZE, nms_window=5):
    import os
    from tqdm import tqdm
    import numpy as np
    import torch
    import json
    from dataset.frame import FPS_SN
    from ActionSpotting import average_mAP

    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[os.path.basename(video)] = (
            np.zeros((video_len, len(dataset._class_dict)), np.float32),  # [T, C]
            np.zeros(video_len, np.int32)  # support mask
        )

    loader = DataLoader(dataset, shuffle=False, num_workers=batch_size * 2, pin_memory=True, batch_size=batch_size)

    for clip in tqdm(loader, desc="Evaluating"):
        # Skip None entries or invalid frame batches
        if clip is None or (isinstance(clip, dict) and isinstance(clip['frame'], int) and clip['frame'] == -1):
            continue

        # Batched [B, T, C, H, W]
        if isinstance(clip['frame'], list) or isinstance(clip['frame'], tuple):
            frames = [torch.stack(f) if isinstance(f, list) else f for f in clip['frame']]
            clip_frames = torch.stack(frames)
        else:
            clip_frames = clip['frame']

        if isinstance(clip_frames, int) or clip_frames.ndim < 5:
            continue  # Skip broken batch

        try:
            batch_pred_scores = model.predict(clip_frames)  # [B, T, C]
        except Exception as e:
            print(f"[WARN] Model prediction failed for clip: {e}")
            continue

        for i in range(clip_frames.shape[0]):
            video = os.path.basename(clip['video'][i])
            start = clip['start'][i].item()

            if video not in pred_dict:
                print(f"[WARN] Video {video} not found in pred_dict keys.")
                continue

            scores, support = pred_dict[video]
            pred_scores = batch_pred_scores[i]  # [T, C]

            if start < 0:
                pred_scores = pred_scores[-start:, :]
                start = 0
            end = start + pred_scores.shape[0]
            if end >= scores.shape[0]:
                end = scores.shape[0]
                pred_scores = pred_scores[:end - start, :]

            scores[start:end, :] += pred_scores[:, 1:]  # remove background class
            support[start:end] += (pred_scores.sum(axis=1) != 0).astype(np.int32)

    detections_numpy = []
    for video, video_len, _ in dataset.videos:
        video = os.path.basename(video)
        scores, support = pred_dict[video]
        support[support == 0] = 1
        scores = scores / support[:, np.newaxis]
        pred = apply_NMS(scores, nms_window, 0.05)
        detections_numpy.append(pred)

    targets_numpy = []
    closests_numpy = []

    for video, video_len, _ in dataset.videos:
        video_basename = os.path.basename(video)
        targets = np.zeros((video_len, len(dataset._class_dict)), np.float32)
        label_path = os.path.join(dataset._labels_dir, "england_efl/2019-2020", video_basename, 'Labels-ball.json')

        if not os.path.exists(label_path):
            print(f"[WARN] Missing label file for video {video_basename}")
            targets_numpy.append(targets)
            closests_numpy.append(np.full_like(targets, -1))
            continue

        labels = json.load(open(label_path))
        for ann in labels["annotations"]:
            event = dataset._class_dict[ann["label"]]
            frame = int(FPS_SN / dataset._stride * (int(ann["position"]) / 1000))
            frame = min(frame, video_len - 1)
            targets[frame, event - 1] = 1

        targets_numpy.append(targets)

        closest = np.full_like(targets, -1)
        for c in range(targets.shape[1]):
            idxs = np.where(targets[:, c] == 1)[0].tolist()
            if len(idxs) == 0:
                continue
            idxs = [-idxs[0]] + idxs + [2 * targets.shape[0]]
            for i in range(1, len(idxs) - 1):
                start = max(0, (idxs[i - 1] + idxs[i]) // 2)
                stop = min(closest.shape[0], (idxs[i] + idxs[i + 1]) // 2)
                closest[start:stop, c] = targets[idxs[i], c]
        closests_numpy.append(closest)

    mAP, AP_per_class, *_ = average_mAP(
        targets_numpy, detections_numpy, closests_numpy,
        FPS_SN / dataset._stride, deltas=np.array([1])
    )

    return mAP, AP_per_class



def apply_NMS(predictions, window, thresh=0.0):

    nf, nc = predictions.shape
    for i in range(nc):
        aux = predictions[:,i]
        aux2 = np.zeros(nf) -1
        while(np.max(aux) >= thresh):
            # Get the max remaining index and value
            max_value = np.max(aux)
            max_index = np.argmax(aux)
            # detections_NMS[max_index,i] = max_value

            nms_from = int(np.maximum(-(window/2)+max_index,0))
            nms_to = int(np.minimum(max_index+int(window/2), len(aux)))

            aux[nms_from:nms_to] = -1
            aux2[max_index] = max_value
        predictions[:,i] = aux2

    return predictions
