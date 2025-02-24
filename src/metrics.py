import cv2
import numpy as np


def remove_noise(mask, kernel_size=3):
    # Create a square kernel. Adjust kernel_size based on image resolution.
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Morphological opening to remove small noise (isolated pixels)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Morphological closing to fill small holes in the detected objects
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed


def boxes_overlap(boxA, boxB):
    """
    Returns True if two boxes [x_min, y_min, x_max, y_max] overlap (non-zero intersection).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return (xA < xB) and (yA < yB)


def merge_overlapping_boxes(boxes):
    """
    Merge overlapping boxes into larger boxes.

    This function uses a simple iterative approach:
      - For each box, it checks against the others.
      - If two boxes overlap, they are merged (by taking the union of their coordinates).
      - The process repeats until no further merging is possible.
    """
    merged = True
    while merged:
        merged = False
        new_boxes = []
        skip = [False] * len(boxes)
        for i in range(len(boxes)):
            if skip[i]:
                continue
            current_box = boxes[i]
            for j in range(i + 1, len(boxes)):
                if skip[j]:
                    continue
                if boxes_overlap(current_box, boxes[j]):
                    # Merge the two boxes into a union rectangle.
                    current_box = [
                        min(current_box[0], boxes[j][0]),
                        min(current_box[1], boxes[j][1]),
                        max(current_box[2], boxes[j][2]),
                        max(current_box[3], boxes[j][3])
                    ]
                    skip[j] = True
                    merged = True
            new_boxes.append(current_box)
        boxes = new_boxes
    return boxes


def extract_bounding_boxes(mask, min_area=800):
    """
    Given a binary mask (uint8 image where nonzero pixels represent detected foreground),
    find contours and return bounding boxes [x_min, y_min, x_max, y_max] for each contour.
    Optionally ignores small contours based on min_area, and then merges overlapping boxes.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue  # filter out small regions
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x, y, x + w, y + h])

    if not boxes:
        return boxes

    # Merge overlapping bounding boxes to avoid very small or fragmented boxes.
    merged_boxes = merge_overlapping_boxes(boxes)
    return merged_boxes

def compute_iou(boxA, boxB):
    """
    Computes Intersection over Union between two boxes.
    Boxes are [x_min, y_min, x_max, y_max].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def compute_frame_average_precision(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Calcula el Average Precision (AP) para un frame dado.
    Se utiliza un matching greedy: para cada caja predicha se busca la caja GT no asignada
    con IoU >= iou_threshold. Si se encuentra, se cuenta como TP.
    Luego se define:
        AP = (número de TP) / (número total de cajas GT)
    Si no hay cajas GT, se define AP = 1.0 si no hay detecciones o 0.0 si existen detecciones.
    """
    matched = [False] * len(gt_boxes)
    tp = 0
    for pred in pred_boxes:
        for i, gt in enumerate(gt_boxes):
            if not matched[i] and compute_iou(pred, gt) >= iou_threshold:
                matched[i] = True
                tp += 1
                break
    if len(gt_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0
    return tp / len(gt_boxes)


def ensure_box_list(boxes):
    """
    Ensures that boxes is a list of boxes.
    If boxes is a numpy array with ndim==1 (i.e. a single box), wrap it in a list.
    If it's a numpy array with ndim==2, convert it to a list.
    """
    import numpy as np
    if isinstance(boxes, np.ndarray):
        if boxes.ndim == 1:
            return [boxes.tolist()]
        elif boxes.ndim == 2:
            return boxes.tolist()
    elif isinstance(boxes, list):
        # Check if the first element is not a list (i.e. single box)
        if boxes and not isinstance(boxes[0], (list, tuple)):
            return [boxes]
    return boxes


def evaluate_detections(pred_boxes, gt_boxes, iou_threshold=0.5):
    # Ensure gt_boxes is a list of boxes
    gt_boxes = ensure_box_list(gt_boxes)

    matched_gt = set()
    iou_list = []
    tp = 0
    fp = 0
    for pb in pred_boxes:
        best_iou = 0
        best_gt = None
        for i, gb in enumerate(gt_boxes):
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_gt = i
        if best_iou >= iou_threshold and best_gt not in matched_gt:
            tp += 1
            matched_gt.add(best_gt)
            iou_list.append(best_iou)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return precision, recall, iou_list, tp, fp, fn

def generate_gt_mask(frame_shape, gt_boxes):
    """
    Given a frame shape and a list of ground truth bounding boxes,
    create a binary mask (0: background, 255: object).
    """
    gt_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    for box in gt_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(gt_mask, (x1, y1), (x2, y2), 255, -1)
    return gt_mask

def compute_pixel_metrics(pred_mask, gt_mask):
    """
    Compute pixel-level true positive rate (TPR) and false positive rate (FPR)
    between a predicted mask and a ground truth mask.
    Both masks are assumed binary (nonzero = object).
    """
    pred_flat = (pred_mask > 0).astype(np.uint8).flatten()
    gt_flat = (gt_mask > 0).astype(np.uint8).flatten()
    TP = np.sum((pred_flat == 1) & (gt_flat == 1))
    TN = np.sum((pred_flat == 0) & (gt_flat == 0))
    FP = np.sum((pred_flat == 1) & (gt_flat == 0))
    FN = np.sum((pred_flat == 0) & (gt_flat == 1))
    TPR = TP / (TP + FN + 1e-6)
    FPR = FP / (FP + TN + 1e-6)
    return TPR, FPR
