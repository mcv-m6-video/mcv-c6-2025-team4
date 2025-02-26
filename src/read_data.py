import numpy as np
import xmltodict

def parse_annotations_xml(xml_path, isGT=False):
    """
    Parses ground truth annotations from an XML file.

    Parameters:
    - xml_path: str
        Path to the XML file containing annotations.
    - isGT: bool, optional
        If True, includes an "already_detected" flag for ground truth objects.

    Returns:
    - gt_complete: list of dicts
        List of dictionaries containing frame-wise bounding boxes.
    - sorted_frames: list
        List of sorted frame indices with annotations.
    """

    # Open and read the XML file
    with open(xml_path, 'r') as xml_file:
        tracks = xmltodict.parse(xml_file.read())['annotations']['track']

    frames = []
    bbxs = []
    gts = []

    # Iterate over each track in the annotation file
    for track in tracks:
        id = track['@id']
        label = track['@label']
        boxes = track['box']

        # Iterate over each bounding box in the track
        for box in boxes:
            if label == 'car':
                # Check if the car is parked (ignore parked vehicles)
                parked = box['attribute']['#text'].lower() == 'true'
            else:
                parked = None

            # Store annotation data
            gt = [int(box['@frame']), int(id), label,
                  float(box['@xtl']), float(box['@ytl']),
                  float(box['@xbr']), float(box['@ybr']),
                  float(-1), parked]
            gts.append(gt)

    # Filter out parked vehicles
    for gt in gts:
        if gt[-1]:  # If parked, skip this bounding box
            continue
        frame = gt[0]
        bbx = [gt[3], gt[4], gt[5], gt[6]]  # Extract bounding box coordinates

        frames.append(frame)
        bbxs.append(bbx)

    # Sort frames and corresponding bounding boxes
    sorted_frames, sorted_bbxs = zip(*sorted(zip(frames, bbxs)))

    bbx = []
    gt_complete = []

    # Group bounding boxes by frame
    for i in range(len(sorted_bbxs)):
        if i == 0:
            bbx.append(sorted_bbxs[i])
        else:
            # Store bounding boxes for each frame
            if isGT:
                gt_complete.append(
                    {"frame": sorted_frames[i - 1], "bbox": np.array(bbx), "already_detected": [False] * len(bbx)}
                )
            else:
                gt_complete.append(
                    {"frame": sorted_frames[i - 1], "bbox": np.array(bbx)}
                )
            bbx = [sorted_bbxs[i]]

        # Ensure last frame is included
        if (i + 1) == len(sorted_bbxs):
            if isGT:
                gt_complete.append(
                    {"frame": sorted_frames[i - 1], "bbox": np.array(bbx), "already_detected": [False] * len(bbx)}
                )
            else:
                gt_complete.append(
                    {"frame": sorted_frames[i - 1], "bbox": np.array(bbx)}
                )

    return gt_complete, sorted_frames


def parse_predictions(path, isGT=False):
    """
    Parses object detection predictions from a text file.

    Parameters:
    - path: str
        Path to the text file containing detection results.
    - isGT: bool, optional
        If True, includes an "already_detected" flag for ground truth objects.

    Returns:
    - detected_info: list of dicts
        List of dictionaries containing frame-wise bounding boxes and scores.
    """

    # Open and read the prediction file
    with open(path, 'r') as predictions_file:
        predictions = predictions_file.readlines()

    frames = []
    bbxs = []
    confidence_scores = []
    preds = []

    # Parse each line in the prediction file
    for pred in predictions:
        track = pred.split(",")
        pred_list = [int(track[0]) - 1, track[1], 'car',
                     float(track[2]), float(track[3]),  # x_min, y_min
                     float(track[2]) + float(track[4]),  # x_max
                     float(track[3]) + float(track[5]),  # y_max
                     float(track[6])]  # Confidence score
        preds.append(pred_list)

    # Extract frame, bounding box, and confidence score for each prediction
    for pred in preds:
        frame = pred[0]
        bbox = [pred[3], pred[4], pred[5], pred[6]]
        confidence = pred[7]

        frames.append(frame)
        bbxs.append(bbox)
        confidence_scores.append(confidence)

    # Sort frames, bounding boxes, and confidence scores
    sorted_frames, sorted_bbxs, sorted_scores = zip(*sorted(zip(frames, bbxs, confidence_scores)))

    bbxs_complete = []
    score_complete = []
    detected_info = []

    # Group bounding boxes and scores by frame
    for i in range(len(sorted_bbxs)):
        if i == 0:
            bbxs_complete.append(sorted_bbxs[i])
            score_complete.append(sorted_scores[i])
        else:
            if isGT:
                detected_info.append(
                    {"frame": sorted_frames[i - 1],
                     "bbox": np.array(bbxs_complete),
                     "score": np.array(score_complete),
                     "already_detected": [False] * len(bbxs_complete)}
                )
            else:
                detected_info.append(
                    {"frame": sorted_frames[i - 1],
                     "bbox": np.array(bbxs_complete),
                     "score": np.array(score_complete)}
                )

            # Reset lists for the next frame
            bbxs_complete = []
            score_complete = []
            bbxs_complete.append(sorted_bbxs[i])
            score_complete.append(sorted_scores[i])

        # Ensure last frame is included
        if (i + 1) == len(sorted_bbxs):
            if isGT:
                detected_info.append(
                    {"frame": sorted_frames[i - 1],
                     "bbox": np.array(bbxs_complete),
                     "score": np.array(score_complete),
                     "already_detected": [False] * len(bbxs_complete)}
                )
            else:
                detected_info.append(
                    {"frame": sorted_frames[i - 1],
                     "bbox": np.array(bbxs_complete),
                     "score": np.array(score_complete)}
                )

    return detected_info
