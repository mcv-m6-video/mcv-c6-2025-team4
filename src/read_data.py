import numpy as np
import xmltodict

def parse_annotations_xml(xml_path, isGT = False):

    with open(xml_path, 'r') as xml_file:
        tracks = xmltodict.parse(xml_file.read())['annotations']['track']

    frames = []
    bbxs = []
    gts = []

    for track in tracks:
        id = track['@id']
        label = track['@label']
        boxes = track['box']

        for box in boxes:
            if label == 'car':
                parked = box['attribute']['#text'].lower() == 'true'
            else:
                parked = None

            gt = [int(box['@frame']), int(id), label,
                  float(box['@xtl']), float(box['@ytl']),
                  float(box['@xbr']), float(box['@ybr']),
                  float(-1), parked]
            gts.append(gt)

    for gt in gts:
        if gt[-1]:
            continue
        frame = gt[0]
        bbx = [gt[3], gt[4], gt[5], gt[6]]

        frames.append(frame)
        bbxs.append(bbx)


    # Sort frames
    sorted_frames, sorted_bbxs = zip(*sorted(zip(frames, bbxs)))

    bbx = []
    gt_complete = []

    for i in range(len(sorted_bbxs)):
        if i == 0:
            bbx.append(sorted_bbxs[i])
        else:
            if isGT:
                gt_complete.append(
                    {"frame": sorted_frames[i - 1], "bbox": np.array(bbx), "already_detected": [False] * len(bbx)}
                )
            else:
                gt_complete.append(
                    {"frame": sorted_frames[i - 1], "bbox": np.array(bbx)}
                )
            bbx = [sorted_bbxs[i]]

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

def parse_predictions(path, isGT = False):

    with open(path, 'r') as predictions_file:
        predictions = predictions_file.readlines()

    frames = []
    bbxs = []
    confidence_scores = []
    preds = []

    for pred in predictions:
        track = pred.split(",")
        pred_list = [int(track[0]) - 1, track[1], 'car', float(track[2]), float(track[3]),
                     float(track[2]) + float(track[4]),
                     float(track[3]) + float(track[5]),
                     float(track[6])]
        preds.append(pred_list)

    for pred in preds:
        # Grab the frames, bounding box and confidence for each prediction
        frame = pred[0]
        bbox = [pred[3], pred[4], pred[5], pred[6]]
        confidence = pred[7]

        # Append each variable
        frames.append(frame)
        bbxs.append(bbox)
        confidence_scores.append(confidence)

    # Sort frames and corresponding bounding boxes and confidence scores
    sorted_frames, sorted_bbxs, sorted_scores = zip(*sorted(zip(frames, bbxs, confidence_scores)))

    bbxs_complete = []
    score_complete = []
    detected_info = []

    for i in range(len(sorted_bbxs)):
        # If first frame --> Add bounding box
        if i == 0:
            bbxs_complete.append(sorted_bbxs[i])
            score_complete.append(sorted_scores[i])
        else:
            if isGT:
                # Parameter to check if box is already detected for the frame
                detected_info.append(
                    {"frame": sorted_frames[i - 1], "bbox": np.array(bbxs_complete), "score": np.array(score_complete), "already_detected": [False] * len(bbxs_complete)}
                )
            else:
                detected_info.append(
                    {"frame": sorted_frames[i - 1], "bbox": np.array(bbxs_complete), "score": np.array(score_complete)}
                )

            bbxs_complete = []
            score_complete = []
            bbxs_complete.append(sorted_bbxs[i])
            score_complete.append(sorted_scores[i])

        if (i + 1) == len(sorted_bbxs):
            if isGT:
                detected_info.append(
                    {"frame": sorted_frames[i - 1], "bbox": np.array(bbxs_complete), "score": np.array(score_complete), "already_detected": [False] * len(bbxs_complete)}
                )
            else:
                detected_info.append(
                    {"frame": sorted_frames[i - 1], "bbox": np.array(bbxs_complete), "score": np.array(score_complete)}
                )
    return detected_info