import sys, os, distutils.core
import torch, detectron2
from torchmetrics.detection import MeanAveragePrecision

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode

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
video_path = "C:/Users/User/Downloads/data/train/S03/c010/vdo.avi"
frames_dir = "./frames"  # Carpeta donde se guardarán los frames

# # Crear la carpeta si no existe
# os.makedirs(frames_dir, exist_ok=True)

# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print("Error: No se pudo abrir el video.")
#     exit()

# frame_idx = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
#     cv2.imwrite(frame_path, frame)
    
#     frame_idx += 1

# cap.release()
# print(f"Se guardaron {frame_idx} fotogramas en '{frames_dir}'")

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



def get_KITTI_dicts(img,part):
    path_annotation ="C:/Users/User/Documents/GitHub/mcv-c6-2025-team4/Week2/data/ai_challenge_s03_c010-full_annotation.xml"

    gt_data, _ = parse_annotations_xml(path_annotation, isGT=True)

    # Organize ground truth data by frame number
    gt_dict = {}
    dataset_dicts = []

    for item in gt_data:
        # Only consider moving vehicles (ignore parked cars)
        if not item.get("parked", False):
            frame_no = item["frame"]
            # Convert bounding box to list if it is a NumPy array
            box = item["bbox"][0].tolist() if isinstance(item["bbox"], np.ndarray) else item["bbox"]
            if frame_no in gt_dict:
                gt_dict[frame_no].append(box)
            else:
                gt_dict[frame_no] = [box]

    # Sort frames to ensure consistency
    sorted_frames = sorted(gt_dict.keys())
    # print(sorted_frames)
    # Compute split point (25% of total frames)
    split_index = int(len(sorted_frames) * 0.25)

    if part == "train":
        selected_frames = sorted_frames[:split_index]  # First 25%
    elif part == "val":
        selected_frames = sorted_frames[split_index:]  # Remaining frames
    else:
        raise ValueError("Invalid part argument. Choose 'train' or 'val'.")

    for frame in selected_frames:
    
        record={}
        # print(gt_dict[frame])

        record["file_name"] = os.path.join("./frames","frame_"+str(frame).zfill(6)+'.jpg')
        image = cv2.imread(record["file_name"])

        h,w,_=np.shape(image)

        record["image_id"] = frame
        record["height"] = h
        record["width"] = w


        objs = []
        for i in gt_dict[frame]:
            obj = {
                "bbox": i,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

if __name__ == '__main__':
    
    for d in ["train","val"]:
        DatasetCatalog.register("KITTI_" + d, lambda d=d: get_KITTI_dicts("C:/Users/User/Documents/MASTER/c5/data_tracking_image_2/training/",d))
        MetadataCatalog.get("KITTI_" + d).set(thing_classes=["car"])
    KITTI_metadata = MetadataCatalog.get("KITTI_train")
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)


    ## predict a few images on the pretrained model to compare with finetuned results##

    dataset_dicts = get_KITTI_dicts("C:/Users/User/Documents/MASTER/c5/data_tracking_image_2/training/","val")
    paths=[]
    for d in random.sample(dataset_dicts, 3):
        paths.append(d)
        
        print(d["file_name"])
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        print("./output/"+d["file_name"].split("/")[-1])
        cv2.imwrite("./output/"+d["file_name"].split("/")[-1],np.array(out.get_image()[:, :, ::-1]))

    ## TRAINING ##


    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("KITTI_train")
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = [] 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Two classes, car 
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=False)
    trainer.train()


    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_KITTI_dicts("C:/Users/User/Documents/MASTER/c5/data_tracking_image_2/training/","val")
    
    #run inference on same images as for pretrained model at the beggining #
    for d in paths:
        
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  
        v = Visualizer(im[:, :, ::-1],
                    metadata=KITTI_metadata,
                    scale=0.5,
                    instance_mode=ColorMode.IMAGE_BW 
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        print(os.path.join("./output/",'C6_'+d["file_name"].split('/')[-1]))
        cv2.imwrite(os.path.join("./output/",'C6_'+d["file_name"].split('/')[-1]),np.array(out.get_image()[:, :, ::-1]))


    evaluator = COCOEvaluator("KITTI_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "KITTI_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

