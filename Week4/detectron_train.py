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


def get_KITTI_dicts(part):
    if part == "train":
        path_seq="E:/aic19-track1-mtmc-train/train/S01/"
        vid_folder=os.listdir(path_seq)
        dataset_dicts = []

        for vid in vid_folder:
            gt_file=path_seq+vid+"/gt/gt.txt"
            gt_dict = {}

            # Read the ground truth file
            with open(gt_file, "r") as f:
                for line in f:
                    values = list(map(int, line.strip().split(',')))
                    frame, ID, left, top, width, height, *_ = values
                    
                    # Convert [left, top, width, height] -> [x_min, y_min, x_max, y_max]
                    bbox = [left, top, left + width, top + height]
                    
                    # Store in dictionary grouped by frame
                    if frame not in gt_dict:
                        gt_dict[frame] = []
                    gt_dict[frame].append(bbox)

            # Process each frame
            for frame in sorted(gt_dict.keys()):
                record = {}

                # Image path
                record["file_name"] = path_seq+vid+"/frames/"+ f"frame_{str(frame-1).zfill(6)}.jpg"
                # Read image to get dimensions
                image = cv2.imread(record["file_name"])
                if image is None:
                    print(f"Warning: Image {record['file_name']} not found.")
                    continue
                
                h, w, _ = np.shape(image)

                record["image_id"] = frame
                record["height"] = h
                record["width"] = w

                # Annotations
                objs = []
                for bbox in gt_dict[frame]:
                    obj = {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": 0,
                    }
                    objs.append(obj)
                record["annotations"] = objs
                dataset_dicts.append(record)

        path_seq="E:/aic19-track1-mtmc-train/train/S04/"
        vid_folder=os.listdir(path_seq)

        for vid in vid_folder:
            gt_file=path_seq+vid+"/gt/gt.txt"
            gt_dict = {}

            # Read the ground truth file
            with open(gt_file, "r") as f:
                for line in f:
                    values = list(map(int, line.strip().split(',')))
                    frame, ID, left, top, width, height, *_ = values
                    
                    # Convert [left, top, width, height] -> [x_min, y_min, x_max, y_max]
                    bbox = [left, top, left + width, top + height]
                    
                    # Store in dictionary grouped by frame
                    if frame not in gt_dict:
                        gt_dict[frame] = []
                    gt_dict[frame].append(bbox)

            # Process each frame
            for frame in sorted(gt_dict.keys()):
                record = {}

                # Image path
                record["file_name"] = path_seq+vid+"/frames/"+ f"frame_{str(frame-1).zfill(6)}.jpg"
                # Read image to get dimensions
                image = cv2.imread(record["file_name"])
                if image is None:
                    print(f"Warning: Image {record['file_name']} not found.")
                    continue
                
                h, w, _ = np.shape(image)

                record["image_id"] = frame
                record["height"] = h
                record["width"] = w

                # Annotations
                objs = []
                for bbox in gt_dict[frame]:
                    obj = {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": 0,
                    }
                    objs.append(obj)
                record["annotations"] = objs
                dataset_dicts.append(record)

    elif part == "val":

        path_seq="E:/aic19-track1-mtmc-train/train/S03/"
        vid_folder=os.listdir(path_seq)
        dataset_dicts = []

        for vid in vid_folder:
            gt_file=path_seq+vid+"/gt/gt.txt"
            gt_dict = {}

            # Read the ground truth file
            with open(gt_file, "r") as f:
                for line in f:
                    values = list(map(int, line.strip().split(',')))
                    frame, ID, left, top, width, height, *_ = values
                    
                    # Convert [left, top, width, height] -> [x_min, y_min, x_max, y_max]
                    bbox = [left, top, left + width, top + height]
                    
                    # Store in dictionary grouped by frame
                    if frame not in gt_dict:
                        gt_dict[frame] = []
                    gt_dict[frame].append(bbox)

            # Process each frame
            for frame in sorted(gt_dict.keys()):
                record = {}

                # Image path
                record["file_name"] = path_seq+vid+"/frames/"+ f"frame_{str(frame-1).zfill(6)}.jpg"
                # Read image to get dimensions
                image = cv2.imread(record["file_name"])
                if image is None:
                    print(f"Warning: Image {record['file_name']} not found.")
                    continue
                
                h, w, _ = np.shape(image)

                record["image_id"] = frame
                record["height"] = h
                record["width"] = w

                # Annotations
                objs = []
                for bbox in gt_dict[frame]:
                    obj = {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": 0,
                    }
                    objs.append(obj)
                record["annotations"] = objs
                dataset_dicts.append(record)

    else:
        raise ValueError("Invalid part argument. Choose 'train' or 'val'.")

    return dataset_dicts

if __name__ == '__main__':
    
    for d in ["train","val"]:
        DatasetCatalog.register("KITTI_" + d, lambda d=d: get_KITTI_dicts(d))
        MetadataCatalog.get("KITTI_" + d).set(thing_classes=["car"])
    KITTI_metadata = MetadataCatalog.get("KITTI_train")
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)


    ## predict a few images on the pretrained model to compare with finetuned results##

    dataset_dicts = get_KITTI_dicts("val")
    paths=[]
    for d in random.sample(dataset_dicts, 3):
        paths.append(d)
        
        print(d["file_name"])

        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        print("./output/"+d["file_name"].split("/")[-3]+'_'+d["file_name"].split("/")[-1])
        cv2.imwrite("./output/"+d["file_name"].split("/")[-3]+'_'+d["file_name"].split("/")[-1],np.array(out.get_image()[:, :, ::-1]))

    ## TRAINING ##


    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("KITTI_train")
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025  
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = [6000,8000]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Two classes, car 
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=False)
    trainer.train()


    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_KITTI_dicts("val")
    
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
        print("./output/"+"FINETUNED_"+d["file_name"].split("/")[-3]+'_'+d["file_name"].split("/")[-1])
        cv2.imwrite("./output/"+"FINETUNED_"+d["file_name"].split("/")[-3]+'_'+d["file_name"].split("/")[-1],np.array(out.get_image()[:, :, ::-1]))


    evaluator = COCOEvaluator("KITTI_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "KITTI_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

