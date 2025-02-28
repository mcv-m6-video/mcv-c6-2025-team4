import os
import cv2
import numpy as np
from tqdm import tqdm
import imageio
from PIL import Image
import torch

from torchvision.io.image import decode_image
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights,retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights,fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights,ssd300_vgg16, SSD300_VGG16_Weights,ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from torchmetrics.detection import MeanAveragePrecision

from src import load_data, metrics, read_data
from src.load_data import load_video_frame, load_frames_list

# torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
print(device)

# Paths to dataset and annotations
path = "./data/AICity_data/train/S03/c010"
output_dir='./output_videos'
path_annotation = "./data/ai_challenge_s03_c010-full_annotation.xml"
path_detection = [
    "./data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt",
    "./data/AICity_data/train/S03/c010/det/det_ssd512.txt",
    "./data/AICity_data/train/S03/c010/det/det_yolo3.txt"
]

video_path = os.path.join(path, "vdo.avi")
# Get the total number of frames in the video
total_frames = load_data.get_total_frames(video_path)

# Use 25% of the video frames for training the background model
training_end = int(total_frames * 0.25)
# training_frames = load_data.load_frames_list(video_path, start=0, end=training_end)
# test_frames = load_data.load_frames_list(video_path, start=training_end, end=total_frames)


test_frames = load_data.load_frames_list(video_path, start=0, end=total_frames)

# Load ground truth annotations from XML file
gt_data = read_data.parse_annotations_xml(path_annotation, isGT=True)

# # Organize ground truth data by frame number
# gt_dict = {}
# for item in gt_data:
#     frame_no = item["frame"]
#     # Convert bounding box to list if it is a NumPy array
#     box = item["bbox"][0].tolist() if isinstance(item["bbox"], np.ndarray) else item["bbox"]
#     if frame_no in gt_dict:
#         gt_dict[frame_no].append(box)
#     else:
#         gt_dict[frame_no] = [box]

models= [
         "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights",
         "FasterRCNN_MobileNet_V3_Large_FPN_Weights",
        "FasterRCNN_ResNet50_FPN_V2_Weights",
        "FasterRCNN_ResNet50_FPN_Weights",
        "RetinaNet_ResNet50_FPN_V2_Weights",
        "SSD300_VGG16_Weights",
        "SSDLite320_MobileNet_V3_Large_Weights",
        "FCOS_ResNet50_FPN_Weights",
        "RetinaNet_ResNet50_FPN_Weights"]

for m in models:
    print(m)
    if m=="FCOS_ResNet50_FPN_Weights":
        
        weights = FCOS_ResNet50_FPN_Weights.COCO_V1
        model = fcos_resnet50_fpn(weights=weights, box_score_thresh=0.9)

    elif m=="FasterRCNN_MobileNet_V3_Large_320_FPN_Weights":
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, box_score_thresh=0.9)

    elif m=="FasterRCNN_MobileNet_V3_Large_FPN_Weights":
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, box_score_thresh=0.9)

    elif m=="FasterRCNN_ResNet50_FPN_V2_Weights":
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)

    elif m=="FasterRCNN_ResNet50_FPN_Weights":
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=0.9)

    elif m=="RetinaNet_ResNet50_FPN_V2_Weights":
        weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        model = retinanet_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)

    elif m=="RetinaNet_ResNet50_FPN_Weights":
        weights = RetinaNet_ResNet50_FPN_Weights.COCO_V1
        model = retinanet_resnet50_fpn(weights=weights, box_score_thresh=0.9)

    elif m=="SSD300_VGG16_Weights":
        weights = SSD300_VGG16_Weights.COCO_V1
        model = ssd300_vgg16(weights=weights, box_score_thresh=0.9)

    elif m=="SSDLite320_MobileNet_V3_Large_Weights":
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        model = ssdlite320_mobilenet_v3_large(weights=weights, box_score_thresh=0.9)

    
    model.to(device)
    next(model.parameters()).device
    model.eval()

    # Step 1: Initialize model with the best available weights
    # weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    # model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    # model.to(device)
    # next(model.parameters()).device
    # model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # # Lists to store all predicted and ground truth bounding boxes
    all_pred_boxes = []
    all_gt_boxes = []

    # Abrir el video con OpenCV VideoCapture
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, training_end)  # Empezar en la zona de test

    # Obtener propiedades del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)

    # Inicializar VideoWriters para cada método
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec AVI 
    # cv2.VideoWriter_fourcc(*'DIVX')
    out_frames = cv2.VideoWriter(os.path.join(output_dir, m+".avi"), fourcc, fps, frame_size, isColor=True)

    for idx, img in tqdm(enumerate(test_frames, start=1)):
        target_dict = {item["frame"]: item for item in gt_data}
        gt_dict= target_dict.get(idx)
        # print(gt_dict)
        image=Image.fromarray(img)
        # Step 3: Apply inference preprocessing transforms
        batch = [preprocess(image).to(device)]

        # Step 4: Use the model and visualize the prediction
        with torch.no_grad():
            prediction = model(batch)[0]

        labels = [weights.meta["categories"][i] for i in prediction["labels"]]
        # print(prediction)
        # print(labels)
        # pred_boxes=[]
        # for i, pred in enumerate(prediction['boxes']):
        
        #     if labels[i] == 'car':
        #         pred_boxes.append(pred.detach().cpu().numpy())
        # labels = [weights.meta["categories"][i] for i in pred_boxes["labels"]]
        # print(pred_boxes)

        
        # gt_boxes = gt_data.get(idx, [])
        # pred_boxes=metrics.merge_nearby_boxes(pred_boxes)
        # print('gt',gt_boxes)

        # print('pred',pred_boxes)
        # print(gt_boxes)

        # Store predictions and ground truth for later AP computation
        all_pred_boxes.append(prediction)
        all_gt_boxes.append(gt_dict)

        labels = [weights.meta["categories"][i] for i in prediction["labels"]]
        image = draw_bounding_boxes(
            pil_to_tensor(image),
            boxes=prediction["boxes"],
            labels=labels,
            colors="red",
            width=4,
            font_size=30,
        )

        labels = [weights.meta["categories"][i] for i in gt_dict["labels"]]
        box = draw_bounding_boxes(
            image,
            boxes=gt_dict["boxes"],
            labels=labels,
            colors="blue",
            width=4,
            font_size=30,
        )
        # Draw predicted bounding boxes in green
        # for box in pred_boxes:
        #     cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)  # Green

        # # Draw ground truth bounding boxes in red
        # for gt_box in gt_boxes:
        #     cv2.rectangle(img, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (255, 0, 0),
        #                 2)  # Red
        

        # Display results
        # cv2.imshow("Frame with Detections", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out_frames.write(cv2.cvtColor(np.array(to_pil_image(box)), cv2.COLOR_RGB2BGR))


        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # Press ESC to exit
            break
        # im = to_pil_image(box.detach())
        # im.show()


    # Compute mean Average Precision (mAP) for object detection
    # video_ap = metrics.compute_video_average_precision(all_pred_boxes, all_gt_boxes, iou_threshold=0.5)
    all_pred_boxes = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in pred.items()} for pred in all_pred_boxes]
    all_gt_boxes = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in gt.items()} for gt in all_gt_boxes]
    metric = MeanAveragePrecision(iou_type="bbox",class_metrics=True)
    metric.update(all_pred_boxes, all_gt_boxes)
    video_metrics=metric.compute()
    print(video_metrics)
    map=video_metrics['map']
    map50=video_metrics['map_50']
    map75=video_metrics['map_75']
    car_ap=video_metrics['map_per_class'][np.where(video_metrics['classes']==3)]
    bike_ap=video_metrics['map_per_class'][np.where(video_metrics['classes']==2)]
    print('mAP for car class: '+str(car_ap))
    cap.release()
    out_frames.release()


    with open("map_results.txt", "a") as f:
        f.write(f"model={m}, mAP={map}, mAP50={map50}, mAP75={map75}, mAP_car={car_ap}, mAP_bike={bike_ap}\n")


    print('Saved video at '+str(os.path.join(output_dir, m+".avi")))

    cv2.destroyAllWindows()
