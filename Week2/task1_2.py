import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights, fasterrcnn_mobilenet_v3_large_320_fpn, fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights, retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_Weights, retinanet_resnet50_fpn,
    SSD300_VGG16_Weights, ssd300_vgg16, SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large, FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from src import load_data, read_data

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
data_dir = "./data/AICity_data/train/S03/c010"
annot_file = "./data/ai_challenge_s03_c010-full_annotation.xml"
video_path = os.path.join(data_dir, "vdo.avi")

# Load dataset
total_frames = load_data.get_total_frames(video_path)
training_end = int(total_frames * 0.25)
test_frames = load_data.load_frames_list(video_path, start=training_end, end=total_frames)
training_frames = load_data.load_frames_list(video_path, start=0, end=training_end)

gt_data = read_data.parse_annotations_xml(annot_file, isGT=True)
gt_dict = {item["frame"]: item for item in gt_data}

# Custom dataset class for fine-tuning
class ObjectDetectionDataset(Dataset):
    def __init__(self, gt_data, video_path):
        self.gt_data = gt_data
        self.video_path = video_path

    def __len__(self):
        return len(self.gt_data)

    def __getitem__(self, idx):
        frame_info = self.gt_data[idx]
        frame = frame_info["frame"]
        image = load_data.load_video_frame(self.video_path, frame)
        image = to_tensor(image).to(device)  # Convert numpy to PyTorch tensor and send to device
        boxes = torch.as_tensor(frame_info["boxes"], dtype=torch.float32).to(device)
        labels = torch.ones((len(boxes),), dtype=torch.int64).to(device)  # Assuming one class (car)
        target = {"boxes": boxes, "labels": labels}
        return image, target

# Prepare dataset and dataloaders
dataset = ObjectDetectionDataset(gt_data, video_path)
train_size = int(0.25 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Load pre-trained Faster R-CNN model
weights = FCOS_ResNet50_FPN_Weights.DEFAULT
model = fcos_resnet50_fpn(weights=weights)
#for param in model.backbone.parameters():
#    param.requires_grad = False

# Get the number of input features for the classifier
#in_features = model.roi_heads.box_predictor.cls_score.in_features

# Define the number of classes (including background)
#num_classes = 3

# Replace the pre-trained head with a new one
#model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

# Optimizer & learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR by 10x every 10 epochs
num_epochs = 100

# Training loop
mAP_max = 0
for epoch in range(num_epochs):
    running_loss = 0.0
    print("Epoch {}/{}".format(epoch + 1, num_epochs))
    model.train()
    for images, targets in tqdm(train_loader):
        images = list(img for img in images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()
    print(f"Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader)}")

    # Step the scheduler
    scheduler.step()

    # Evaluation loop
    model.eval()
    all_pred_boxes, all_gt_boxes = [], []
    for images, targets in tqdm(test_loader):
        images = list(img for img in images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        with torch.no_grad():
            predictions = model(images)

        for pred, target in zip(predictions, targets):
            all_pred_boxes.append(
                {"boxes": pred["boxes"].cpu(), "labels": pred["labels"].cpu(), "scores": pred["scores"].cpu()})
            all_gt_boxes.append({"boxes": target["boxes"].cpu(), "labels": target["labels"].cpu()})

    # Compute mean Average Precision (mAP)
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    metric.update(all_pred_boxes, all_gt_boxes)
    video_metrics = metric.compute()
    print(video_metrics)

    if float(video_metrics["map_50"]) > mAP_max:
        mAP_max = float(video_metrics["map_50"])
        torch.save(model.state_dict(), "fcos_resnet50_05_og.pth")
        print("Saved model at epoch {}".format(epoch + 1))

print("Fine-tuning and evaluation complete!")




