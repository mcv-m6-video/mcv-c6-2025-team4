import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.transforms.functional import to_tensor
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.data import DataLoader, Dataset
from src import load_data, read_data
from sklearn.model_selection import KFold
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
data_dir = "./data/AICity_data/train/S03/c010"
annot_file = "./data/ai_challenge_s03_c010-full_annotation.xml"
video_path = os.path.join(data_dir, "vdo.avi")



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


# Load dataset
total_frames = load_data.get_total_frames(video_path)
k_folds=4
kf = KFold(n_splits=k_folds, shuffle=True)

gt_data = read_data.parse_annotations_xml(annot_file, isGT=True)
gt_dict = {item["frame"]: item for item in gt_data}
dataset = ObjectDetectionDataset(gt_data, video_path)


for fold, (test_idx, train_idx) in enumerate(kf.split(dataset)): #change test and train order because kfold assumes that the 25% corresponds to testing
    
    with open(str(fold)+"_fold_indices.txt", "a") as f:
          f.write(str(train_idx))
          
    train_loader = DataLoader(dataset, batch_size=4,collate_fn=lambda x: tuple(zip(*x)),sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    test_loader = DataLoader(dataset, batch_size=4,collate_fn=lambda x: tuple(zip(*x)),sampler=torch.utils.data.SubsetRandomSampler(test_idx))

    # Load pre-trained Faster R-CNN model
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)

    model.to(device)

    # Optimizer & loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 5

    # Training loop

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

    torch.save(model.state_dict(), str(fold)+"_fold_RANDOM_fine_tuned_faster_rcnn_05.pth")

    # Evaluation loop
    model.eval()
    all_pred_boxes, all_gt_boxes = [], []
    for images, targets in test_loader:
        images = list(img for img in images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        with torch.no_grad():
            predictions = model(images)

        for pred, target in zip(predictions, targets):
            all_pred_boxes.append(
                {"boxes": pred["boxes"].cpu(), "labels": pred["labels"].cpu(), "scores": pred["scores"].cpu()})
            all_gt_boxes.append(
                {"boxes": target["boxes"].cpu(), "labels": target["labels"].cpu()})

    # Compute mean Average Precision (mAP)
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    metric.update(all_pred_boxes, all_gt_boxes)
    video_metrics = metric.compute()
    print(video_metrics)

    with open("RANDOM_Faster_R_CNN_map_results.txt", "a") as f:
        f.write(
            f"RANDOM Fine-tuned Faster R-CNN fold: {fold}, mAP: {video_metrics['map']}, mAP50: {video_metrics['map_50']}, mAP75: {video_metrics['map_75']}\n")

    print("Fine-tuning and evaluation complete!")



