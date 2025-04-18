# Week 2: Video Segmentation and Object detection

## Overview

This week implementation for object detection and tracking using various deep learning models and traditional methods were added. The project is designed to work with AI City dataset videos. 

This week, we tried and fine-tuned some pre-trained models (Faster RCNN + ResNet, RetinaNet + MobileNet, Yolo, ...) to perform object detection and for tracking, we implemented Maximum Overlpa and Kalman Filter. For evaluation, we implemented mAP50 for object detection and HOTA and IDENTITY for object tracking.

## Project structure

```bash
|-- src/
|  |-- load_data.py          # Functions to load video frames
|  |-- read_data.py          # Functions to load annotations
|  |-- compute_metrics.py    # Functions to compute mean average precision and other metrics
|  |-- hota.py               # File with functions necessary to compute HOTA metric
|  |-- identity.py           # File with functions necessary to compute IDENTITY metric
|  |-- metrics.py            # File inherited from las week that computes metrics (as mAP)
|  |-- plots.py              # File that contains ways of computing plots
|  |-- sort.py               # Utilities used in object tracking
|
|-- task1_1.py               # Object detection using multiple deep learning models
|-- task1_1_yolo.py          # YOLO-based object detection and evaluation
|-- task1_2.py               # Fine-tuning object detection models
|-- task1_3.py               # K-fold strategy B and evaluation of the fine-tuned models
|-- task1_3random.py         # K-fold strategy C (random) and evaluation of the fine-tuned models
|-- task2_1.py               # Object tracking using maximal overlapping method
|-- task2_2.py               # Object tracking using Kalman Filter
|-- task2_3.py               # Identity and Hota metrics on the object tracking method
```

## Dependencies

To install the required dependencies of this project, execute the following command:

```bash
pip install -r requirements.txt
```
## Methodology

This project implements:
- **Object detection**: Uses models such as Faster-RCNN, RetineNet, SSD and YOLO for vehicle detection.
- **Tracking**: Implements tracking using Maximal Overlap and Kalman Filters, after some preprocessing.
- **Evaluation metrics**: Computes mean Average Precision (mAP) for object detection and ID metrics for tracking evaluation.

## Usages of Scripts

For each task, this is the way of executing the files:

```bash
python Week2/{task_file.py}
```
Nevertheless, some tasks require additional files:

### Task 2.1 and 2.2

Require to have the .pth model saved and trained so as to use it in object tracking. You can solve it in two different ways:

1. If you have not already done it, run **task 1.2** so as to achieve a saved pretrained model.
2. If not, download out best pretrained model from here and save it outside the Week2 folder:
   https://drive.google.com/file/d/1DcAAeBvltv22vaUyItMqNepfH_bjTaho/view?usp=sharing

These two files will create two .txt files (MOTS-train-1.txt and MOTS-train.txt, respectively), which will be used in the following task. These files contain the predictions made in the object tracking, saved un the MOTS file format).

### Task 2.3

This task computes the HOTA and IDENTITY metrics for object tracking. Having the files saved earlier, we can compute them by changing the name of the file in the bottom of the script.

You can also load these files from here and place them on the Week2 folder: 
1. Maximal Overlap method (MOTS-train-1.txt): https://drive.google.com/file/d/1DWvDNhZhlaW-MzyYPWj3pgOcNepilGtC/view?usp=sharing
2. Kalman Filters (MOTS-train.txt): https://drive.google.com/file/d/10WuN455wvJXIVOhIlwLmWR44C08F0uIP/view?usp=sharing





