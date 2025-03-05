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
|-- task1_3.py               # K-fold and random evaluation of the fine-tuned models
|-- task2_1.py               # Object tracking using maximal overlapping method
|-- task2_2.py               # Object tracking using Kalman Filter
|-- task2_3.py               # Identity and Hota metrics on the object tracking method
```

## Dependencies

To install the required dependencies of this project, execute the following command:

```bash
pip install -r requirements.txt
```


