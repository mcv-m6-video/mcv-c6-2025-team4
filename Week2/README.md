# Week 2: Video Segmentation and Object detection

## Overview

This week implementation for object detection and tracking using various deep learning models and traditional methods were added. The project is designed to work with AI City dataset videos. 

This week, we tried and fine-tuned some pre-trained models (Faster RCNN + ResNet, RetinaNet + MobileNet, Yolo, ...) to perform object detection and for tracking, we implemented Maximum Overlpa and Kalman Filter. For evaluation, we implemented mAP50 for object detection and HOTA and IDENTITY for object tracking.

## Project structure

```bash
|-- src/
|  |-- load_data.py
|  |-- read_data.py
|  |-- compute_metrics.py
|  |-- convert.py
|  |-- hota.py
|  |-- identity.py
|  |-- metrics.py
|  |-- plots.py
|  |-- sort.py
|
|-- task1_1.py
|-- task1_1_yolo.py
|-- task1_2.py
|-- task1_3.py
|-- task2_1.py
|-- task2_2.py
|-- task2_3.py
```
