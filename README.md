# TEAM 4: Video Surveillance for Road Project & Human Action Recognition

Repository for the MCV C6 Module housing the implemented code of Team 4. 

Presentation link: https://docs.google.com/presentation/d/1PV_OrBNS-a47guJkhmbBVzSQKUFTouRWYfiWXaZqtKc/edit?usp=sharing 

## Week 1: Background Estimation

This project focuses on background estimation in video sequences for road traffic monitoring. The goal is to classify background and foreground objects, using statistical models. We implement a Single Gaussian Model for background estimation, including both adaptive and non-adaptive approaches. The results of these models will be compared against more complex, state-of-the-art techniques.

### Key Features:
- **Single Gaussian Model** for background estimation.
- **Adaptive and Non-Adaptive Models** for background subtraction.
- **Comparison with more advanced models** for background subtraction and object detection.

## Week 2: Video Segmentation and Object detection

This week implementation for object detection and tracking using various deep learning models and traditional methods were added. The project is designed to work with AI City dataset videos. 

This week, we tried and fine-tuned some pre-trained models (Faster RCNN + ResNet, RetinaNet + MobileNet, Yolo, ...) to perform object detection and for tracking, we implemented Maximum Overlpa and Kalman Filter. For evaluation, we implemented mAP50 for object detection and HOTA and IDENTITY for object tracking.

## Week 3: Optical flow

In this third week, we focused on implementing optical flow. We saw that it could be beneficil in some cases, and in other cases not. We tried some different optical flowmodels, but the best one regarding computational cost and accuracy was FlowFormer. 

## Week 4: Multi-camera tracking

This week, apart from doing multi-camera tracking by using the gps coordinates and some appearance based methods, we also tried to improve the object tracker of week 2 and 3 by using deep sort and a custom embedder, fastreid, which is also used to extract visual cues for the multi-camera tracking.
