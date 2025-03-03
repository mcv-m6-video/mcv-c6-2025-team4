
# README - Object Detection and Tracking Project

## Overview
This week's project focuses on object detection and tracking in video using various deep learning models to enhance accuracy and robustness. The goal is to identify and track objects such as cats, dogs, and ducks in the **S03 - C010** video sequence.

The presentation is divided into different tasks reflecting each phase of the work:

---

## **Task 1: Object Detection**

### **1.1. Off-the-Shelf Object Detection**
- Pretrained models such as **YOLO, Faster R-CNN, RetinaNet, and SSD** are used to detect objects in video frames.
- Predictions from different models are compared to select the most suitable one.
- Differences in accuracy and detection quality are analyzed.

### **1.2. Fine-tuning the Model with Specific Data**
- Pretrained models are adjusted using project-specific data.
- **Transfer Learning** is employed to improve detection of target classes.
- Different configurations of **layer freezing and head modifications** are tested.

### **1.3. K-Fold Cross-Validation**
- Model performance is evaluated by dividing training and test data in various combinations.
- Strategies used:
  - **Strategy A**: First 25% of frames for training, 75% for testing.
  - **Strategy B**: K-Fold Cross-validation (K=4, fixed splits).
  - **Strategy C**: Random K-Fold (25% for training, rest for testing).

---

## **Task 2: Object Tracking**

### **2.1. Tracking by Overlap**
- A **unique ID** is assigned to each object in the current frame and matched to the most overlapping object (highest IoU) in the next frame.
- **Bounding boxes with excessive overlap** (IoU > 0.9) are filtered out.
- A new ID is assigned if an object disappears or reappears elsewhere.

### **2.2. Tracking with Kalman Filter**
- Tracking is improved using a motion model based on **Kalman Filter**.
- The **SORT (Simple Online and Realtime Tracking)** methodology is applied:
  - Object position (u, v).
  - Bounding box area and aspect ratio.
  - Estimated velocities.
- Results are compared with the overlap tracking method (2.1).

### **2.3. Tracking Evaluation (IDF1, HOTA)**
- Evaluation metrics are used to analyze tracking performance:
  - **IDF1**: Measures how well objects are identified throughout the video.
  - **HOTA**: Measures combined detection and tracking accuracy.
- Results from tracking methods **Overlap (2.1) vs. Kalman Filter (2.2)** are compared.
- A results table is generated for final analysis.

---

## Scripts and requirements

### Running the Adaptive Model

To run the adaptive background estimation model, execute the following command:

```bash
python main_adaptive.py

