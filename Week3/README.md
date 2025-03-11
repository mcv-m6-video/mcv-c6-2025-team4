# - Week 3

## Overview
This folder contains scripts and files related to Week 3 of the project, focusing on optical flow estimation and multi-target single-camera tracking.

### Goals for Week 3:
- Estimate the optical flow of a video sequence.
- Improve object tracking using optical flow.
- Evaluate tracking results using AI City Challenge datasets.

## Dataset: AI City Challenge
The dataset used for this project comes from the AI City Challenge, which contains traffic surveillance videos captured from multiple cameras in an urban environment.

### **Dataset Structure**
- **`cam_framenum/`** → Contains TXT files (`S01.txt`, `S02.txt`, etc.) listing the number of frames per sequence.
- **`cam_loc/`** → PNG images (`S01.png`, `S02.png`) showing the camera locations.
- **`cam_timestamp/`** → TXT files (`S01.txt`, `S02.txt`) containing timestamps for each frame.
- **`train/`** → The main folder containing video sequences and related annotations:
  - **`S01/`, `S03/`, `S04/`** → Each represents a different sequence.
  - Inside each sequence, multiple cameras are labeled (`c001/`, `c002/`, ...).
  - Inside each camera folder:
    - `det/` → Object detections from different models (Mask R-CNN, SSD512, YOLOv3).
    - `gt/` → Ground truth vehicle annotations (`gt.txt`).
    - `mtsc/` → Multi-target tracking results from different algorithms.
    - `segm/` → Segmentation data (`segm_mask_rcnn.txt`).
    - `calibration.txt` → Camera calibration information.
    - `roi.jpg` → Region of Interest image showing where vehicles are detected.
    - `vdo.avi` → Video file containing the recorded traffic scene.

The dataset is used to evaluate vehicle tracking and detection algorithms. Our focus is on tracking vehicles within single-camera views.

## Folder Contents

### **Example Files**
- `000045_10.png`: Example frame from the dataset.
- `000045_10_gt.png`: Ground truth optical flow for the frame.
- `000045_11.png`: Next frame in the sequence.
- `outFlow_farneback.npy`: Optical flow output using the Farneback method (NumPy format).
- `outFlow_farneback.png`: Visual representation of the Farneback optical flow.
- `detections.json`: Contains detected objects and tracking information.

### **Scripts**
#### **Optical Flow Estimation**
- `farneback_task1_1.py`: Computes optical flow using the Farneback method (OpenCV).
- `pyflow_task1_1.py`: Computes optical flow using PyFlow.
- `raft_task1_1.py`: Computes optical flow using RAFT (Recurrent All-Pairs Field Transforms).

#### **Tracking & Evaluation**
- `task1_2.py`: Implements object tracking using optical flow results.
- `graphs1.py`: Generates graphs and visualizations of the tracking performance.
- `hota_idf1.py`: Computes HOTA and IDF1 metrics for tracking evaluation.
- `output_mot_format_of.txt`: Stores tracking results in MOT format for further evaluation.

## How to Use
1. **Run Optical Flow Computation**
   - Choose an optical flow method and execute the corresponding script (e.g., `farneback_task1_1.py`).
   - This generates optical flow outputs (`.npy`, `.png`).

2. **Perform Object Tracking**
   - Run `task1_2.py` to apply tracking using the computed optical flow.
   - The script updates `detections.json` with tracking results.

3. **Evaluate Tracking Performance**
   - Run `hota_idf1.py` to compute HOTA/IDF1 metrics.
   - Use `graphs1.py` to visualize results.

## Notes
- The dataset is from the AI City Challenge (sequences SEQ01, SEQ03, SEQ04).
- Tracking results should be compared against ground truth annotations.
- Optical flow techniques (RAFT, Farneback, PyFlow) impact tracking accuracy differently.



