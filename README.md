
# Video Surveillance for Road Traffic Monitoring - Week 1 - Background Estimation

## Overview

This project focuses on background estimation in video sequences for road traffic monitoring. The goal is to classify background and foreground objects, using statistical models. We implement a **Single Gaussian Model** for background estimation, including both **adaptive** and **non-adaptive** approaches. The results of these models will be compared against more complex, state-of-the-art techniques.

### Key Features:
- **Single Gaussian Model** for background estimation.
- **Adaptive and Non-Adaptive Models** for background subtraction.
- **Comparison with more advanced models** for background subtraction and object detection.

## Goals

1. **Background Estimation**: Model the background pixels in a video sequence using a simple Gaussian model.
2. **Single Gaussian per Pixel**: Use a single Gaussian distribution to model background for each pixel in the video sequence.
3. **Adaptive and Non-Adaptive Models**: Implement both adaptive (with updates) and non-adaptive (static model) approaches.
4. **Foreground Classification**: Classify pixels as foreground or background using the models.
5. **Model Comparison**: Compare the performance of these models against advanced background subtraction techniques.

## Tasks

- **Task 1.1: Gaussian Modeling**: Implement the basic Single Gaussian model for background estimation.
- **Task 1.2 & 1.3: Evaluate Results**: Evaluate the performance of the Gaussian model in classifying foreground and background.
- **Task 2.1: Recursive Gaussian Modeling**: Implement the recursive version of the Gaussian model, which adapts to changes over time.
- **Task 2.2: Evaluate and Compare Models**: Compare the recursive Gaussian model with the static non-recursive model.
- **Task 3: Compare with State-of-the-Art**: Compare the results of your models with more advanced, state-of-the-art background estimation models.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Scripts Overview](#scripts-overview)
   - [main.py](#mainpy)
   - [main_adaptive.py](#main_adaptivepy)
   - [gaussian_modelling.py](#gaussian_modellingpy)
   - [load_data.py](#load_datapy)
   - [metrics.py](#metricspy)
   - [read_data.py](#read_datapy)
   - [utils.py](#utils)
4. [Background Estimation Method](#background-estimation-method)
5. [Model Comparison](#model-comparison)
6. [Evaluation Metrics](#evaluation-metrics)

## Requirements

This project requires the following Python libraries:

- numpy
- opencv-python
- matplotlib
- scikit-learn
- xmltodict
- (additional libraries your project uses)

You can install all the required dependencies by using the `requirements.txt` file provided in this repository.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/background-estimation.git
   cd background-estimation
2. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
## Scripts Overview

### `main.py`

This script runs the non adaptive Gaussian model for background estimation.

- **Usage**: `python main.py`
- **Description**: The non adaptive model infers the background once and uses it to estimate foregroudn objects.

### `main_adaptive.py`

This script runs the adaptive Gaussian model, specifically designed for the adaptive background estimation technique.

- **Usage**: `python main_adaptive.py`
- **Description**: This script uses an adaptive approach to background modeling, where the Gaussian parameters are updated over time as new video frames are processed.

### `gaussian_modelling.py`

This script contains the implementation of the Gaussian model used in background estimation. It includes methods for:
- **Model Initialization**: Set up initial Gaussian parameters.
- **Updating Gaussian Parameters**: Adjusting mean and variance for each pixel.
- **Foreground Classification**: Using Gaussian distributions to classify pixels as foreground or background.

### `load_data.py`

This script handles the loading and preparation of video data for background estimation. It processes input video files and extracts frames for further analysis.

- **Usage**: `load_data.py` is called in the main scripts to load the necessary video data for processing.

### `metrics.py`

This script includes various evaluation metrics used to measure the performance of the background estimation models. The metrics include:
- **Precision**
- **Recall**
- **Intersection over Union (IoU)**
- **True Positive Rate (TPR) / False Positive Rate (FPR)**

### `read_data.py`

This script is used to read and process the data from video files. It prepares the data by converting video frames into formats that are suitable for background subtraction algorithms.

### `utils.py`

This script contains helper functions used by both `main.py` and `main_adaptive.py` for various tasks such as:
- **Data preprocessing**
- **Parameter setting**
- **Utility functions for background modeling**

## Background Estimation Method

Both adaptive and non-adaptive models use a single Gaussian distribution to represent the background at each pixel. Here's how each approach works:

- **Non-Adaptive Model**: This model uses a fixed Gaussian distribution for each pixel. The background model remains static throughout the video, calculated from an initial set of frames. This model does not update itself.

- **Adaptive Model**: In contrast, the adaptive model updates the Gaussian parameters (mean and variance) dynamically. It tracks changes over time, allowing the model to adapt to moving objects or environmental changes, such as changes in lighting or shadows.

In both cases, temporal median filtering is used to improve foreground detection and reduce noise in the video sequence.

## Model Comparison

Both the adaptive and non-adaptive models are evaluated based on their ability to separate background from foreground in the video footage. The comparison involves the following aspects:
- **Foreground Detection**: How accurately the model identifies moving objects (e.g., vehicles).
- **Adaptability**: The ability of the adaptive model to adjust to changes in the scene (compared to the fixed model).
- **Real-Time Processing**: How well the models perform in real-time, which is crucial for traffic monitoring.

Results will be compared against more advanced models such as YOLO, Mask-RCNN, and SSD to assess their performance in real-world applications.

## Evaluation Metrics

We evaluate the models using the following metrics:
- **Precision**: The percentage of true positive foreground detections out of all predicted foreground pixels.
- **Recall**: The percentage of true positive foreground detections out of all actual foreground pixels.
- **IoU (Intersection over Union)**: Measures the overlap between predicted and ground truth bounding boxes for foreground objects.
- **Average Precision (AP)**: A combined metric summarizing precision and recall over different IoU thresholds.

Pixel-level metrics include:
- **True Positive Rate (TPR)**: Accuracy of background and foreground classification at the pixel level.
- **False Positive Rate (FPR)**: Rate at which background pixels are misclassified as foreground.

## Usage

### Running the Adaptive Model

To run the adaptive background estimation model, execute the following command:

```bash
python main_adaptive.py

