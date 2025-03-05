
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
## Week1: Scripts overview

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

## Week 2: Scripts Overview

### Usage

All scripts are executed on the following way: 

```bash
python Week2/task{task_first_number}_{task_second_number}.py
```

### Task 1.1 and Task 1.1 yolo

Tests the pre-trained models on our target domain without further training. 

### Task 1.2

Fine-tunes the models described on task 1.1, and outputs their performance on mAP metrics for the test set. 75% of the data was used for testing. It also saves the best finetuned weights for the specified model.

### Task 1.3

Taking the fine-tuned model (.pth format), 

