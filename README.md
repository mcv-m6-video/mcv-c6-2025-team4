# Video Surveillance for Road Traffic Monitoring - Week 1 - Background estimation


## Overview

This project focuses on modeling the background pixels of a video sequence using a simple statistical model to classify the background and foreground. The model uses a **Single Gaussian per pixel** and includes both **adaptive** and **non-adaptive** approaches for background estimation. The goal is to preliminarily classify foreground objects from the background and compare the performance of this simple model with more complex state-of-the-art models.

### Key Features:
- **Single Gaussian model** for background estimation.
- **Adaptive** and **Non-Adaptive** approaches for background modeling.
- Comparison of the results with more complex models.

## Goals

1. **Background Estimation**: Model the background pixels of a video sequence using a simple statistical model.
2. **Single Gaussian per Pixel**: Use a single Gaussian distribution to model the background for each pixel in the video sequence.
3. **Adaptive and Non-Adaptive Models**: Implement both adaptive and non-adaptive models to classify foreground and background.
4. **Preliminary Foreground Classification**: Use the statistical model to preliminarily classify pixels as foreground or background.
5. **Comparison with Complex Models**: Compare the simple model with more complex models to assess its effectiveness.

## Tasks

- **Task 1.1: Gaussian Modelling**: Implement the single Gaussian model to represent background pixels.
- **Task 1.2 & 1.3: Evaluate Results**: Evaluate the performance of the Gaussian model in classifying background and foreground.
- **Task 2.1: Recursive Gaussian Modelling**: Implement a recursive version of the Gaussian model to adapt to changes in the video.
- **Task 2.2: Evaluate and Compare to Non-Recursive**: Compare the recursive Gaussian model with the non-recursive model.
- **Task 3: Compare with State-of-the-Art**: Compare the results of your models with more advanced, state-of-the-art background estimation models.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Running Adaptive Model](#running-adaptive-model)
   - [Running Non-Adaptive Model](#running-non-adaptive-model)
4. [Background Estimation Method](#background-estimation-method)
5. [Model Comparison](#model-comparison)

## Requirements

This project requires the following Python libraries:

- numpy
- opencv-python
- matplotlib
- scikit-learn
- (additional libraries your project uses)

You can install all the required dependencies by using the `requirements.txt` file provided in this repository.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/background-estimation.git
   cd background-estimation

