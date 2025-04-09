### Master in Computer Vision (Barcelona) 2024/25
# Project 2 (Task 2) @ C6 - Video Analysis

This repository provides the starter code for Task 2 of Project 2: Action spotting on the SoccerNet Ball Action Spotting 2025 (SN-BAS-2025) dataset.

The installation of dependencies, how to obtain the dataset, and instructions on running the spotting baseline are detailed next.

## Dependencies

You can install the required packages for the project using the following command, with `requirements.txt` specifying the versions of the various packages:

```
pip install -r requirements.txt
```

## Getting the dataset and data preparation

Refer to the README files in the [data/soccernetball](/data/soccernetball) directory for instructions on how to download the SNABS2025 dataset, preparation of directories, and extraction of the video frames.


## Running the baseline for Task 2

The `main_spotting.py` is designed to train and evaluate the baseline using the settings specified in a configuration file. You can run `main_spotting.py` using the following command:

```
python3 main_spotting.py --model <model_name>
```

Here, `<model_name>` can be chosen freely but must match the name of a configuration file (e.g. `baseline.json`) located in the config directory [config](/config/). For example, to chose the baseline model, you would run: `python3 main_spotting.py --model baseline`.

For additional details on configuration options using the configuration file, refer to the README in the [config](/config/) directory.

## Important notes

- Before running the model, ensure that you have downloaded the dataset frames and updated the directory-related configuration parameters in the relevant [config](/config/) files.
- Make sure to run the `main_spotting.py` with the `mode` parameter set to `store` at least once to generate the clips and save them. After this initial run, you can set the `mode` to `load` to reuse the same clips in subsequent executions.

## Model specific configurations
Replace `<model_name>` with the name of the model you want to run.  Refer to the code and configuration files for the available model names.

## Model Variations and Experiments

This project explores several variations of the baseline model with a focus on temporal modeling:

-   **Baseline Model + LSTM:** An LSTM layer is added to the baseline to capture temporal dependencies across video frames. This helps the network understand motion patterns and temporal context. 
-   **LSTM with Stride 3:** Experiments with increasing the temporal stride to 3 within the LSTM to reduce redundancy between consecutive frames.
-   **Transformer (Encoder-only):** A Transformer-based temporal modeling architecture is implemented, replacing LSTM/GRU models with a Transformer encoder.
-   **Hybrid (Transformer Encoder and LSTM Decoder):** Combines a Transformer encoder with an LSTM decoder to leverage both global attention and local sequential modeling. 
-   **Perceiver-styled Cross-Attention:** Explores a Perceiver IO approach with a latent bottleneck of queries and cross-attention. 
-   **Temporal Pyramid Transformer:** Uses a temporal transformer operating at multiple temporal scales.
-   **X3D with Unfrozen Last Layer + LSTM:** Replaces the classification head of the X3D model with a BiLSTM and unfreezes the last block for training. 
-   **SnaTCHer (Transformer with Time-Conditioned Embeddings):** Implements a temporal transformer with time-conditioned embeddings to create continuous time-aware positional encodings.
-   **SnaTCHer + LSTM:** Combines the SnaTCHer architecture with an LSTM layer.
-   **Slow Fusion:** A model that concatenates consecutive frames and uses a combination of 2D and 3D convolutions.
-   
## Training Strategy

All models were trained using the following strategy:

-   **Linear Warm-up:** A linear warm-up phase was used to gradually increase the learning rate in the initial epochs. [cite: 30, 31]
-   **Cosine Annealing Learning Rate Scheduler:** A Cosine Annealing Learning Rate Scheduler was applied to progressively decrease the learning rate. [cite: 31]
-   **Mixed Precision Training:** Mixed precision training with GradScaler was used for computational efficiency. [cite: 32]
-   **Early Stopping:** Early stopping was implemented based on validation mAP. [cite: 33]

## Results and Analysis

(This section summarizes the key findings from your ablation study. Adapt the numbers as needed.)

| Model                                  | AP₁₂  | AP₁₀  | Params (M) | FLOPs (G) | Notes                                                                                                | Improvement AP₁₀ over baseline |
| :------------------------------------- | :---- | :---- | :--------- | :-------- | :--------------------------------------------------------------------------------------------------- | :----------------------------- |
| Baseline                               | 5.69  | 6.82  | 2.79       | 40.7      | Default setup.                                                                                       | 0                              |
| Baseline + LSTM                        | 18.6  | 21.23 | 6.7        | 40.7      | Adding LSTM.                                                                                         | 9.53                           |
| LSTM with Stride 3                     | 13.62 | 16.35 | 6.7        | 40.7      | Changing stride.                                                                                     | 14.41                          |
| Transformer Encoder                    | 6.9   | 8.28  | 6.9        | 41.52     | Adding a transformer encoder.                                                                          | 1.46                           |
| Transformer Encoder + LSTM Decoder     | 19.42 | 21.49 | 9.02       | 41.22     | Combining transformer encoder and LSTM decoder.                                                        | 14.67                          |
| Perceiver Transformer                  | 3.59  | 4.31  | 6.49       | 40.9      | Using a perciever transformer.                                                                         | -1.23                          |
| Pyramid Transformer                    | 4.66  | 5.59  | 5.56       | 41.25     | Using a pyramid transformer.                                                                           | -2.51                          |
| X3D - Unfreeze Last Layer             | 4.04  | 4.85  | 7.96       | 61.36     | Unfreezing the last layer of X3D model.                                                              | -1.97                          |
| SNATCHER                               | 16.1  | 16.6  | 6.91       | 41.52     | SNATCHER                                                                                             | 9.78                           |
| SNATCHER + LSTM                        | 6.53  | 7.83  | 10.53      | 41.52     | SNATCHER + hybrid                                                                                    | 1.01                           |
| Slow Fusion                            | 0.53  | 0.63  | 7.03       | 153.065   | Slow fusion                                                                                          | -6.19                          |

### Key Insights from Ablation Study

-   Explicitly modeling temporal information (e.g., with LSTM) significantly improves performance over the baseline. 
-   Increasing temporal stride with LSTM can reduce performance, possibly due to over-downsampling.
-   Transformer-based approaches showed potential but required careful hyperparameter tuning.
-   Combining a Transformer encoder with an LSTM decoder achieved strong performance. 
-   Fine-tuning pre-trained models (like X3D) on the target dataset is crucial (only training last layers is not enough). 

### Best Model and Conclusions (X3D-6)

The X3D-6 model, with three unfrozen layers, achieved the best performance.

-   Unfreezing more layers in X3D improves performance (X3D-5 vs. X3D-6). 
-   (2+1)D Conv is computationally expensive.
-   Feature extractor choice matters (RegNetY-004 was less effective).
-   Optical Flow did not provide significant improvements in this setup. 
-   Lowering the learning rate was not effective. 
-   SlowFast is computationally heavy. 

**Conclusions on best model:**

-   X3D-6 with three unfrozen layers is the most optimized model, balancing accuracy and computational efficiency. 
-   Linear warmup and cosine annealing are effective for learning rate scheduling. 
-   Further hyperparameter tuning and longer training could potentially improve performance. 
-   The model performs well on frequent action classes (PASS, DRIVE) but needs improvement on less frequent ones (PLAYER SUCCESSFUL TACKLE, BALL PLAYER BLOCK). 

## Support

For any issues related to the code, please email [aclapes@ub.edu](mailto:aclapes@ub.edu) and CC [arturxe@gmail.com](mailto:arturxe@gmail.com).
