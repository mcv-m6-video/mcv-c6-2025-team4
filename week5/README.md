# Master in Computer Vision (Barcelona) 2024/25
## Project 2 (Task 1) @ C6 - Video Action Classification

This repository provides the code for Task 1 of Project 2: Action Classification on the SoccerNet Ball Action Spotting 2025 (SN-BAS-2025) dataset. The goal of this task is to develop a model that can classify different actions occurring during a soccer match.

## Dependencies

To install the required dependencies, use the following command. The required versions of the various packages are specified in `requirements.txt`:

```
pip install -r requirements.txt
```

## Getting the Dataset and Data Preparation

Refer to the README files in the [data/soccernetball](data/soccernetball) directory for instructions on how to download the SNABS2025 dataset, preparation of directories, and extraction of the video frames.

## Running the Baseline for Task 1

The `main_classification.py` is designed to train and evaluate the baseline using the settings specified in a configuration file. You can run `main_classification.py` using the following command:


```
python3 main_classification.py --model <model_name>
```


Here, `<model_name>` can be chosen from the available models listed in the [config](config/) directory. For example, to choose the baseline model, you would run:

`python3 main_classification.py --model baseline`.

For additional details on configuration options using the configuration file, refer to the README in the [config](config/) directory.

## Important Notes

- Before running the model, ensure that you have downloaded the dataset frames and updated the directory-related configuration parameters in the relevant [config](config/) files.
- Make sure to run the `main_classification.py` with the `mode` parameter set to `store` at least once to generate the clips and save them. After this initial run, you can set the `mode` to `load` to reuse the same clips in subsequent executions.
  
### Model-Specific Configurations

For **X3D** and **SlowFast** models, the model name must be changed in the `model_classification.py` file, located in the [model](model/) directory. You need to set the correct model name to the one you want to use.

Additionally, the `.pt` file for the best checkpoint is located in `week5/checkpoint_best.pt',. Ensure that this file is properly placed before running the model.

## Best Model: X3D-6

While multiple versions of the X3D model were experimented with, **X3D-6** proved to be the best-performing model for this task. The following settings were used for this model:

- **Model**: X3D-6
- **Trainable Layers**: Three last blocks unfrozen
- **Epochs**: 15
- **Learning Rate**: 0.0008
- **Scheduler**: Linear Warmup (3) + Cosine Annealing LR (97)

### Performance (X3D-6)

- **AP₁₂**: 57.19
- **AP₁₀**: 55.51
- **Params**: 2.999M / 2.909M
- **FLOPs**: ~110.985G

### Class-wise Performance (X3D-6)
| Class                    | AP (%) |
|--------------------------|--------|
| PASS                     | 91.4   |
| DRIVE                    | 86.19  |
| HEADER                   | 64.58  |
| HIGH PASS                | 69.38  |
| OUT                      | 49.28  |
| CROSS                    | 49.89  |
| THROW IN                 | 69.16  |
| SHOT                     | 49.61  |
| BALL PLAYER BLOCK        | 20.2   |
| PLAYER SUCCESSFUL TACKLE | 5.45   |
| FREE KICK                | 100    |
| GOAL                     | 31.19  |

## Other Models Tested

- **Baseline**: Default setup for comparison.
- **rny004**: Using the RegNetY-004 feature extractor.
- **Baseline Attention**: Adding self-attention to the baseline model did not improve results as expected.
- **(2+1)D Conv**: A spatio-temporal convolution model, but computationally expensive.
- **X3D-1**: Fine-tuned last block for better temporal modeling.
- **X3D-2**: Same as X3D-1 but trained for 50 epochs.
- **X3D-3**: Linear Warmup (3) + Cosine Annealing LR (97).
- **X3D-4**: Lower learning rate (0.00008) didn't improve results.
- **X3D-5**: Unfreezing two last blocks significantly improved results.
- **X3D-6**: Best-performing model with three last blocks unfrozen.
- **X3D-7**: Extended training for 100 epochs.
- **X3D + Optical Flow**: Combined model using optical flow for improved motion understanding.
- **SlowFast**: Multi-pathway model, better for motion understanding, but computationally heavy.

