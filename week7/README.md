## Master in Computer Vision (Barcelona) 2024/25  
# Project 2 – Task 2: Ball Action Spotting (Week 7)  
C6 - Video Analysis

This repository contains the codebase and model implementations for Project 2: **Ball Action Spotting** using the SoccerNet Ball Action Spotting 2025 (SN-BAS-2025) dataset.

---

## Presentation

The slides to the presentation are in the following link [Slides](https://docs.google.com/presentation/d/14QikPjwjq13F65nfjwDNtAF7TYx914huTOx_ya_hraI/edit?usp=sharing)

---

## Dependencies

To install all the required packages:

```bash
pip install -r requirements.txt
```

---

## Dataset and Preprocessing

- Dataset: [SoccerNet Ball Action Spotting 2025 (SN-BAS-2025)](https://huggingface.co/datasets/SoccerNet/SN-BAS-2025)  
- Requires signing an NDA.
- Contains 7 full football matches: 4 for training, 1 for validation, 2 for testing.
- 12 action classes: pass, drive, header, high pass, out, cross, throw in, shot, ball player block, player successful tackle, free kick, goal.

### Preprocessing Details

- Clips of 50 frames  
- Stride of 2 (12.5 FPS)  
- 90% temporal overlap for training and validation; 0% for testing  
- Backbone: RegNetY or X3D to extract frame-level features  
- Frame-level classification with softmax over C+1 classes (includes no-action)

---

## Running the Spotting Code

Train or evaluate a model using:

```bash
python3 main_spotting_w7.py --model <model_name>
```

Where `<model_name>` matches a config file in `/config`, e.g., `baseline`, `TCNMultiscale2`, etc.

### Notes

- Set the mode parameter in the config to `store` (first run, generates clips) or `load` (reuse clips).
- Directory paths must be updated in the config files before running.

---

## Models Implemented

All models aim to improve temporal modeling over the baseline. Key architectures explored:

### Temporal Models

- TemporalAnchorSpotting (frame features + temporal convolutions)
- HybridTransformerDownscale (RegNetY + Transformer + BiLSTM + Downscaling)
- TCNTemporalOffset (multi-scale TCNs with separate heads)
- TCNMultiscale, TCNMultiscale2, TCNMultiscale3, TCNMultiscale4 (multi-scale TCN variants)
- TCNMultiscaleAttention (TCNs + attention)
- Slow Fusion (early fusion of frame and short temporal features)

### X3D-based Models

- X3D + LSTM
- X3D + Transformer
- X3D + Positional Encoding + Transformer + Pooling
- Bi-X3D + Transformer
- Bi-X3D + Positional Encoding + Transformer + Pooling

### Visual Pooling Models

- VLADNetModel (NetVLAD aggregation)
- TAVLADNetModel (TAPPooling + temporal convolutions)

Model code is in `model_spotting_w7.py`.

---

## Training Strategy

- Frame-wise cross-entropy loss  
  - Background class weight = 1  
  - Action classes weight = 5
- Linear warm-up followed by cosine annealing  
- Mixed precision training with GradScaler  
- Early stopping based on validation mAP

---

## Results (Week 7)

| Model | mAP₁₂ | mAP₁₀ | Params (M) | FLOPs (G) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline | 5.69 | 6.82 | 2.79 | 40.69 | Default setup |
| TemporalAnchorSpotting | 13.72 | 15.79 | 3.36 | 40.81 | Added a temporal convolutional module |
| HybridTransformerDownscale | 4.82 | 5.78 | 9.29 | 41.24 | Transformer + BiLSTM with downscaling |
| HybridTransformerDownscale2 | 7.79 | 9.35 | 11.38 | 41.34 | Double downscale version |
| TemporalAnchorSpottingTCN | 14.39 | 16.99 | 4.26 | 40.99 | RegNetY backbone with TCN |
| TCNTemporalOffset | 16.33 | 19.59 | 3.62 | 40.86 | Multi-scale TCNs with separate heads |
| TCNMultiscale | 18.94 | 20.00 | 3.61 | 40.86 | Multi-scale TCNs fused |
| TCNMultiscale (s1) | 18.04 | 18.32 | 3.61 | 40.86 | Stride 1 |
| TCNMultiscale2 | 18.46 | 22.15 | 3.76 | 40.89 | GroupNorm + SiLU feature fusion |
| TCNMultiscale3 | 16.04 | 19.25 | 4.65 | 41.10 | Temporal ASPP + Self-Attention |
| TCNMultiscale4 | 18.80 | 20.74 | 3.95 | 40.93 | MixUp augmentation on features |
| MultiScaleDownscale | 5.65 | 6.77 | 4.70 | 40.77 | Multi-scale downsampling + YOLO heads |
| TCNMultiscaleAttention | 16.00 | 16.47 | 3.38 | 40.81 | TCNs + Attention mechanisms |
| VLADNetModel | 0.58 | 0.70 | 5.83 | 40.71 | NetVLAD aggregation |
| TAVLADNetModel | 0.79 | 0.95 | 3.07 | 40.79 | TAPPooling + temporal convolutions |
| Slow Fusion | 4.68 | 5.60 | 2.95 | 60.45 | Slow fusion + temporal reduction |
| X3D + LSTM | 3.44 | 4.13 | 2.47 | 60.36 | X3D features + LSTM |
| X3D + Transformer | 1.82 | 2.19 | 2.90 | 60.55 | X3D features + Transformer |
| X3D + Pos. Encoding + Transformer + Pooling | - | - | - | - | Not evaluated separately |
| Bi-X3D + Transformer | 8.51 | 10.22 | 5.90 | 121.09 | Bidirectional X3D features |
| Bi-X3D + Pos. Encoding + Transformer + Pooling | - | - | - | - | Not evaluated separately |

---

## Insights

- Temporal convolutional models (especially multi-scale TCNs) outperform the baselines significantly.
- Hybrid Transformer models benefit from downscaling and BiLSTM integration, but temporal convolutions are still better.
- Attention mechanisms on top of TCNs offer additional improvements.
- VLAD and TAP pooling methods underperform compared to TCNs.
- X3D-based models are less effective unless fine-tuned heavily or combined with proper temporal heads.

---

## Best Models – Week 7

**Top Performing:**  
- TCNMultiscale2 (mAP₁₂: 18.46, mAP₁₀: 22.15)

Additionally, the `.pt` file for the best checkpoint is located in `week7/checkpoint_best.pt',. Ensure that this file is properly placed before running the model.

**Runner-ups:**  
- TCNMultiscale (mAP₁₂: 18.94, mAP₁₀: 20.00)  
- TCNMultiscale4 (mAP₁₂: 18.80, mAP₁₀: 20.74)

---

## Evaluation Metric

- mAP₁₂: mean AP across all 12 classes  
- mAP₁₀: excludes "free kick" and "goal"  
- Tolerance: 1 second (SoccerNet metric)

---

## Contact

- Albert Clapés: [aclapes@ub.edu](mailto:aclapes@ub.edu)  
- Artur Xarles: [arturxe@gmail.com](mailto:arturxe@gmail.com)
