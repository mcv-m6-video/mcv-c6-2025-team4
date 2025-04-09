
## Master in Computer Vision (Barcelona) 2024/25  
# Project 2 – Task 2: Ball Action Spotting (Week 6)  
C6 - Video Analysis

This repository contains the codebase and model implementations for Task 2 of Project 2: **Ball Action Spotting** using the SoccerNet Ball Action Spotting 2025 (SN-BAS-2025) dataset.

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
- Backbone: RegNetY (without final FC layer) to extract frame-level features  
- Frame-level classification with softmax over C+1 classes (includes no-action)

---

## Running the Spotting Code

Train or evaluate a model using:

```bash
python3 main_spotting.py --model <model_name>
```

Where `<model_name>` matches a config file in `/config`, e.g., `baseline`.

### Notes

- Set the mode parameter in the config to `store` (first run, generates clips) or `load` (reuse clips).
- Directory paths must be updated in the config files before running.

---

## Models Implemented

All models maintain temporal resolution (no downsampling). Variants build on the baseline by adding temporal modeling layers.

### Variations

- Baseline (RegNet-Y + FC)
- Baseline + LSTM
- LSTM with Stride 3
- Transformer (Encoder-only)
- Transformer Encoder + LSTM Decoder (Hybrid)
- Perceiver-style Transformer
- Temporal Pyramid Transformer
- X3D + BiLSTM (Unfreeze last block)
- SnaTCHer (time-conditioned embeddings)
- SnaTCHer + LSTM
- Slow Fusion (2D + 3D convolution)

Model code is in `model_spotting.py`.

---

## Training Strategy

- Frame-wise cross-entropy loss  
  - Background class weight = 1  
  - Action classes weight = 5
- 20 epochs  
- Linear warm-up followed by cosine annealing  
- Mixed precision training with GradScaler  
- Early stopping based on validation mAP

---

## Results (Week 6)

| Model                              | mAP₁₂ | mAP₁₀ | Params (M) | FLOPs (G) | Notes |
| ---------------------------------- | ------| ------|-------------|-----------|-------|
| Baseline                           | 5.69  | 6.82  | 2.79        | 40.7      | Default setup |
| Baseline + LSTM                    | 18.60 | 21.23 | 6.7         | 40.7      | Adds LSTM |
| LSTM with Stride 3                 | 13.62 | 16.35 | 6.7         | 40.7      | Increased temporal stride |
| Transformer (Encoder)              | 6.90  | 8.28  | 6.9         | 41.52     | No positional encodings |
| Transformer + LSTM (Hybrid)       | 19.42 | 21.49 | 9.02        | 41.22     | Best performing this week |
| Perceiver Transformer              | 3.59  | 4.31  | 6.49        | 40.9      | Lightweight, underperforms |
| Pyramid Transformer                | 4.66  | 5.59  | 5.56        | 41.25     | Multi-scale modeling |
| X3D + Unfreeze Last Block + LSTM  | 4.04  | 4.85  | 7.96        | 61.36     | Fine-tunes final block only |
| SnaTCHer (Time Embeddings)         | 16.10 | 16.60 | 6.91        | 41.52     | Time-aware transformer |
| SnaTCHer + LSTM                    | 6.53  | 7.83  | 10.53       | 41.52     | Redundant, performs worse |
| Slow Fusion                        | 0.53  | 0.63  | 7.03        | 153.06    | Sliding window 2D+3D |

---
| Hybrid + Positional Encoding         | 25.92 | 24.74 | 9.02        | 41.22     | Best overall (positional encoding + hybrid) |
## Insights

- Explicit temporal modeling improves performance significantly (LSTM, Transformer).
- Increasing stride in LSTM models hurts performance due to reduced context.
- Transformer-only models require proper tuning and benefit from added LSTM.
- Best results are achieved by combining Transformer and LSTM.
- Unfreezing only the last block in X3D limits its learning capacity.
- Slow Fusion is computationally expensive and underperforms.

---

## Best Model – Week 6

**Hybrid Transformer + LSTM**

- Highest mAP₁₂: 19.42  
- Highest mAP₁₀: 21.49  
- Parameters: ~9M  
- Balanced in terms of accuracy and computation  
- Combines global and local temporal reasoning

---

## Evaluation Metric

- mAP₁₂: mean AP across all 12 classes  
- mAP₁₀: excludes "free kick" and "goal"  
- Tolerance: 1 second (SoccerNet metric)

---

## Contact

- Albert Clapés: [aclapes@ub.edu](mailto:aclapes@ub.edu)  
- Artur Xarles: [arturxe@gmail.com](mailto:arturxe@gmail.com)

