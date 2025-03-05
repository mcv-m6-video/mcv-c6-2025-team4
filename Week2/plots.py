import matplotlib.pyplot as plt
import numpy as np
from src import load_data, metrics, read_data
from torchmetrics.detection import MeanAveragePrecision
import torch
import os
import tqdm
import textwrap

# Read data from file
file_path = "map_results.txt" 

models = []
mAP50= []

with open(file_path, "r") as file:
    for line in file:

        parts = line.split(", ")
        model_name = parts[0].split("=")[1]
        models.append(model_name)
        mAP50.append(float(parts[2].split("=")[1]))


models_wrapped = ['\n'.join(textwrap.wrap(m, width=14)) for m in models]

# Plotting
x = np.arange(len(models))  # X-axis positions
width = 0.5  # Width for each bar

fig, ax = plt.subplots(figsize=(14, 7))

ax.bar(x,mAP50,width)

# Formatting
ax.set_xlabel("Model")
ax.set_ylabel("mAP50 Score")
ax.set_title("Object Detection Performance Off-the-shelf",)
ax.set_xticks(x)
ax.set_ylim([0,1.0])
ax.set_xticklabels(models_wrapped, ha="center")
ax.legend()
plt.tight_layout()

# Show the plot
plt.show()
