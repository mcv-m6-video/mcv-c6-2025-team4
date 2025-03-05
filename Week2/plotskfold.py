import matplotlib.pyplot as plt
import numpy as np
from src import load_data, metrics, read_data
from torchmetrics.detection import MeanAveragePrecision
import torch
import os
import tqdm
import textwrap

import matplotlib.pyplot as plt
import numpy as np


# Read data from file
data_file = "map_results_k_fold.txt"  # Update with the actual filename
data = {}

with open(data_file, "r") as file:
    for line in file:
        parts = line.split(", ")
        model_fold = parts[0].split(" fold: ")
        model = model_fold[0].replace("Fine-tuned ", "")
        fold = int(model_fold[1])
        
        if model not in data:
            data[model] = {"mAP50": []}

        data[model]["mAP50"].append(float(parts[2].split(": ")[1]))

models = list(data.keys())

folds = len(next(iter(data.values()))["mAP50"])
bar_width = 0.2
x = np.arange(len(models))

plt.figure(figsize=(12, 12))

for i in range(folds):
    values = [data[model]["mAP50"][i] for model in models]
    plt.bar(x + i * bar_width, values, width=bar_width, label=f"Fold {i}")

models = ['\n'.join(textwrap.wrap(m, width=14)) for m in models]
plt.xlabel("Model")
plt.ylabel("mAP50 Score")
plt.ylim([0,1.0])
plt.title("K-Fold Cross Validation Results")
plt.xticks(x + bar_width * (folds / 2 - 0.5), models)
plt.legend()
plt.grid(axis='y')
plt.show()


# Read data from file
data_file = "map_results_k_fold_random.txt"  # Update with the actual filename
data = {}

with open(data_file, "r") as file:
    for line in file:
        parts = line.split(", ")
        model_fold = parts[0].split(" fold: ")
        model = model_fold[0].replace("RANDOM Fine-tuned ", "")
        fold = int(model_fold[1])
        
        if model not in data:
            data[model] = {"mAP50": []}
        
        data[model]["mAP50"].append(float(parts[2].split(": ")[1]))

models = list(data.keys())

folds = len(next(iter(data.values()))["mAP50"])
bar_width = 0.2
x = np.arange(len(models))

plt.figure(figsize=(12, 6))

for i in range(folds):
    values = [data[model]["mAP50"][i] for model in models]
    plt.bar(x + i * bar_width, values, width=bar_width, label=f"Fold {i}")

models = ['\n'.join(textwrap.wrap(m, width=14)) for m in models]
plt.xlabel("Model")
plt.ylabel("mAP50 Score")
plt.ylim([0,1.0])
plt.title("Random K-Fold Cross Validation Results")
plt.xticks(x + bar_width * (folds / 2 - 0.5), models)
plt.legend()
plt.grid(axis='y')
plt.show()



data_file = "map_results_k_fold.txt"  # Update with the actual filename
data = {}
with open(data_file, "r") as file:
    for line in file:
        parts = line.split(", ")
        model_fold = parts[0].split(" fold: ")
        model = model_fold[0].replace("Fine-tuned ", "")
        fold = int(model_fold[1])
        
        if model not in data:
            data[model] = {"mAP50": []}
        
        data[model]["mAP50"].append(float(parts[2].split(": ")[1]))

models = list(data.keys())

mAP_values = [data[model]["mAP50"] for model in models]

models = ['\n'.join(textwrap.wrap(m, width=14)) for m in models]
plt.figure(figsize=(10, 6))
plt.boxplot(mAP_values, labels=models, patch_artist=True, notch=True)

means = [np.mean(values) for values in mAP_values]
variances = [np.var(values) for values in mAP_values]

plt.xlabel("Model")
plt.ylabel("mAP50 Score")
plt.ylim([0.4,1.0])
plt.title("Box Plot of mAP Scores across folds")
plt.grid(axis='y')
plt.show()



data_file = "map_results_k_fold_random.txt"  # Update with the actual filename
data = {}
with open(data_file, "r") as file:
    for line in file:
        parts = line.split(", ")
        model_fold = parts[0].split(" fold: ")
        model = model_fold[0].replace("RANDOM Fine-tuned ", "")
        fold = int(model_fold[1])
        
        if model not in data:
            data[model] = {"mAP50": []}
        
        data[model]["mAP50"].append(float(parts[2].split(": ")[1]))

models = list(data.keys())

mAP_values = [data[model]["mAP50"] for model in models]

models = ['\n'.join(textwrap.wrap(m, width=14)) for m in models]
plt.figure(figsize=(10, 6))
plt.boxplot(mAP_values, labels=models, patch_artist=True, notch=True)

means = [np.mean(values) for values in mAP_values]
variances = [np.var(values) for values in mAP_values]

plt.xlabel("Model")
plt.ylabel("mAP50 Score")
plt.ylim([0.4,1.0])
plt.title("Box Plot of mAP Scores across Random folds")
plt.grid(axis='y')
plt.show()




# Read off-the-shelf mAP50 results
off_the_shelf_file = "map_results.txt"
fine_tuned_file = "map_results_k_fold.txt"

off_the_shelf_data = {}
fine_tuned_data = {}

with open(off_the_shelf_file, "r") as file:
    for line in file:
        parts = line.split(", ")
        model = parts[0].split("=")[1]
        mAP50 = float(parts[2].split("=")[1])
        off_the_shelf_data[model] = mAP50

with open(fine_tuned_file, "r") as file:
    for line in file:
        parts = line.split(", ")
        model_fold = parts[0].split(" fold: ")
        model = model_fold[0].replace("Fine-tuned ", "")
        mAP50 = float(parts[2].split(": ")[1])
        
        if model not in fine_tuned_data:
            fine_tuned_data[model] = []
        
        fine_tuned_data[model].append(mAP50)

models = list(off_the_shelf_data.keys())

off_the_shelf_values = [off_the_shelf_data[model] for model in models]
fine_tuned_means = [np.max(fine_tuned_data[model]) for model in models]

x = np.arange(len(models))
bar_width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - bar_width/2, off_the_shelf_values, width=bar_width, label="Off-the-Shelf", color="gray")
plt.bar(x + bar_width/2, fine_tuned_means, width=bar_width, label="Fine-Tuned", color="blue")

models = ['\n'.join(textwrap.wrap(m, width=14)) for m in models]
plt.xlabel("Model")
plt.ylabel("mAP50 Score")
plt.title("Comparison of mAP50 Scores: Off-the-Shelf vs Fine-Tuned")
plt.xticks(x, models)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()