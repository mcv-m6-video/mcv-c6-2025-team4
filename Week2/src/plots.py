import matplotlib.pyplot as plt
import numpy as np

# Read data from file
file_path = "map_results.txt"  # Update if using a different filename

models = []
mAP, mAP50, mAP75 = [], [], []
mAP_car, mAP_bike = [], []

with open(file_path, "r") as file:
    for line in file:
        parts = line.split(", ")
        model_name = parts[0].split("=")[1]
        models.append(model_name)
        mAP.append(float(parts[1].split("=")[1]))
        mAP50.append(float(parts[2].split("=")[1]))
        mAP75.append(float(parts[3].split("=")[1]))
        mAP_car.append(float(parts[4].split("=")[1].replace("tensor([", "").replace("])", "")))
        mAP_bike.append(float(parts[5].split("=")[1].replace("tensor([", "").replace("])", "")))

# Plotting
x = np.arange(len(models))  # X-axis positions
width = 0.15  # Width for each bar

fig, ax = plt.subplots(figsize=(14, 7))

# ax.bar(x - 2 * width, mAP, width, label="mAP")
ax.bar(x - width, mAP50, width, label="mAP50")
ax.bar(x, mAP75, width, label="mAP75")
ax.bar(x + width, mAP_car, width, label="mAP_car")
ax.bar(x + 2 * width, mAP_bike, width, label="mAP_bike")

# Formatting
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_title("Object Detection Performance (mAP Metrics)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha="right")
ax.legend()
plt.tight_layout()

# Show the plot
plt.show()
