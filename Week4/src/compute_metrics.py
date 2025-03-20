import json
import torch
import numpy as np
from torchmetrics.detection import MeanAveragePrecision
from src import metrics

# Cargar predicciones desde JSON
with open("predictions_yolov3-tiny.json", "r") as f:
    predictions_data = json.load(f)

all_pred_boxes, all_gt_boxes = [], []

# Procesar los datos para el cálculo de métricas
for entry in predictions_data:
    # Asegurarse de que las cajas de predicción estén bien formateadas y convertidas a float
    if entry["pred_boxes"]:
        pred_boxes = torch.tensor(np.array([[float(x) for x in box] for box in entry["pred_boxes"]]), dtype=torch.float32)
    else:
        pred_boxes = torch.empty((0, 4), dtype=torch.float32)

    # Asegurarse de que las puntuaciones de predicción estén bien formateadas
    if entry["pred_scores"]:
        pred_scores = torch.tensor(entry["pred_scores"], dtype=torch.float32)
    else:
        pred_scores = torch.empty((0,), dtype=torch.float32)
    
    # Asegurarse de que las etiquetas de predicción estén bien formateadas
    if entry["pred_labels"]:
        pred_labels = torch.tensor(entry["pred_labels"], dtype=torch.int64)
    else:
        pred_labels = torch.empty((0,), dtype=torch.int64)

    # Asegurarse de que las cajas de verdad de terreno estén bien formateadas y convertidas a float
    if entry["gt_boxes"]:
        gt_boxes = torch.tensor(np.array([[float(x) for x in box] for box in entry["gt_boxes"]]), dtype=torch.float32)
    else:
        gt_boxes = torch.empty((0, 4), dtype=torch.float32)

    # Asegurarse de que las etiquetas de verdad de terreno estén bien formateadas
    if entry["gt_labels"]:
        gt_labels = torch.tensor(entry["gt_labels"], dtype=torch.int64)
    else:
        gt_labels = torch.empty((0,), dtype=torch.int64)

    # Imprimir los valores para depuración
    # print(f"Frame: {entry['frame']}")
    # print(f"Pred Boxes: {pred_boxes}")
    # print(f"GT Boxes: {gt_boxes}")

    # Añadir las predicciones y las cajas de verdad de terreno a las listas
    all_pred_boxes.append({"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels})
    all_gt_boxes.append({"boxes": gt_boxes, "labels": gt_labels})

# Verificar que hay datos antes de calcular métricas
if len(all_pred_boxes) == 0 or len(all_gt_boxes) == 0:
    print("⚠️ Error: No hay datos en las predicciones o en los ground truths.")
else:
    # Calcular métricas
    video_ap = metrics.compute_video_average_precision(all_pred_boxes, all_gt_boxes, iou_threshold=0.5)
    print(f"mAP (AP para 'car'): {video_ap:.4f}")

    # with open("map_results.txt", "a") as f:
    #     f.write(f"mAP={video_ap:.4f}\n")

    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    metric.update(all_pred_boxes, all_gt_boxes)
    video_metrics = metric.compute()
    
    # Verificar si `map_per_class` es un tensor válido
    if isinstance(video_metrics["map_per_class"], torch.Tensor):
        if video_metrics["map_per_class"].ndim == 0:  # Si es un escalar
            print("⚠️ No hay detecciones válidas, 'map_per_class' es un escalar.")
            car_ap, bike_ap = None, None
        else:
            car_ap = float(video_metrics["map_per_class"][0]) if len(video_metrics["map_per_class"]) > 0 else None
            bike_ap = float(video_metrics["map_per_class"][1]) if len(video_metrics["map_per_class"]) > 1 else None
    else:
        car_ap, bike_ap = None, None

    metrics_data = {
        "mAP": float(video_metrics["map"]),
        "mAP50": float(video_metrics["map_50"]),
        "mAP75": float(video_metrics["map_75"]),
        "mAP_car": car_ap,
        "mAP_bike": bike_ap
    }
    
    print(metrics_data)