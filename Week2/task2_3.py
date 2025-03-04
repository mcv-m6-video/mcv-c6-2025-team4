import trackeval
import pandas as pd

# Definir los métodos de tracking a evaluar
tracking_methods = {
    "Tracking Overlap (task 2.1)": "tracker2_1",
    "Tracking Kalman Filter (task 2.2)": "tracker2_z1"
}

# Configuración de TrackEval
eval_config = {
    "USE_PARALLEL": False,
    "NUM_PARALLEL_CORES": 4,
    "PRINT_RESULTS": True,
    "OUTPUT_SUMMARY": False,
    "OUTPUT_EMPTY_CLASSES": False,
    "METRICS": ["HOTA", "IDF1"],
}

# Configuración del dataset (MOTChallenge)
dataset_config = {
    "GT_FOLDER": "./data/",  # Carpeta donde está ground_truth.txt
    "TRACKERS_FOLDER": "./TrackEval/trackers/",
    "TRACKER_SUB_FOLDER": "data",  
    "OUTPUT_FOLDER": "./TrackEval/results/",
    "TRACKERS_TO_EVAL": list(tracking_methods.values()),
    "BENCHMARK": "MOT17",
    "SPLIT_TO_EVAL": "train"
}

# Inicializar evaluador y dataset
evaluator = trackeval.Evaluator(eval_config)
dataset_list = [trackeval.datasets.MOTSChallenge(dataset_config)]
metrics_list = [trackeval.metrics.HOTA()] #, trackeval.metrics.IDF1()]

# Evaluar y almacenar resultados
results_dict = {}
raw_results = evaluator.evaluate(dataset_list, metrics_list)

# Extraer y formatear los resultados
for method, tracker_name in tracking_methods.items():
    if tracker_name in raw_results["MotChallenge"]["train"]:
        method_results = raw_results["MotChallenge"]["train"][tracker_name]
        idf1_score = method_results["IDF1"]["OVERALL"] if "IDF1" in method_results else None
        hota_score = method_results["HOTA"]["OVERALL"] if "HOTA" in method_results else None
        results_dict[method] = {"IDF1": idf1_score, "HOTA": hota_score}

# Convertir a DataFrame y guardar resultados
df_results = pd.DataFrame.from_dict(results_dict, orient="index")
print(df_results)
df_results.to_csv("tracking_evaluation_results.csv", index=True)
print("Tabla guardada como 'tracking_evaluation_results.csv'")
