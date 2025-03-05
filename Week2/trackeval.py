import trackeval 
import pandas as pd

# Define the tracking methods to evaluate
tracking_methods = {
    "Tracking Overlap ( task 2.1)": "ruta_a_tus_resultados_2.1.txt",
    "Tracking Kalman Filter ( task 2.2)": "ruta_a_tus_resultados_2.2.txt"
}

ground_truth_path = "ruta_a_ground_truth.txt"

# configuration of assessment paremeters
eval_config = {
    "USE_PARALLEL": False,
    "NUM_PARALLEL_CORES": 4,
    "PRINT_RESULTS": True,
    "OUTPUT_SUMMARY": False,
    "OUTPUT_EMPTY_CLASSES": False,
    "METRICS": ["HOTA", "IDF1"]
}

evaluator = trackeval.Evaluator(eval_config)

results_dict = {}

# Evaluar cada método de tracking
for method, result_path in tracking_methods.items():
    metrics = evaluator.evaluate(ground_truth_path, result_path)
    
    idf1_score = metrics["IDF1"] if "IDF1" in metrics else None
    hota_score = metrics["HOTA"] if "HOTA" in metrics else None
    
    # save the results
    results_dict[method] = {"IDF1": idf1_score, "HOTA": hota_score}

# Create a dataframe with the results
df_results = pd.DataFrame.from_dict(results_dict, orient="index")

print(df_results)

# Save the table
df_results.to_csv("tracking_evaluation_results.csv", index=True)
print("Tabla guardada como 'tracking_evaluation_results.csv'")