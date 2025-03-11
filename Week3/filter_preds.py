import pandas as pd

# Cargar el archivo de predicciones (pred_file) y el de ground truth (gt_file)
pred_file = pd.read_csv('Week3/outputs_track/output_mot_format_c001.txt', header=None)

# Filtrar para detectar el movimiento (basado en la diferencia de las coordenadas)
# El criterio será comparar la diferencia en las coordenadas (x, y). Si no cambia mucho, lo consideramos estacionado.

# Calculamos la diferencia en las coordenadas (en las columnas 3 y 4)
pred_file['delta_x'] = pred_file[3].diff().abs()
pred_file['delta_y'] = pred_file[4].diff().abs()

# Filtramos los coches que no tienen cambios significativos en sus coordenadas
# Vamos a suponer que los coches estacionados tienen una pequeña diferencia en las coordenadas.
# Puedes ajustar este umbral según tus necesidades.
umbral = 200  # Ajusta este valor según la variabilidad de tus coordenadas para detectar coches estacionados

# Filtramos donde la diferencia en x y y es baja
filtered_pred_file = pred_file[(pred_file['delta_x'] > umbral) | (pred_file['delta_y'] > umbral)]

# Eliminamos las columnas de diferencia que añadimos
filtered_pred_file = filtered_pred_file.drop(columns=['delta_x', 'delta_y'])

# Guardamos el archivo filtrado
filtered_pred_file.to_csv('Week3/outputs_track/output_mot_format_c001b.txt', header=False, index=False)

print("Archivo pred_file corregido guardado como 'filtered_pred_file.txt'")
