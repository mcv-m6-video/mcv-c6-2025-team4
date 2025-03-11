import matplotlib.pyplot as plt
import numpy as np

# Datos
methods = ["Farneback", "PyFlow", "RAFT", "FlowFormer (kitti.pth)", "FlowFormer (sintel.pth)"]
time_taken = [0.37, 15.93, 20.29, 0.55, 0.38]  # DeepMind no tiene tiempo especificado
msen = [4.2130, 0.9364, 6.1995, 2.1554, 1.3410]
pepn = [28.4367, 7.4293, 49.41, 20.7760, 11.4779]

x = np.arange(len(methods))  # Posiciones en el eje x
width = 0.3  # Ancho de las barras

# Gráfico de MSEN y PEPN con ejes separados
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()  # Crear un segundo eje y

ax1.bar(x - width/2, msen, width, label='MSEN', color='royalblue')
ax2.bar(x + width/2, pepn, width, label='PEPN', color='lightcoral')

ax1.set_xlabel("Métodos")
ax1.set_ylabel("MSEN", color='royalblue')
ax2.set_ylabel("PEPN", color='lightcoral')
ax1.set_xticks(x)
ax1.set_xticklabels(methods)

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title("Comparación de MSEN y PEPN")
plt.show()

# Gráfico de Tiempo de Ejecución
fig, ax3 = plt.subplots(figsize=(10, 6))
time_taken_clean = [t if t is not None else 0 for t in time_taken]  # Reemplazar None por 0
ax3.bar(x, time_taken_clean, width, label='Tiempo de ejecución (s)', color='darkorange')
ax3.set_xlabel("Métodos")
ax3.set_ylabel("Tiempo (s)")
ax3.set_xticks(x)
ax3.set_xticklabels(methods)
ax3.legend()
plt.title("Tiempo de Ejecución por Método")
plt.show()