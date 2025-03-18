import cv2
import os
import glob

# Configuraciones
frames_folder = "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/Week4/video_frames/s03_c010_global"  # Carpeta donde se encuentran los frames
output_video = "Week4/results/s03_c010_global.mp4"  # Nombre del video de salida
fps = 5                        # Fotogramas por segundo deseados para el video

# Obtener la lista de archivos de imagen (se asume extensión .jpg)
frame_paths = sorted(glob.glob(os.path.join(frames_folder, "*.png")))
if not frame_paths:
    print("No se encontraron frames en la carpeta especificada.")
    exit(1)

# Leer el primer frame para obtener dimensiones
first_frame = cv2.imread(frame_paths[0])
if first_frame is None:
    print("No se pudo leer el primer frame.")
    exit(1)
height, width, channels = first_frame.shape

# Configurar VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # También puedes probar "XVID" o "MJPG"
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print(f"Creando video: {output_video}")
for frame_path in frame_paths:
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Advertencia: No se pudo leer el frame {frame_path}. Se omitirá.")
        continue
    video_writer.write(frame)
    print(f"Procesado: {frame_path}")

video_writer.release()
print(f"Video creado exitosamente: {output_video}")
