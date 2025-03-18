import cv2
import os
import random


def read_detections(file_path):
    """
    Lee el archivo de detecciones y lo organiza en un diccionario.
    Se asume que cada línea tiene el formato:
       frame, track_id, x, y, w, h, ...
    Retorna un diccionario donde la llave es el número de frame y el valor es una lista
    de detecciones (cada una es un diccionario con 'track_id' y 'bbox').
    """
    detections = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            # Convertir cada parte a entero (se puede adaptar si hay decimales)
            parts = [int(x.strip()) for x in line.split(',')]
            frame_num = parts[0]
            track_id = parts[1]
            x = parts[2]
            y = parts[3]
            w = parts[4]
            h = parts[5]
            if frame_num not in detections:
                detections[frame_num] = []
            detections[frame_num].append({'track_id': track_id, 'bbox': (x, y, w, h)})
    return detections


def get_color(track_id, color_map):
    """
    Devuelve un color (B, G, R) único para cada track_id.
    Si el track_id no está en el diccionario color_map, se genera un color aleatorio.
    """
    if track_id not in color_map:
        color_map[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color_map[track_id]


def main(video_path, detections_file, output_folder, target_fps=10):
    # Crear carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Leer las detecciones del archivo
    detections = read_detections(detections_file)

    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error abriendo el archivo de video.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    # Calcular cada cuántos frames se debe extraer para lograr target_fps
    sample_interval = int(round(video_fps / target_fps))
    if sample_interval < 1:
        sample_interval = 1
    print(f"Video FPS: {video_fps:.2f}. Se extraerán 1 de cada {sample_interval} frames para obtener {target_fps} fps.")

    frame_index = 0
    output_frame_count = 0
    color_map = {}  # diccionario para asignar un color único a cada track_id

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extraer frame si es el indicado para 10 fps
        if frame_index % sample_interval == 0:
            # Dependiendo de cómo se indexen los frames en el archivo de detecciones, puede que
            # sea necesario usar frame_index o frame_index+1. En este ejemplo se prueba ambas.
            frame_dets = detections.get(frame_index, [])
            if not frame_dets:
                frame_dets = detections.get(frame_index + 1, [])

            # Dibujar cada detección sobre el frame
            for det in frame_dets:
                track_id = det['track_id']
                x, y, w, h = det['bbox']
                color = get_color(track_id, color_map)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, str(track_id), (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Guardar el frame anotado en la carpeta de salida
            output_filename = os.path.join(output_folder, f"frame_{output_frame_count:05d}.png")
            cv2.imwrite(output_filename, frame)
            output_frame_count += 1

        frame_index += 1

    cap.release()
    print("Proceso finalizado. Se han guardado los frames en la carpeta:", output_folder)


if __name__ == "__main__":
    video_path = "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/data/aic19-track1-mtmc-train/train/S03/c010/vdo.avi"  # Ruta al video .avi
    detections_file = "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/Global_Tracking_Miren_S03/c010_global.txt"  # Ruta al archivo .txt con las detecciones
    output_folder = "/home/toukapy/Dokumentuak/Master CV/C6/mcv-c6-2025-team4/Week4/video_frames/s03_c010_global"  # Carpeta de salida para los frames anotados
    main(video_path, detections_file, output_folder, target_fps=10)