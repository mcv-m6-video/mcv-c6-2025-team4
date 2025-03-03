import cv2
import numpy as np
import tensorflow as tf
import torch
from torchvision import models, transforms
from src import metrics
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights,retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights,fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights,ssd300_vgg16, SSD300_VGG16_Weights,ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Función para hacer el seguimiento de objetos entre dos cuadros
def track_objects(frame_n_boxes, frame_n1_boxes, threshold=0.4):
    tracked_boxes = []
    track_id = 1
    frame_n_ids = {}
    
    # Asignar ID de seguimiento a los objetos en el primer cuadro
    for i, box in enumerate(frame_n_boxes):
        frame_n_ids[i] = track_id
        track_id += 1

    # Comparar las cajas en el siguiente cuadro
    for box_n1 in frame_n1_boxes:
        best_iou = 0
        best_match = None
        best_track_id = None

        for i, box_n in enumerate(frame_n_boxes):  # Agregar índice 'i'
            iou = metrics.compute_iou(box_n, box_n1)
            if iou > best_iou:
                best_iou = iou
                best_match = box_n
                best_track_id = frame_n_ids[i]

        if best_iou >= threshold:
            tracked_boxes.append((box_n1, best_track_id))
        else:
            tracked_boxes.append((box_n1, track_id))
            track_id += 1

    return tracked_boxes

# Función para realizar la detección de objetos dependiendo del modelo cargado
def detect_objects(frame, model, framework='tensorflow'):
    """
    Detecta objetos usando el modelo proporcionado.
    
    Parameters:
    - frame: El cuadro del video.
    - model: El modelo preentrenado para la detección de objetos.
    - framework: 'tensorflow', 'torch', o 'opencv'.
    
    Returns:
    - boxes: Lista de cajas delimitadoras detectadas.
    """
    if framework == 'tensorflow':
        # Preprocesar la imagen para el modelo de TensorFlow
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir a RGB
        input_frame = np.expand_dims(input_frame, axis=0)
        input_frame = tf.image.resize(input_frame, (800, 800))  # Redimensionar si es necesario
        
        # Realizar la predicción
        detections = model(input_frame)
        boxes = detections['detection_boxes'].numpy()[0]
        return boxes

    elif framework == 'torch':
        # Preprocesar la imagen para el modelo de PyTorch
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
        ])
        input_frame = transform(frame).unsqueeze(0)  # Añadir batch dimension
        model.eval()  # Poner el modelo en modo evaluación

        # Realizar la predicción
        with torch.no_grad():
            predictions = model(input_frame)
        
        # Obtener las cajas delimitadoras de las predicciones
        boxes = predictions[0]['boxes'].cpu().numpy()
        return boxes

    elif framework == 'opencv':
        # Preprocesar la imagen para el modelo Mask R-CNN de OpenCV
        blob = cv2.dnn.blobFromImage(frame, 1.0, (800, 800), (0, 0, 0), swapRB=True, crop=False)
        model.setInput(blob)
        output = model.forward(["detection_out_final", "detection_masks"])

        # Extraer las cajas delimitadoras
        boxes = []
        for i in range(output[0].shape[2]):
            confidence = output[0][0, 0, i, 2]
            if confidence > 0.5:
                x1 = int(output[0][0, 0, i, 3] * frame.shape[1])
                y1 = int(output[0][0, 0, i, 4] * frame.shape[0])
                x2 = int(output[0][0, 0, i, 5] * frame.shape[1])
                y2 = int(output[0][0, 0, i, 6] * frame.shape[0])
                boxes.append([x1, y1, x2, y2])

        return boxes

    else:
        raise ValueError(f"Framework '{framework}' no soportado.")

# Función para cargar un modelo preentrenado basado en el framework
def load_model(model_path, framework='tensorflow'):
    if framework == 'tensorflow':
        model = tf.saved_model.load(model_path)
        return model
    elif framework == 'torch':
        model = torch.load(model_path)
        model.eval()  # Poner el modelo en modo evaluación
        return model
    elif framework == 'opencv':
        net = cv2.dnn.readNetFromTensorflow(model_path)
        return net
    else:
        raise ValueError(f"Framework '{framework}' no soportado.")

# Función para generar un color aleatorio basado en el ID de seguimiento
def generate_color(track_id):
    np.random.seed(track_id)  # Semilla basada en track_id para obtener colores consistentes
    color = np.random.randint(0, 255, 3).tolist()  # Generar un color aleatorio (BGR)
    return tuple(color)

# Video de entrada
video_path = "./data/AICity_data/train/S03/c010/vdo.avi"  # Ruta al video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()


# Obtener propiedades del video
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Crear el objeto VideoWriter para guardar el video
output_path = './output_videos/task2_1.mp4'  # Ruta de salida para el video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para guardar el video en formato .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Cargar el modelo que se va a utilizar
# model_path = 'path_to_your_model'  # Ruta al modelo
framework = 'torch'  # Cambiar a 'torch' o 'opencv' según el modelo que uses
# model = load_model(model_path, framework)
weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
model = retinanet_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.to(device)
next(model.parameters()).device
model.eval()

# Leer el primer cuadro
ret, frame_n = cap.read()
frame_n_boxes = detect_objects(frame_n, model, framework)

frame_n1_boxes = []

# Leer los cuadros sucesivos y hacer el seguimiento
while True:
    ret, frame_n1 = cap.read()
    if not ret:
        break
    
    # Detectar objetos en el cuadro N+1
    frame_n1_boxes = detect_objects(frame_n1, model, framework)
    
    # Realizar el seguimiento de objetos
    tracked_objects = track_objects(frame_n_boxes, frame_n1_boxes, threshold=0.4)
    
    # Dibujar las cajas de los objetos y los IDs en el cuadro
    for box, track_id in tracked_objects:
        x1, y1, x2, y2 = map(int, box)
        
        # Generar un color único para cada track_id
        color = generate_color(track_id)
        
        # Dibujar el rectángulo con el color generado
        if x1 is None or y1 is None or x2 is None or y2 is None:
            print(f"Error: Coordenadas no válidas ({x1}, {y1}, {x2}, {y2})")
        else:
            cv2.rectangle(frame_n1, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Colocar el texto con el track_id encima del rectángulo
        cv2.putText(frame_n1, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Escribir el cuadro procesado en el archivo de salida
    out.write(frame_n1)
    
    # Actualizar el cuadro anterior para la siguiente iteración
    frame_n_boxes = frame_n1_boxes

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
