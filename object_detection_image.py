import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import ObjectDetectorResult
from mediapipe.tasks.python.vision import ObjectDetectorOptions
from mediapipe.tasks.python.vision import ObjectDetector, ObjectDetectorOptions

import cv2

#Especificar las opciones de configuracion
options = vision.ObjectDetectorOptions(
    base_options = BaseOptions(model_asset_path='efficientdet_lite0.tflite'), #Ruta del modelo
    max_results=5,
    score_threshold =0.2,
    running_mode=vision.RunningMode.IMAGE)

detector = vision.ObjectDetector.create_from_options(options)

#Leer imagen de entrada
image = cv2.imread("./Data/image_001.jpg")
image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

#Detectar objetos sobre la imagen
detection_result = detector.detect(image_rgb)

for detection in detection_result.detections:
    print(detection)

cv2.imshow("Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()