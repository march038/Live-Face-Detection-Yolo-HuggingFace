# -*- coding: utf-8 -*-

import numpy as np
import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import PIL.Image
from IPython.display import display, clear_output
import os
from dotenv import load_dotenv

load_dotenv=(r"huggingface_access.env")
HF_TOKEN=os.getenv("HF_TOKEN")

# Funktion zur Erstellung eines Rechtecks um die erkannten Objekte
def draw_rectangles(frame, detections):
    for bbox, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        class_name = detections.data['class_name'][class_id]
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Download des Modells
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# Laden des Modells
model = YOLO(model_path)

# Starte die Videoaufnahme mit der Webcam
cap = cv2.VideoCapture(0)

while True:
    # Erfasse einen Frame von der Webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Führe die Inferenz mit dem Modell durch
    output = model(frame)
    results = Detections.from_ultralytics(output[0])

    # Zeichne Rechtecke um die erkannten Gesichter
    frame_with_rectangles = draw_rectangles(frame, results)

    # Zeige den Frame mit den umkreisten Objekten
    frame_rgb = cv2.cvtColor(frame_with_rectangles, cv2.COLOR_BGR2RGB)
    img_pil = PIL.Image.fromarray(frame_rgb)
    cv2.imshow('Frame', frame_with_rectangles)

    # Überprüfe, ob die 'q'-Taste gedrückt wurde, um die Schleife zu beenden
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Freigabe der Webcam und Schließen aller Fenster
cap.release()
cv2.destroyAllWindows()