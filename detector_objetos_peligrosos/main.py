# main.py
from ultralytics import YOLO
import cv2
import os
import time
import numpy as np
import gradio as gr

model = YOLO("yolov8m.pt")

# Objetos considerados peligrosos
dangerous_objects = ["knife", "scissors", "fork"]

os.makedirs("capturas", exist_ok=True)

def detectar_imagen_gradio(imagen):
    """
    Detecta objetos peligrosos en una imagen y devuelve:
    - Imagen anotada
    - Estado (PELIGRO / SEGURO)
    """
    results = model(imagen, imgsz=640, conf=0.25)
    
    detected_objects = {}
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            cls = int(box.cls[0])
            label = model.names[cls]
            if label == "toothbrush":
                continue
            if label in dangerous_objects:
                detected_objects[label] = detected_objects.get(label, 0) + 1

    annotated_frame = results[0].plot()
    
    total_danger = sum(detected_objects.values())
    if total_danger == 0:
        estado = "SEGURO"
        color = (0, 255, 0)
    else:
        estado = "PELIGRO"
        color = (0, 0, 255)
    
    cv2.putText(annotated_frame, f"Estado: {estado}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Guardar la captura
    filename = f"capturas/resultado_{int(time.time())}.jpg"
    cv2.imwrite(filename, annotated_frame)

    
    objetos_str = ", ".join([f"{k}({v})" for k,v in detected_objects.items()]) or "Ninguno"

    return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), estado, objetos_str

with gr.Blocks() as demo:
    gr.Markdown("## Detector de Objetos Peligrosos IA")

    with gr.Tab("Imagen"):
        imagen_input = gr.Image(type="numpy", label="Sube tu imagen")
        detectar_btn = gr.Button("Detectar")
        salida_img = gr.Image(label="Resultado")
        estado_txt = gr.Textbox(label="Estado")
        objetos_txt = gr.Textbox(label="Objetos detectados")

        detectar_btn.click(
            detectar_imagen_gradio,
            inputs=imagen_input,
            outputs=[salida_img, estado_txt, objetos_txt]
        )

demo.launch(share=True)