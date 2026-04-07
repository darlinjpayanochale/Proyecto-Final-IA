# main.py

# Importamos las librerías necesarias
from ultralytics import YOLO  # Modelo de detección de objetos (YOLOv8)
import cv2                    # OpenCV para procesamiento de imágenes
import os                     # Manejo de carpetas
import time                   # Para timestamps
import gradio as gr           # Interfaz web interactiva

# Cargamos el modelo preentrenado
model = YOLO("yolov8x.pt")

# Lista de objetos que consideramos peligrosos
dangerous_objects = ["knife", "scissors", "fork"]

# Mensajes personalizados según el objeto detectado
danger_messages = {
    "knife": "Alto: puede causar heridas profundas y corte.",
    "scissors": "Alto: puede causar heridas profundas y corte.",
    "fork": "Medio: puede causar lesiones superficiales"
}

# Creamos la carpeta "capturas" si no existe
os.makedirs("capturas", exist_ok=True)

# Función principal que procesa la imagen desde Gradio
def detectar_imagen_gradio(imagen):
    # Ejecutamos el modelo sobre la imagen con confianza mínima de 0.5
    results = model(imagen, conf=0.5)

    # Copiamos la imagen original para dibujar encima
    annotated_frame = imagen.copy()

    # Obtenemos dimensiones de la imagen
    h, w = annotated_frame.shape[:2]

    # Ajustamos el tamaño del texto dependiendo de la resolución
    font_scale_alert = max(0.5, h / 480)    # Texto "PELIGRO"
    thickness_alert = max(1, int(h / 300))  # Grosor del texto

    font_scale_estado = max(0.5, h / 480)   # Texto "Estado"
    thickness_estado = max(1, int(h / 300)) # Grosor del texto

    # Diccionario para contar objetos detectados
    detected_objects = {}

    # Recorremos los resultados del modelo
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])           # Clase detectada
            label = model.names[cls]        # Nombre de la clase

            # Solo trabajamos con objetos peligrosos
            if label in dangerous_objects:
                # Contamos cuántas veces aparece cada objeto
                detected_objects[label] = detected_objects.get(label, 0) + 1

                # Coordenadas de la caja
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Dibujamos el rectángulo rojo
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Escribimos "PELIGRO" encima del objeto
                cv2.putText(annotated_frame, "PELIGRO", (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale_alert, (0, 0, 255), thickness_alert)

    # Definimos el estado general según si hay objetos peligrosos o no
    if detected_objects:
        estado = "PELIGRO"
        color = (0, 0, 255)
    else:
        estado = "SEGURO"
        color = (0, 255, 0)

    # Guardamos la imagen procesada con un nombre único basado en el tiempo
    filename = f"capturas/resultado_{int(time.time())}.jpg"
    cv2.imwrite(filename, annotated_frame)

    # Generamos el texto de peligrosidad según los objetos detectados
    peligrosidad_str = "\n".join(
        [danger_messages[obj] for obj in detected_objects]
    ) if detected_objects else "Ningún objeto peligroso detectado"

    # Convertimos la imagen a RGB para que Gradio la muestre bien
    return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), estado, peligrosidad_str


# Interfaz con Gradio
with gr.Blocks(css="""
.container {
    max-width: 900px;
    margin: auto;
}
""") as demo:

    # Título de la app
    gr.Markdown("## Detector de Objetos Peligrosos IA")

    # Pestaña para trabajar con imágenes
    with gr.Tab("Imagen"):
        imagen_input = gr.Image(
            type="numpy",
            label="Sube tu imagen o toca el icono de la cámara para usarla.",
            height=400
        )

        # Botón para ejecutar la detección
        detectar_btn = gr.Button("Detectar")

        # Salidas
        salida_img = gr.Image(label="Resultado", height=400)
        estado_txt = gr.Textbox(label="Estado")
        peligrosidad_txt = gr.Textbox(label="Grados de Peligrosidad")

        # Conectamos el botón con la función
        detectar_btn.click(
            detectar_imagen_gradio,
            inputs=imagen_input,
            outputs=[salida_img, estado_txt, peligrosidad_txt]
        )

# Lanzamos la app (y genera un link público)
demo.launch(share=True)