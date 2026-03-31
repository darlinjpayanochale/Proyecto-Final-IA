from ultralytics import YOLO
import cv2
import pygame

# Cargar modelo (puedes cambiar a yolov8m.pt si tu PC aguanta)
model = YOLO("yolov8s.pt")

# Cámara
cap = cv2.VideoCapture(0)

# Sonido
pygame.mixer.init()
pygame.mixer.music.load("assets/alarma.mp3")

# Objetos peligrosos
dangerous_objects = ["knife", "scissors", "fork"]

frame_count = 0
alarm_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Procesar cada 3 frames para mejorar rendimiento
    if frame_count % 3 == 0:

        results = model(frame, imgsz=320, conf=0.25)

        detected_danger = False

        for r in results:
            for box in r.boxes:

                conf = float(box.conf[0])

                # Filtrar detecciones débiles
                if conf < 0.5:
                    continue

                cls = int(box.cls[0])
                label = model.names[cls]

                print(f"Detectado: {label} ({conf:.2f})")

                # Ignorar falsos positivos comunes
                if label == "toothbrush":
                    continue

                # Detectar peligro
                if label in dangerous_objects:
                    detected_danger = True

        annotated_frame = results[0].plot()

        # Alarma
        if detected_danger:
            if not alarm_playing:
                pygame.mixer.music.play(-1)
                alarm_playing = True

            cv2.putText(
                annotated_frame,
                "PELIGRO DETECTADO",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        else:
            if alarm_playing:
                pygame.mixer.music.stop()
                alarm_playing = False

        cv2.imshow("Detector IA", annotated_frame)

    # Salir con Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar todo
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()