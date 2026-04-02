from ultralytics import YOLO
import cv2
import pygame
import os
import time

model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(0)

pygame.mixer.init()
pygame.mixer.music.load("assets/alarma.mp3")

dangerous_objects = ["knife", "scissors", "fork"]

if not os.path.exists("capturas"):
    os.makedirs("capturas")

frame_count = 0
alarm_playing = False
last_capture_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 3 == 0:

        results = model(frame, imgsz=320, conf=0.25)

        detected_danger = False

        for r in results:
            for box in r.boxes:

                conf = float(box.conf[0])
                if conf < 0.5:
                    continue

                cls = int(box.cls[0])
                label = model.names[cls]

                print(f"Detectado: {label} ({conf:.2f})")

                if label == "toothbrush":
                    continue

                if label in dangerous_objects:
                    detected_danger = True

        annotated_frame = results[0].plot()

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

            current_time = time.time()

            if current_time - last_capture_time > 2:
                filename = f"capturas/peligro_{int(current_time)}.jpg"
                cv2.imwrite(filename, frame)
                print("Imagen guardada:", filename)
                last_capture_time = current_time

        else:
            if alarm_playing:
                pygame.mixer.music.stop()
                alarm_playing = False

        cv2.imshow("Detector IA", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()