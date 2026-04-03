from ultralytics import YOLO
import cv2
import pygame
import os
import time
import numpy as np

def detector_camara():
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

    danger_count = 0
    prev_detected = False
    fps = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

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

                    if label == "toothbrush":
                        continue

                    if label in dangerous_objects:
                        detected_danger = True

            annotated_frame = results[0].plot()

            if detected_danger:
                if not prev_detected:
                    danger_count += 1
                prev_detected = True

                if not alarm_playing:
                    pygame.mixer.music.play(-1)
                    alarm_playing = True

                current_time = time.time()

                if current_time - last_capture_time > 2:
                    filename = f"capturas/peligro_{int(current_time)}.jpg"
                    cv2.imwrite(filename, frame)
                    last_capture_time = current_time

            else:
                prev_detected = False
                if alarm_playing:
                    pygame.mixer.music.stop()
                    alarm_playing = False

            if detected_danger:
                color = (0, 0, 255)
                estado = "PELIGRO"
            else:
                color = (0, 255, 0)
                estado = "SEGURO"

            cv2.putText(annotated_frame, f"Estado: {estado}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.putText(annotated_frame, f"Eventos: {danger_count}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            cv2.putText(annotated_frame, f"ESC: Salir", (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow("Detector IA", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()


def menu_visual():
    while True:
        screen = np.zeros((400, 600, 3), dtype=np.uint8)

        cv2.putText(screen, "DETECTOR IA", (150, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        cv2.putText(screen, "1 - Camara en tiempo real", (120, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.putText(screen, "2 - Modo imagen (proximamente)", (120, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.putText(screen, "ESC - Salir", (120, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Menu", screen)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):
            cv2.destroyAllWindows()
            detector_camara()
        elif key == 27:
            break

    cv2.destroyAllWindows()


menu_visual()