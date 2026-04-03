from ultralytics import YOLO
from tkinter import Tk, filedialog
import cv2
import pygame
import os
import time
import numpy as np

pygame.mixer.init()
click_sound = pygame.mixer.Sound("assets/click.mp3")

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
    click = False
    mx, my = 0, 0

    def mouse_event(event, x, y, flags, param):
        nonlocal mx, my, click
        mx, my = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            click = True

    cv2.namedWindow("Menu")
    cv2.setMouseCallback("Menu", mouse_event)

    while True:
        screen = np.zeros((560, 900, 3), dtype=np.uint8)

        for i in range(screen.shape[0]):
            color = int(20 + (i / screen.shape[0]) * 60)
            screen[i, :] = (color, color, color + 20)

        cv2.putText(screen, "DETECTOR IA", (260, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)

        cv2.putText(screen, "Sistema de deteccion de objetos peligrosos.", (280, 140),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        botones = [
            ("CAMARA", (300, 220, 300, 60)),
            ("IMAGEN", (300, 310, 300, 60)),
            ("SALIR", (300, 400, 300, 60))
        ]
        for text, (x, y, w, h) in botones:
            hover = x < mx < x+w and y < my < y+h

            color = (50, 50, 50) if not hover else (90, 90, 90)

            cv2.rectangle(screen, (x, y), (x+w, y+h), color, -1)

            cv2.rectangle(screen, (x, y), (x+w, y+h), (120, 120, 120), 2)

            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2

            cv2.putText(screen, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if hover and click:
                click = False
                click_sound.play()
                cv2.destroyAllWindows()

                if text == "CAMARA":
                    detector_camara()
                elif text == "IMAGEN":
                    detector_imagen()
                elif text == "SALIR":
                    return

                cv2.namedWindow("Menu")
                cv2.setMouseCallback("Menu", mouse_event)

        cv2.imshow("Menu", screen)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()


def detector_imagen():
    model = YOLO("yolov8m.pt")

    root = Tk()
    root.withdraw()
    ruta = filedialog.askopenfilename()

    if not os.path.exists(ruta):
        print("Imagen no encontrada")
        return

    frame = cv2.imread(ruta)

    pygame.mixer.init()
    pygame.mixer.music.load("assets/alarma.mp3")

    dangerous_objects = ["knife", "scissors", "fork"]

    results = model(frame, imgsz=640, conf=0.25)

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
        pygame.mixer.music.play()
        estado = "PELIGRO"
        color = (0, 0, 255)
    else:
        estado = "SEGURO"
        color = (0, 255, 0)

    cv2.putText(annotated_frame, f"Estado: {estado}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if not os.path.exists("capturas"):
        os.makedirs("capturas")

    filename = f"capturas/resultado_{int(time.time())}.jpg"
    cv2.imwrite(filename, annotated_frame)
    print("Resultado guardado en:", filename)

    while True:
        cv2.imshow("Resultado Imagen", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()
    pygame.mixer.quit()

menu_visual()