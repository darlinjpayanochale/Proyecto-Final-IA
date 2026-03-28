from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Lista de objetos peligrosos
dangerous_objects = ["knife", "scissors"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    detected_danger = False

    # Revisar detecciones
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in dangerous_objects:
                detected_danger = True

    annotated_frame = results[0].plot()

    # Mostrar alerta
    if detected_danger:
        cv2.putText(
            annotated_frame,
            "PELIGRO DETECTADO",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # Mostrar ventana (SIEMPRE)
    cv2.imshow("Detector IA", annotated_frame)

    # Salir con Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()