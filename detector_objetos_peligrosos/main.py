from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture(0)

dangerous_objects = ["knife", "scissors", "fork", "spoon"]

frame_count = 0

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
                cls = int(box.cls[0])
                label = model.names[cls]

                print("Detectado:", label)

                if label in dangerous_objects:
                    detected_danger = True

        annotated_frame = results[0].plot()

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

        cv2.imshow("Detector IA", annotated_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()