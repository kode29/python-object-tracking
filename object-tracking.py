import cv2
import numpy as np
from object_detection import ObjectDetection

cap = cv2.VideoCapture("assets/video/traffic.mp4")

# Initialize object detection
od = ObjectDetection()
while True:
    ret, frame = cap.read()

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        print(box)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()