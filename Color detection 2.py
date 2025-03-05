import cv2
import numpy as np

color_boundaries = {
    'red': ([0, 120, 70], [10, 255, 255]),
    'green': ([40, 40, 40], [90, 255, 255]),
    'blue': ([100, 150, 0], [140, 255, 255]),
    'yellow': ([20, 100, 100], [30, 255, 255]),
    'black': ([0, 0, 0], [180, 255, 30]),
    'white': ([0, 0, 200], [180, 20, 255]),
    'grey': ([0, 0, 40], [180, 20, 200]),
}

def detect_color(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_color = 'None'
    largest_area = 0
    object_mask = None

    for color, (lower, upper) in color_boundaries.items():
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound) 

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > largest_area:  
                largest_area = area
                detected_color = color
                object_mask = contour

    return detected_color, object_mask

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
else:
    print("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        detected_color, object_contour = detect_color(frame)

        if object_contour is not None:
            x, y, w, h = cv2.boundingRect(object_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Color: {detected_color}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Real-Time Color Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
