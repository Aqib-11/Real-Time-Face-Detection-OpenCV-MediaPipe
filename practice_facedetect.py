import cv2
import mediapipe as mp
import cv2 as cv

face_detection = mp.solutions.face_detection
face_detect = face_detection.FaceDetection(min_detection_confidence=0.8)
draw = mp.solutions.drawing_utils

webcam = cv2.VideoCapture(0)
while True:
    ret, frame = webcam.read()
    if not ret:
        break
    image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_detect.process(image_rgb)
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h,w ,_ = frame.shape
            x,y = int(bbox.xmin * w), int(bbox.ymin * h)
            w_box, h_box = int(bbox.width * w), int(bbox.height * h)
            cv.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 0, 255), 5)

        cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()