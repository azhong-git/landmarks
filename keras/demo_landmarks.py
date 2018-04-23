import cv2
import time
import numpy as np
from MTCNN import create_Kao_Onet

detection_model_path = 'models/haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(detection_model_path)

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

Onet = create_Kao_Onet('models/48net.h5')

while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    if len(faces) > 0:
        for face_coordinates in faces:
            batch_in = []
            x, y, w, h = face_coordinates
            face_crop = bgr_image[y:(y+h), x:(x+w)]
            face_crop = cv2.resize(face_crop, (48, 48))
            face_crop = (face_crop - 127.5) / 127.5
            batch_in.append(cv2.resize(face_crop, (48, 48)))
            cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (255, 255, 255), 2)

            batch_in = np.array(batch_in)
            start = time.time()
            output = Onet.predict(batch_in)
            print(output)
            prob = output[0][0][1]
            pts = output[2][0]
            print('Onet took {} ms, face prob = {}'.format((time.time()-start)*1e3, prob))
            for i in range(0, 5):
                cv2.circle(bgr_image, (int(pts[i]*w+x), int(pts[i+5]*h+y)), 2, (0, 255, 0))


    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
