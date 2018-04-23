import cv2
import time
import numpy as np
import tensorflow as tf

detection_model_path = 'models/haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(detection_model_path)
landmark_model_path = 'models/mtcnn_onet.h5.pb'


# load tensorflow model
graph = tf.Graph()
graph_def = tf.GraphDef()
with open(landmark_model_path, "rb") as f:
    graph_def.ParseFromString(f.read())
with graph.as_default():
    tf.import_graph_def(graph_def)
input_name = 'import/input_1'
prob_output_name = 'import/output_node0'
roi_output_name = 'import/output_node1'
landmark_output_name = 'import/output_node2'
input_operation = graph.get_operation_by_name(input_name)
prob_output_operation = graph.get_operation_by_name(prob_output_name)
roi_output_operation = graph.get_operation_by_name(roi_output_name)
landmark_output_operation = graph.get_operation_by_name(landmark_output_name)
input_shape = (int(input_operation.outputs[0].shape.dims[1]),
               int(input_operation.outputs[0].shape.dims[2]),
               1)
sess = tf.Session(graph = graph)

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
face_tracking = False

while True:
    bgr_image = video_capture.read()[1]
    height = bgr_image.shape[0]
    width = bgr_image.shape[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    start_frame = time.time()
    if face_tracking == False:
        faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
        if len(faces) > 0:
            face_coordinates = faces[0]
            face_tracking = True
        else:
            continue
    batch_in = []
    x, y, w, h = face_coordinates
    face_crop_original = bgr_image[y:(y+h), x:(x+w)]
    face_crop_original = cv2.resize(face_crop_original, (48, 48))
    face_crop = (face_crop_original - 127.5) / 127.5
    face_crop = np.expand_dims(face_crop, 0)
    cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
    print("cropping took {} ms".format((time.time()-start_frame)*1e3))
    start = time.time()
    landmark_prediction = sess.run([prob_output_operation.outputs[0],
                                    roi_output_operation.outputs[0],
                                    landmark_output_operation.outputs[0]],
                                   {
                                       input_operation.outputs[0]: face_crop
                                   })
    print("forward pass took {} ms".format((time.time()-start)*1e3))
    # print(landmark_prediction)
    prob = landmark_prediction[0][0][1]
    roi = landmark_prediction[1][0]
    pts = landmark_prediction[2][0]
    if prob < 0.7:
        face_tracking = False
        cv2.imshow('failed', face_crop_original)
        continue
    x1, y1, x2, y2 = [int(roi[0]*w+x), int(roi[1]*h+y), int(roi[2]*w+x+w), int(roi[3]*h+y+h)]
    cv2.rectangle(bgr_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if (x2 - x1) > (y2 - y1):
        diff = (x2-x1)-(y2-y1)
        y1 = max(int(y1-diff/2), 0)
        y2 = min(int(y2+diff/2), height-1)
    else:
        diff = (y2-y1)-(x2-x1)
        x1 = max(int(x1-diff/2), 0)
        x2 = min(int(x2+diff/2), width-1)
    face_coordinates = [x1, y1, x2-x1, y2-y1]

    for i in range(0, 5):
        cv2.circle(bgr_image, (int(pts[i]*w+x), int(pts[i+5]*h+y)), 2, (0, 255, 0))
    print('One frame took {} ms'.format((time.time()-start_frame)*1e3))


    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
