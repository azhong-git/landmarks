import cv2
import time
import numpy as np
import tensorflow as tf

detection_model_path = 'models/haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(detection_model_path)
landmark_model_path = 'models/mtcnn_onet.h5.pb'

def draw_pose(image, rotation_matrix, where_to_draw):
    direction = np.dot(rotation_matrix, np.array([[1], [0], [0]]))
    # print('dirX is {}'.format(direction[0][0]))
    # if (direction[0][0] < 0):
    #     return

    for i in range(3):
        axis = np.array([[0], [0], [0]])
        axis[i][0] = 50
        direction = np.dot(rotation_matrix, axis)
        color = [0, 0, 0]
        color[i] = 255
        color = tuple(color)
        cv2.line(image, where_to_draw, (int(where_to_draw[0]+direction[0][0]), int(where_to_draw[1]+direction[1][0])), color, 3)

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

# for head pose solving
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    # (0.0, -330.0, -65.0),        # Chin
    (-200.0, 170.0, -135.0),     # Left eye
    (200.0, 170.0, -135.0),      # Right eye
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

last_frame_valid = False

while True:
    bgr_image = video_capture.read()[1]
    height = bgr_image.shape[0]
    width = bgr_image.shape[1]

    # for head pose solving
    focal_length = width
    center = (width/2.0, height/2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype = "double"
    )
    dist_coeffs = np.zeros((4,1))
    # end of head pose solving

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
    cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
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

    #2D image points. If you change the image, you need to change vector
    chin_x = pts[2] + (pts[4] - pts[2]) + (pts[3] - pts[2])
    chin_y = pts[7] + (pts[9] - pts[7]) + (pts[8] - pts[7])
    image_points = np.array([
        (pts[2]*w+x, pts[7]*h+y),
        # (chin_x*w+x, chin_y*h+y),
        (pts[0]*w+x, pts[5]*h+y),
        (pts[1]*w+x, pts[6]*h+y),
        (pts[3]*w+x, pts[8]*h+y),
        (pts[4]*w+x, pts[9]*h+y)
    ], dtype="double")

    if not last_frame_valid:
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    else:
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, last_rotation_vector, last_translation_vector, True)

    # print(rotation_vector, translation_vector)

    if success:
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        direction = np.dot(rotation_matrix, np.array([[1], [0], [0]]))
        if direction[0][0] >= 0:
            print("rotation vector", rotation_vector)
            norm = np.linalg.norm(rotation_vector)
            rotation_vector_normalized = rotation_vector / norm
            print("norm", norm)
            print("rotation vector norm", rotation_vector_normalized)
            last_frame_valid = True
            last_rotation_vector = rotation_vector.copy()
            last_translation_vector = translation_vector.copy()
            draw_pose(bgr_image, rotation_matrix, (50, 50))
        else:
            last_frame_valid = False
    else:
        last_frame_valid = False

    for i in range(0, 5):
        cv2.circle(bgr_image, (int(pts[i]*w+x), int(pts[i+5]*h+y)), 2, (0, 255, 0))
    print('One frame took {} ms'.format((time.time()-start_frame)*1e3))


    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
