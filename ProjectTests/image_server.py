import io
import socket
import struct
import cv2
import numpy as np
import time
import dlib
from gaze_tracking import GazeTracking
from head_pose_estimation import *


def track_color(image, range1, range2):
    # change image to hsv and create mask
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    maskFinger = cv2.inRange(hsv_image, range1, range2)

    ut = [1000, 1000]

    # calc moments and call PID
    try:
        M = cv2.moments(maskFinger)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), 4, 3)
        ut = PID_controller((cX, cY), DESIRED_POS)
    except ZeroDivisionError:
        pass

    return image, ut

def PID_controller(actual_pos, desire_pos):
    global previous_error
    global previous_time
    Kp = 0.5
    Kd = 0.1
    error = np.subtract(desire_pos, actual_pos)
    current_time = time.time()
    ut = Kp * error
    if previous_error is not None:
        derivative = (error - previous_error) / (current_time - previous_time)
        ut += Kd * derivative
    previous_error = error
    previous_time = current_time
    return ut

def face_tracker_tensorflow(img, gaze):
    rects = find_faces(img, face_model)
    ut = [1000, 1000]
    max_area = 0
    max_centre = None
    for rect in rects:
        x1, y1, x2, y2 = rect
        area = np.abs((x2-x1)*(y2-y1))
        if area > max_area:
            max_area = area
            cX = int(np.round(0.5 * (x1 + x2)))
            cY = int(np.round(0.5 * (y1 + y2)))
            max_centre = (cX, cY)
        img = gaze_direction(img, gaze)
        head_direction(img, rect)
    if max_centre is not None:
        cv2.circle(img, max_centre, 5, (0, 0, 255), 4, 3)
        ut = PID_controller(max_centre, DESIRED_POS)
    return img, ut

def face_tracker_dlib(img, gaze):
    gaze.refresh(img)
    ut = [1000, 1000]
    max_area = 0
    max_centre = None
    for face in gaze.faces:
        x1 = face.left()
        x2 = face.right()
        y1 = face.top()
        y2 = face.bottom()
        rect = [x1, y1, x2, y2]
        area = np.abs((x2 - x1) * (y2 - y1))
        if area > max_area:
            max_area = area
            cX = int(np.round(0.5 * (x1 + x2)))
            cY = int(np.round(0.5 * (y1 + y2)))
            max_centre = (cX, cY)
        head_direction(img, rect)
    if max_centre is not None:
        cv2.circle(img, max_centre, 5, (0, 0, 255), 4, 3)
        ut = PID_controller(max_centre, DESIRED_POS)
    return img, ut

def gaze_direction(frame, gaze):
    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    return frame



def head_direction(img, face):
    try:
        marks = detect_marks(img, landmark_model, face)
        # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
        image_points = np.array([
            marks[30],  # Nose tip
            marks[8],  # Chin
            marks[36],  # Left eye left corner
            marks[45],  # Right eye right corne
            marks[48],  # Left Mouth corner
            marks[54]  # Right mouth corner
        ], dtype="double")
        image_points_reshape = np.ascontiguousarray(image_points[:, :2]).reshape((image_points.shape[0], 1, 2))


        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points_reshape, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector, camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

        cv2.line(img, p1, p2, (0, 255, 255), 2)
        cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
        # for (x, y) in marks:
        #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
        # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
        try:
            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
            ang1 = int(math.degrees(math.atan(m)))
        except:
            ang1 = 90

        try:
            m = (x2[1] - x1[1]) / (x2[0] - x1[0])
            ang2 = int(math.degrees(math.atan(-1 / m)))
        except:
            ang2 = 90

            # print('div by zero error')
        if ang1 >= 48:
            print('Head down')
            cv2.putText(img, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
        elif ang1 <= -48:
            print('Head up')
            cv2.putText(img, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)

        if ang2 >= 48:
            print('Head right')
            cv2.putText(img, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
        elif ang2 <= -48:
            print('Head left')
            cv2.putText(img, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)

        cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
        cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)

    except cv2.error:
        pass


def nothing(x):
    pass

# prepare face recognition and eye tracking libraries
face_model = get_face_detector()
landmark_model = get_landmark_model()
# eye tracking requirements
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]
img = np.zeros((480, 640, 3), np.uint8)
thresh = img.copy()
cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)
cv2.createTrackbar('threshold', 'image', 75, 255, nothing)
# head pose requirements
size = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX
# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])
# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

# Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
# all interfaces)
image_server_socket = socket.socket()
image_server_socket.bind(('0.0.0.0', 8001))
image_server_socket.listen(0)

# start new socket to send data back to the client
result_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result_server_socket.bind(('0.0.0.0', 8080))
result_server_socket.listen(0)

# Accept a single connection and make a file-like object out of it
image_connection = image_server_socket.accept()[0].makefile('rb')
result_connection = result_server_socket.accept()[0]

# declare global variables for control algorithm
range_1 = (170, 110, 50)
range_2 = (180, 230, 150)
DESIRED_POS = (320, 240)  # centre of screen
previous_error = None
previous_time = None

gaze = GazeTracking()

try:
    while True:
        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop
        image_len = struct.unpack('<L', image_connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        image_stream.write(image_connection.read(image_len))
        # Rewind the stream, open it as an image with PIL and do some
        # processing on it
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        ##### COMPUTER VISION #######
        img, ut = face_tracker_tensorflow(img, gaze)
        #img, ut = track_color(img, range_1, range_2)
        ##############################

        cv2.imshow("Image", img)
        send_data = struct.pack('<2f', ut[0], ut[1])
        result_connection.send(send_data)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('yep')
            message = 'close'
            result_connection.send(message.encode())

finally:
    image_connection.close()
    image_server_socket.close()
    result_connection.close()
    result_server_socket.close()
