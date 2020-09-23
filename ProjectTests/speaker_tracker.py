import cv2
import numpy as np
import time
import dlib
from gaze_tracking import GazeTracking
from head_pose_estimation import *

class SpeakerTracker(object):
    def __init__(self):
        # prepare face recognition and eye tracking libraries
        self.face_model = get_face_detector()
        self.landmark_model = get_landmark_model()

        # declare variables for control algorithm
        self.desire_pos = (320, 240)  # centre of screen
        self.Kp = 0.5
        self.Kd = 0.1
        self.previous_error = None
        self.previous_time = None

        # declare gaze object
        self.gaze = GazeTracking()

        # variables to be updated each image
        self.img = None
        self.size = None
        self.ut = None
        self.actual_pos = None

    def refresh(self, image):
        self.img = image
        self.size = image.shape
        self.ut = [1000, 1000]
        self.face_tracker()

    def face_tracker(self):
        rects = find_faces(self.img, self.face_model)
        max_centre = None
        max_area = 0
        for rect in rects:
            x1, y1, x2, y2 = rect
            area = np.abs((x2-x1)*(y2-y1))
            if area > max_area:
                max_area = area
                cX = int(np.round(0.5 * (x1 + x2)))
                cY = int(np.round(0.5 * (y1 + y2)))
                max_centre = (cX, cY)
            self.gaze_direction()
            self.head_direction(rect)
        if max_centre is not None:
            self.actual_pos = max_centre
            cv2.circle(self.img, max_centre, 5, (0, 0, 255), 4, 3)
            self.PID_controller()

    def gaze_direction(self):
        # We send this frame to GazeTracking to analyze it
        self.gaze.refresh(self.img)

        self.img = self.gaze.annotated_frame()
        text = ""

        if self.gaze.is_blinking():
            text = "Blinking"
        elif self.gaze.is_right():
            text = "Looking right"
        elif self.gaze.is_left():
            text = "Looking left"
        elif self.gaze.is_center():
            text = "Looking center"

        cv2.putText(self.img, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = self.gaze.pupil_left_coords()
        right_pupil = self.gaze.pupil_right_coords()
        cv2.putText(self.img, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(self.img, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    def head_direction(self, face):
        # head pose requirements
        font = cv2.FONT_HERSHEY_SIMPLEX
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])
        focal_length = self.size[1]
        center = (self.size[1] / 2, self.size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        try:
            marks = detect_marks(self.img, self.landmark_model, face)
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
                cv2.circle(self.img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            x1, x2 = head_pose_points(self.img, rotation_vector, translation_vector, camera_matrix)

            cv2.line(self.img, p1, p2, (0, 255, 255), 2)
            cv2.line(self.img, tuple(x1), tuple(x2), (255, 255, 0), 2)
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
                cv2.putText(self.img, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
            elif ang1 <= -48:
                print('Head up')
                cv2.putText(self.img, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)

            if ang2 >= 48:
                print('Head right')
                cv2.putText(self.img, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
            elif ang2 <= -48:
                print('Head left')
                cv2.putText(self.img, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)

            cv2.putText(self.img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
            cv2.putText(self.img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)

        except cv2.error:
            pass

    def PID_controller(self):
        error = np.subtract(self.desire_pos, self.actual_pos)
        current_time = time.time()
        self.ut = self.Kp * error
        if self.previous_error is not None:
            derivative = (error - self.previous_error) / (current_time - self.previous_time)
            self.ut += self.Kd * derivative
        self.previous_error = error
        self.previous_time = current_time