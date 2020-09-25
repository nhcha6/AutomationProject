import cv2
import numpy as np
import time
import dlib
from gaze_tracking import GazeTracking
from head_pose_estimation import *
from headpose import HeadposeDetection

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
        self.headpose = HeadposeDetection()

        # variables to be updated each image
        self.img = None
        self.size = None
        self.ut = None
        self.faces = None
        self.track_face = None
        self.new_faces = []

    def refresh(self, image):
        self.img = image
        self.size = image.shape
        self.track_face = None
        self.ut = [1000, 1000]
        self.new_face_tracker()

    def new_face_tracker(self):
        # get faces
        self.faces = find_faces(self.img, self.face_model)
        # run head_direction analysis on faces
        self.head_pose_new()

        # if one face has been isolated track_face will not be None and we call the PID controller and return
        #if self.track_face:
        #    self.PID_controller()
        #    return

        self.gaze_direction()

        # if one face has been isolated track_face will not be None and we call the PID controller and return
        if self.track_face:
            self.PID_controller()
            return

        # We get to here with either multiple faces. If multiple split based on size.
        if self.faces:
            self.biggest_face()
            self.track_face = self.faces[0]
            self.PID_controller()

    def head_pose_new(self):
        self.img, self.new_faces = self.headpose.process_image(self.img, False)
        # update faces if there is a at least one person facing the camera.
        if self.new_faces:
            self.faces = self.new_faces
            self.new_faces = []
            self.highlight_faces("green")

        # if no faces we wish to update faces to be hold only the largest face.
        else:
            self.biggest_face()

        # if only one face left, add it to track_face
        if len(self.faces) == 1:
            self.track_face = self.faces[0]


    def biggest_face(self):
        max_face = None
        max_area = 0
        for face in self.faces:
            x1, y1, x2, y2 = face
            area = np.abs((x2 - x1) * (y2 - y1))
            if area > max_area:
                max_area = area
                cX = int(np.round(0.5 * (x1 + x2)))
                cY = int(np.round(0.5 * (y1 + y2)))
                max_face = face
        if max_face is not None:
            self.faces = [max_face]
        self.highlight_faces("red")

    def highlight_faces(self, colour):
        # select colour
        if colour == "green":
            colour_bgr = (0, 255, 0)
            radius = 3
        if colour == "red":
            colour_bgr = (0, 0, 255)
            radius = 6
        if colour == "blue":
            colour_bgr = (255, 0, 0)
            radius = 9

        for face in self.faces:
            x1, y1, x2, y2 = face
            cX = int(np.round(0.5 * (x1 + x2)))
            cY = int(np.round(0.5 * (y1 + y2)))
            cv2.circle(self.img, (cX, cY), radius, colour_bgr, 4, 3)


    def gaze_direction(self):
        print("GAZEDIRECTION")
        # We send this frame to GazeTracking to analyze it
        self.new_faces = self.gaze.refresh(self.img, self.faces)

        # update faces if there is a at least one person facing the camera.
        if self.new_faces:
            self.faces = self.new_faces
            self.new_faces = []
            self.highlight_faces("blue")

        # if no faces we wish to update faces to be hold only the largest face.
        else:
            self.biggest_face()

        # if only one face left, add it to track_face
        if len(self.faces) == 1:
            self.track_face = self.faces[0]

        #self.img = self.gaze.annotated_frame()

        #ratio = str(self.gaze.horizontal_ratio())
        #cv2.putText(self.img, ratio, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        # text = ""
        #
        # if self.gaze.is_blinking():
        #     text = "Blinking"
        # elif self.gaze.is_right():
        #     text = "Looking right"
        # elif self.gaze.is_left():
        #     text = "Looking left"
        # elif self.gaze.is_center():
        #     text = "Looking center"
        #
        # cv2.putText(self.img, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        #
        # left_pupil = self.gaze.pupil_left_coords()
        # right_pupil = self.gaze.pupil_right_coords()
        # cv2.putText(self.img, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        # cv2.putText(self.img, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    def old_head_pose(self):
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

        for face in self.faces:
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

                # for p in image_points:
                #     cv2.circle(self.img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                x1, x2 = head_pose_points(self.img, rotation_vector, translation_vector, camera_matrix)

                #cv2.line(self.img, p1, p2, (0, 255, 255), 2)
                #cv2.line(self.img, tuple(x1), tuple(x2), (255, 255, 0), 2)

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

                # uncomment to see visual representation of calculated angles
                # if ang1 >= 48:
                #     print('Head down')
                #     cv2.putText(self.img, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
                # elif ang1 <= -48:
                #     print('Head up')
                #     cv2.putText(self.img, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)
                #
                # if ang2 >= 48:
                #     print('Head right')
                #     cv2.putText(self.img, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
                # elif ang2 <= -48:
                #     print('Head left')
                #     cv2.putText(self.img, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)

                if ang2 <= 45:
                    if ang2 >= -45:
                        self.new_faces.append(face)

                #cv2.putText(self.img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
                #cv2.putText(self.img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)

            except cv2.error:
                pass

        # update faces if there is a at least one person facing the camera.
        if self.new_faces:
            self.faces = self.new_faces
            self.new_faces = []
            self.highlight_faces("green")
        # if no faces we wish to update faces to be hold only the largest face.
        else:
            self.biggest_face()

        # if only one face left, add it to track_face
        if len(self.faces)==1:
            self.track_face = self.faces[0]


    def PID_controller(self):
        x1, y1, x2, y2 = self.track_face
        cX = int(np.round(0.5 * (x1 + x2)))
        cY = int(np.round(0.5 * (y1 + y2)))
        actual_pos = (cX, cY)
        error = np.subtract(self.desire_pos, actual_pos)
        current_time = time.time()
        self.ut = self.Kp * error
        if self.previous_error is not None:
            derivative = (error - self.previous_error) / (current_time - self.previous_time)
            self.ut += self.Kd * derivative
        self.previous_error = error
        self.previous_time = current_time