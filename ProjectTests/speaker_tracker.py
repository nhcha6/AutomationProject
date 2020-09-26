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

        # declare hyper parameters for control algorithm
        self.desire_pos = (320, 240)  # centre of screen
        self.Kp = 0.5
        self.Kd = 0.1
        self.previous_error = None
        self.previous_time = None

        # declare hyper paramaters for speaker tracking
        self.headpose_angle_limit = 25
        self.gaze_lower_ratio = 0.6
        self.gaze_upper_ratio = 0.8
        self.num_previous = 5

        # declare gaze object
        self.gaze = GazeTracking(self.gaze_lower_ratio, self.gaze_upper_ratio)
        self.headpose = HeadposeDetection(self.headpose_angle_limit)
        self.previous_speaker_data = []

        # variables to be updated each image
        self.img = None
        self.size = None
        self.ut = None
        self.faces = None
        self.track_face = None
        self.speaker_dict = None
        self.head_pose_faces = []
        self.gaze_faces = []

    def refresh(self, image):
        self.img = image
        self.size = image.shape
        self.track_face = None
        self.speaker_dict = None
        self.ut = [1000, 1000]
        self.new_face_tracker()

    def new_face_tracker(self):
        # get faces
        self.faces = find_faces(self.img, self.face_model)

        # run head_direction analysis on faces
        self.head_pose_new()

        # run gaze direction analysis on faces
        self.gaze_direction()

        # convert face lists to a dictionary summarising the data
        self.summarise_speaker_data()

        # summarise the speaker details on the frame
        self.summarise_frame()

        # find the face to track
        self.find_track_face()

        # run PID controller
        if self.track_face:
            self.PID_controller()

    def summarise_speaker_data(self):
        self.speaker_dict = {}
        self.speaker_dict['gaze'] = self.gaze_faces
        self.speaker_dict['headpose'] = []
        self.speaker_dict['face'] = []
        for face in self.head_pose_faces:
            if face not in self.speaker_dict['gaze']:
                self.speaker_dict['headpose'].append(face)
        for face in self.faces:
            if face not in self.speaker_dict['headpose']:
                if face not in self.speaker_dict['gaze']:
                    self.speaker_dict['face'].append(face)

        self.previous_speaker_data.insert(0, self.speaker_dict)
        if len(self.previous_speaker_data) > self.num_previous:
            self.previous_speaker_data.pop()

    def summarise_frame(self):
        for key, value in self.speaker_dict.items():
            self.highlight_faces(key, value)


    def find_track_face(self):
        if self.gaze_faces:
            if len(self.gaze_faces)==1:
                self.track_face = self.gaze_faces[0]
            else:
                self.biggest_face(self.gaze_faces)
        elif self.head_pose_faces:
            if len(self.head_pose_faces)==1:
                self.track_face = self.head_pose_faces[0]
            else:
                self.biggest_face(self.head_pose_faces)
        elif self.faces:
            if len(self.faces)==1:
                self.track_face = self.faces[0]
            else:
                self.biggest_face(self.faces)


    def head_pose_new(self):
        self.img, self.head_pose_faces = self.headpose.process_image(self.img, self.faces, False)

    def gaze_direction(self):
        # We send this frame to GazeTracking to analyze it
        self.gaze_faces = self.gaze.refresh(self.img, self.head_pose_faces)

        #self.img = self.gaze.annotated_frame()

        #ratio = str(self.gaze.horizontal_ratio())
        #cv2.putText(self.img, ratio, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)


    def biggest_face(self, faces):
        max_face = None
        max_area = 0
        for face in faces:
            x1, y1, x2, y2 = face
            area = np.abs((x2 - x1) * (y2 - y1))
            if area > max_area:
                max_area = area
                #cX = int(np.round(0.5 * (x1 + x2)))
                #cY = int(np.round(0.5 * (y1 + y2)))
                max_face = face
        if max_face is not None:
            self.track_face = max_face

    def highlight_faces(self, priority, faces):
        # select colour
        if priority == 'headpose':
            colour_bgr = (0, 255, 0)
        if priority == 'face':
            colour_bgr = (0, 0, 255)
        if priority == 'gaze':
            colour_bgr = (255, 0, 0)

        for face in faces:
            x1, y1, x2, y2 = face
            cX = int(np.round(0.5 * (x1 + x2)))
            cY = int(np.round(0.5 * (y1 + y2)))
            cv2.circle(self.img, (cX, cY), 5, colour_bgr, 4, 3)

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