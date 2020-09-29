#import cv2
#import numpy as np
import time
#import dlib
from gaze_tracking import GazeTracking
from head_pose_estimation import *
from headpose import HeadposeDetection
from compare import *
import copy
import pandas as pd

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
        self.required_sim = 1

        # declare gaze object
        self.gaze = GazeTracking(self.gaze_lower_ratio, self.gaze_upper_ratio)
        self.headpose = HeadposeDetection(self.headpose_angle_limit)
        self.faces_df = pd.DataFrame(columns=['faces', 'landmarks', 'priority'])

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
        self.orig_img = copy.deepcopy(image)
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

        # update pandas df with new faces
        self.update_pandas_faces()

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

    def update_pandas_faces(self):
        # add empty list to start of each entry and pop the end
        for index, row in self.faces_df.iterrows():
            row['faces'].insert(0, [])
            row['priority'].insert(0, [])
            row['landmarks'].insert(0, [])
            row['faces'].pop()
            row['priority'].pop()
            row['landmarks'].pop()

        for key, value in self.speaker_dict.items():
            # flag to track when a face has been matched
            matched_face = True

            # iterate through each face in the current priority
            for face in value:
                x1, y1, x2, y2 = face
                roi = self.orig_img[y1:y2, x1:x2]

                # continue without match if no face is found
                try:
                    landmarks = getRep(roi)
                except:
                    print("couldn't find face")
                    continue

                # we have found a face, so set matched_face to False
                matched_face = False
                # loop through each row (corresponding to a set of the same faces) to see
                # if this face matches a recently seen one.
                for index, row in self.faces_df.iterrows():
                    # loop through all faces in a row and calculate the average similarity
                    # between faces
                    counter = 0
                    total_sim = 0
                    for j in range(1, self.num_previous):
                        if row["faces"][j]:
                            counter+=1
                            # simply compare with previous face in each entry!!
                            d = landmarks - row["landmarks"][j]
                            sim_score = np.dot(d, d)
                            print(sim_score)
                            total_sim+=sim_score
                    ave_sim = 1000
                    if counter:
                        ave_sim = total_sim/counter
                    print(ave_sim)
                    # if the average similarity is sufficiently low, the faces are considered
                    # matched
                    if ave_sim < self.required_sim:
                        row['faces'][0] = face
                        row['priority'][0] = key
                        row['landmarks'][0] = landmarks
                        matched_face = True
                        break

                # if the face has been matched or the landmarks could not be applied to the face,
                # continue to the next face.
                # if a face was found but not matched, we need to add it as a new entry to the df,
                # so we move to the final section of code.
                if matched_face:
                    continue

                # if face gets to hear, it hasn't been matched to a previous face
                # we thus want to add it as a new entry
                a = []
                b = []
                c = []
                for i in range(self.num_previous):
                    a.append([])
                    b.append([])
                    c.append([])
                a.insert(0, face)
                b.insert(0, landmarks)
                c.insert(0, key)
                a.pop()
                b.pop()
                c.pop()
                self.faces_df = self.faces_df.append({"faces": a, "landmarks": b, "priority": c}, ignore_index=True)

        # delete any rows which no longer store any faces
        for index, row in self.faces_df.iterrows():
            if not any(row["faces"]):
                self.faces_df = self.faces_df.drop([index])

        print(self.faces_df["faces"])
        print(self.faces_df["landmarks"])
        print(self.faces_df["priority"])




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