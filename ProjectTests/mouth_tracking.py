from __future__ import division
import os
import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np


class MouthTracking(object):

    def __init__(self):
        self.frame = None
        self.face = None
        self.mouth_aspect_ratio = None

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "model/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    def aspectRatio(self):
        try:
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            x1, y1, x2, y2 = self.face
            rect = dlib.rectangle(x1, y1, x2, y2)
            landmarks = self._predictor(gray, rect)
            #points = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
            points = [x for x in range(60, 68)]
            #points = [x for x in range(48, 58)]
            landmarks = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
            A = dist.euclidean(landmarks[1], landmarks[7])
            B = dist.euclidean(landmarks[3], landmarks[5])
            C = dist.euclidean(landmarks[0], landmarks[4])
            self.mouth_aspect_ratio = (A+B)/(2.0*C)
        except IndexError as e:
            print(e)
            pass

    def refresh(self, frame, face):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame=frame
        self.face=face
        self.mouth_aspect_ratio = None
        self.aspectRatio()
        return(self.mouth_aspect_ratio)


