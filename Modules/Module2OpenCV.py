from picamera import PiCamera
from picamera.array import PiRGBArray
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

# function for displaying video of raspberry pi
def continuousCapture():
    # initialise object
    camera = PiCamera()
    # configure camera setting
    camera.resolution = (640, 480)
    camera.framerate = 32
    # sleep and update settings
    time.sleep(2)
    camera.awb_mode = 'auto'
    # initialise the picture arrage with the corresponding size
    rawCapture = PiRGBArray(camera, size=(640, 480))

    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

        # grab the raw numpy array representing the image, then initialise the timestamp and occiped/unoccupied text
        image = frame.array
        # call canny to find edges
        #image = cv2.Canny(image,100,200)
        # Display the resulting frame
        cv2.imshow("PP", image)
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    # cv2.destroyAllWindows()

def blurImage():
    # initialise object
    camera = PiCamera()
    # configure camera setting
    camera.resolution = (640, 480)
    camera.framerate = 32
    # initialise the picture arrage with the corresponding size
    rawCapture = PiRGBArray(camera, size=(640, 480))
    blur = 0
    pix = 0
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

        # grab the raw numpy array representing the image, then initialise the timestamp and occiped/unoccupied text
        image = frame.array
        blurseimage = image;
        if blur == 1:
            blurseimage = cv2.GaussianBlur(image, (51, 51), 0)

        if pix == 1:
            blurseimage = img_pixelate(image, 20)

        # Display the resulting frame
        cv2.imshow("PP", blurseimage)
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('b'):
            # toggle the blur output
            blur = 1 - blur
        if key == ord('p'):
            # toggle the blur output
            pix = 1 - pix

        if key == ord('q'):
            break

def img_pixelate(image, blocks=3):
    # divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (B, G, R), -1)
    # return the pixelated blurred image
    return image

    # When everything done, release the capture
    # cv2.destroyAllWindows()


def color_segmentation(range1, range2):
    # initialise object
    camera = PiCamera()
    # configure camera setting
    camera.resolution = (640, 480)
    camera.framerate = 32
    # sleep and update settings
    time.sleep(2)
    camera.awb_mode = 'off'
    camera.awb_gains = 1.3
    # initialise the picture arrage with the corresponding size
    rawCapture = PiRGBArray(camera, size=(640, 480))
    
    centres = []
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_image, range1, range2)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # calc moments
        try:
            M = cv2.moments(mask)
            cX = int(M["m10"]/M["m00"])
            cY = int(M["m01"]/M["m00"])
            centres.append([cX,cY])
            cv2.circle(image, (cX, cY), 5, (0, 0, 255), 4, 3)
        except ZeroDivisionError:
            pass
        cv2.polylines(image, [np.array(centres)], isClosed = False, color = (255,0,0), thickness=2)


        #cv2.imshow('PP', masked_image)
        cv2.imshow('PP', image)
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    # cv2.destroyAllWindows()

def median_hsv():
    # initialise object
    camera = PiCamera()
    # configure camera setting
    camera.resolution = (640, 480)
    camera.framerate = 32
    # sleep and update settings
    time.sleep(2)
    camera.awb_mode = 'off'
    camera.awb_gains = 1.3
    # initialise the picture arrage with the corresponding size
    rawCapture = PiRGBArray(camera, size=(640, 480))

    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        cv2.circle(image, (320, 240), 5, (0, 0, 255), 4, 3)
        cv2.imshow('PP', image)
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            print(hsv_image[240,320])

        if key == ord('q'):
            break

    # When everything done, release the capture
    # cv2.destroyAllWindows()
def continuousCaptureFace():
    # initialise object
    camera = PiCamera()
    # configure camera setting
    camera.resolution = (640, 480)
    camera.framerate = 32
    # sleep and update settings
    time.sleep(2)
    camera.awb_mode = 'off'
    camera.awb_gains = 1.3
    # initialise the picture arrage with the corresponding size
    rawCapture = PiRGBArray(camera, size=(640, 480))

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

        # grab the raw numpy array representing the image, then initialise the timestamp and occiped/unoccupied text
        image = frame.array
        # call canny to find edges
        #image = cv2.Canny(image,100,200)
        # Display the resulting frame
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
    
        cv2.imshow("PP", image)
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    # cv2.destroyAllWindows()


range_1 = (165, 140, 60)
range_2 = (175, 230, 130)
#color_segmentation(range_1, range_2)

#median_hsv()

#continuousCapture()

#blurImage()
continuousCaptureFace()