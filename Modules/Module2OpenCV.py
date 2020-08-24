from picamera import PiCamera
from picamera.array import PiRGBArray
import numpy as np
import cv2
import time
import bluring

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
            blurseimage = bluring.img_pixelate(image, 100)

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

    # When everything done, release the capture
    # cv2.destroyAllWindows()


#continuousCapture()

blurImage()

