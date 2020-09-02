from picamera import PiCamera
from picamera.array import PiRGBArray
import numpy as np
import cv2
import time
import RPi.GPIO as GPIO  # Import module “RPi.GPIO” and gave it a nickname “GPIO”
import matplotlib.pyplot as plt

# setup GPIO modules
# Define button pin numbers
btn_pin_1 = 16
btn_pin_2 = 18

# declare global variables
# draw_erase = "draw", "erase" or "off"
draw_erase = "off"

# Suppress warnings
GPIO.setwarnings(False)

# set pin numbering to board
GPIO.setmode(GPIO.BOARD)

# set pins
GPIO.setup(btn_pin_1, GPIO.IN)
GPIO.setup(btn_pin_2, GPIO.IN)


# interrupt function of button 1 changes global boolean
def button_callback_1(channel):
    global draw_erase
    if draw_erase == "draw":
        draw_erase = "off"
    elif draw_erase == "off":
        draw_erase = "draw"

# interrupt function of button 2 changes global boolean
def button_callback_2(channel):
    global button_2_edge
    button_2_edge = True

def color_segmentation(range1, range2):
    # checked for rising edge of either button (switch transitions closed to open)
    GPIO.add_event_detect(btn_pin_1, GPIO.RISING, callback=button_callback_1, bouncetime=50)
    GPIO.add_event_detect(btn_pin_2, GPIO.RISING, callback=button_callback_2, bouncetime=50)
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

    currentLine = []
    listOfLines = []
    previous = draw_erase
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_image, range1, range2)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        if draw_erase == "draw":
            if previous != draw_erase:
                listOfLines.append(np.array(currentLine))
                currentLine = []

            # calc moments
            try:
                M = cv2.moments(mask)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                currentLine.append([cX, cY])
                cv2.circle(image, (cX, cY), 5, (0, 0, 255), 4, 3)
            except ZeroDivisionError:
                pass

        previous = draw_erase
        cv2.polylines(image, listOfLines, isClosed=False, color=(255, 0, 0), thickness=2)
        cv2.polylines(image, [np.array(currentLine)], isClosed=False, color=(255, 0, 0), thickness=2)

        # cv2.imshow('PP', masked_image)
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
            print(hsv_image[240, 320])

        if key == ord('q'):
            break

    # When everything done, release the capture
    # cv2.destroyAllWindows()


range_1 = (165, 140, 60)
range_2 = (175, 230, 130)
color_segmentation(range_1, range_2)

# median_hsv()

# continuousCapture()

# blurImage()
# continuousCaptureFace()