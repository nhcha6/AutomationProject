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
# draw_erase = "draw", "erase", "offErase" or "offDraw"
draw_erase = "offErase"
color_button = "Blue"
drawing_color = (0, 0, 0)
thickness = 0
chosen_color = (255, 0, 0)
currentLine = []
# Suppress warnings
GPIO.setwarnings(False)

# set pin numbering to board
GPIO.setmode(GPIO.BOARD)

# set pins
GPIO.setup(btn_pin_1, GPIO.IN)
GPIO.setup(btn_pin_2, GPIO.IN)


# interrupt function of button 1. When pressed, the global variable draw_erase is changed to move the program into the
# next state. If "draw" or "erase" are the next state in the cycle, the appropriate thickness and color are set on the
# corresponding global variables. The currentLine variable is also reset.
def button_callback_1(channel):
    global draw_erase
    global drawing_color
    global thickness
    global currentLine
    if draw_erase == "draw":
        draw_erase = "offDraw"
    elif draw_erase == "offDraw":
        draw_erase = "erase"
        currentLine = []
        drawing_color = (0, 0, 0)
        thickness = 15
    elif draw_erase == "erase":
        draw_erase = "offErase"
    elif draw_erase == "offErase":
        draw_erase = "draw"
        currentLine = []
        drawing_color = chosen_color
        thickness = 5

# interrupt function of button 2. When pressed, the drawing colour global variable is changed to the next one in the
# cycle.
def button_callback_2(channel):
    global chosen_color
    global drawing_color
    global color_button
    global currentLine
    if color_button == "Blue":
        chosen_color = (0, 255, 0)
        color_button = "Green"
        
    elif color_button == "Green":
        chosen_color = (0, 0, 255)
        color_button = "Red"
        
    elif color_button == "Red":
        chosen_color = (255, 0, 0)
        color_button = "Blue"

    # if in draw mode currently, make sure the drawing color is updated with the new chosen color, and the
    # currentLine variable is reset.
    if draw_erase == "draw":
        drawing_color = chosen_color
        currentLine = []


# function runs drawing code.
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

    # create an all black image to be the base for the drawing image.
    drawing = np.zeros((480, 640, 3), np.uint8)

    # continuously capture images in for loop.
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # extract image, convert to hsv and then apply range to get mask.
        image = frame.array
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        maskFinger = cv2.inRange(hsv_image, range1, range2)
        #masked_image = cv2.bitwise_and(image, image, mask=maskFinger)
        
        # create mask of drawn image, which has been added to in previous loops.
        drawing_gray = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(drawing_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # add the drawing image to the captured image.
        image_masked = cv2.bitwise_and(image, image, mask = mask_inv)
        drawn_image = cv2.add(image_masked, drawing)

        # if in draw mode or erase mode, capture the centre of mass of the coloured object, display it
        # on the screen as a circle, and then add the point to the line currently being drawn.
        if draw_erase == "draw" or draw_erase == "erase":
            #if previous != draw_erase:
                
                #currentLine = []

            # calc moments
            try:
                M = cv2.moments(maskFinger)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                currentLine.append([cX, cY])
                cv2.circle(drawn_image, (cX, cY), 5, (0, 0, 255), 4, 3)
            except ZeroDivisionError:
                pass

        # draw the line on the black backed drawing image.
        cv2.polylines(drawing, [np.array(currentLine)], isClosed=False, color = drawing_color, thickness=thickness)

        # cv2.imshow('PP', masked_image)
        cv2.imshow('PP', drawn_image)
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    # cv2.destroyAllWindows()

# function for displaying the hsv values for the middle point of the frame.
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

# range calculated for blue lid using the median_hsv() function.
range_1 = (170, 110, 50)
range_2 = (180, 230, 150)
# call drawing function using the prescribed range.
color_segmentation(range_1, range_2)

#median_hsv()+