import pigpio
import time
from picamera import PiCamera
from picamera.array import PiRGBArray
import numpy as np
import cv2
import RPi.GPIO as GPIO  # Import module “RPi.GPIO” and gave it a nickname “GPIO”
import matplotlib.pyplot as plt

def track_colour(range1, range2):
    """
    function which isolates the colour between the input colour range and actuates the servos
    to centre the object in the middle of the camera field of view.
    :param range1: colour lower bound
    :param range2: colour upper bound
    :return:
    """

    global previous_error
    global previous_time
 
    camera = PiCamera()
    # configure camera setting
    camera.resolution = (640, 480)
    camera.framerate = 32
    desire_pos = (320, 240)
    # sleep and update settings
    time.sleep(2)
    camera.awb_mode = 'off'
    camera.awb_gains = 1.3
    # initialise the picture arrage with the corresponding size
    rawCapture = PiRGBArray(camera, size=(640, 480))

    # continuously capture frames
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        maskFinger = cv2.inRange(hsv_image, range1, range2)
       
        # calc moments
        try:
            M = cv2.moments(maskFinger)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(image, (cX, cY), 5, (0, 0, 255), 4, 3)
            # give actual position to PID_controller function
            PID_controller((cX, cY), desire_pos)
        except ZeroDivisionError:
            #previous_time = None
            #previous_error = None
            pass

        # display image
        cv2.imshow('PP', image)
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            servoCamCentre()
            break

    # When everything done, release the capture
    # cv2.destroyAllWindows()

# takes the actual position of tracked object and calculates the change in servo angle required
# to bring it towards the desired position (the centre of the camera).
def PID_controller(actual_pos, desire_pos):
    global previous_error
    global previous_time
    Kp = 0.3
    Kd = 0.02
    error = np.subtract(desire_pos, actual_pos)
    current_time = time.time()
    ut = Kp * error
    if previous_error is not None:
        derivative = (error - previous_error) / (current_time - previous_time)
        #print(Kd*derivative)
        ut += Kd * derivative

    # set the servos to be equal to the calculate change.
    set_servo(ut)
    previous_error = error
    previous_time = current_time

# takes a tuple containing the changes required to the servo pulsewidth, and updates the
# servo position.
def set_servo(ut):
    # pi.set_mode(23, pigpio.INPUT) #set pin 23 as input
    # pi.set_pull_up_down(23, pigpio.PUD_UP) #set internal pull up resistor for pin 23
    # print(pi.read(23)) #get the pin status, should print 1
    try:
        current_pulsewidth_X = pi.get_servo_pulsewidth(18)
        desire_pulsewidth_X = current_pulsewidth_X + ut[0]
        if desire_pulsewidth_X < 500:
            desire_pulsewidth_X = 500
        if desire_pulsewidth_X > 2500:
            desire_pulsewidth_X = 2500
        pi.set_servo_pulsewidth(18, desire_pulsewidth_X)

        current_pulsewidth_Y = pi.get_servo_pulsewidth(17)
        desire_pulsewidth_Y = current_pulsewidth_Y - ut[1]
        if desire_pulsewidth_Y < 500:
            desire_pulsewidth_Y = 500
        if desire_pulsewidth_Y > 2500:
            desire_pulsewidth_Y = 2500
        pi.set_servo_pulsewidth(17, desire_pulsewidth_Y)
    except AttributeError as e:
        print(e)


# other function involved in testing the servo/motor disk control. Not related directly to
# Module Task 3.
def servoCamPan():
    # pi.set_mode(23, pigpio.INPUT) #set pin 23 as input
    # pi.set_pull_up_down(23, pigpio.PUD_UP) #set internal pull up resistor for pin 23
    # print(pi.read(23)) #get the pin status, should print 1
    pi.set_servo_pulsewidth(18, 0)    # off
    pi.set_servo_pulsewidth(17, 0)    # off
    time.sleep(2)
    # 500 anti clockwise, 2500 clockwise
    for i in range(500, 2500, 5):
        pi.set_servo_pulsewidth(18, i)
        pi.set_servo_pulsewidth(17, i)
        time.sleep(0.02)
    pi.set_servo_pulsewidth(18, 0)    # off
    pi.set_servo_pulsewidth(17, 0)    # off

def servoCamCentre():
    pi.set_servo_pulsewidth(18, 1500)
    pi.set_servo_pulsewidth(17, 1000)

def vibrateControl():
    pi.set_mode(23, pigpio.INPUT) #set pin 23 as input
    pi.set_pull_up_down(23, pigpio.PUD_UP) #set internal pull up resistor for pin 23
    print(pi.read(23)) #get the pin status, should print 1
    pi.set_mode(18, pigpio.OUTPUT)
    pi.set_PWM_frequency(18, 50)
    for i in range(0, 50, 5):
        pi.set_PWM_dutycycle(18, i)
        time.sleep(1)

# set the range to track the blue bottle cap.
range_1 = (170, 110, 50)
range_2 = (180, 230, 150)
# declare previous error and previous time,
previous_error = None
previous_time = None
pi = pigpio.pi() #connect to local Pi.
# set pin modes for servos.
pi.set_mode(18, pigpio.OUTPUT)
pi.set_mode(17, pigpio.OUTPUT)
# centre servo to begin
servoCamCentre()
# call the tracking function.
track_colour(range_1, range_2)