import pigpio
import time
from picamera import PiCamera
from picamera.array import PiRGBArray
import numpy as np
import cv2
import RPi.GPIO as GPIO  # Import module “RPi.GPIO” and gave it a nickname “GPIO”
import matplotlib.pyplot as plt

def vibrateTest():
    #pi.set_mode(23, pigpio.INPUT) #set pin 23 as input
    #pi.set_pull_up_down(23, pigpio.PUD_UP) #set internal pull up resistor for pin 23
    #print(pi.read(23)) #get the pin status, should print 1
    
    for i in range(0, 50, 5):
        pi.set_PWM_dutycycle(5, i)
        time.sleep(1)
        print(i)
    pi.set_PWM_dutycycle(5, 0)

def hapticControl(value,attentionLevel,maxAttentionLevel,thresholdAttentionLevel):
    #range of angle : 500 - 2500
    angle = ((value - 1500) / 2000) * 180
    if angle < -54 and angle >= -90:
        pin = 5 
    elif angle < -18 and angle >= -54:
        pin = 6
    elif angle < 18 and angle >= -18:
        pin = 13   
    elif angle < 54 and angle >= 18:
        pin = 19
    elif angle <= 90 and angle >= 54:
        pin = 26  

    runHaptic(pin,attentionLevel,maxAttentionLevel,thresholdAttentionLevel)


def runHaptic(pin, attentionLevel,maxAttentionLevel,thresholdAttentionLevel):
    #level range (0, 60)
    #dutycycle range (0, 30)
    resetHaptic()
    dutycycle = ( attentionLevel / maxAttentionLevel) * 40
    if attentionLevel >= thresholdAttentionLevel:
        pi.set_PWM_dutycycle(pin,dutycycle)
    print(pin, dutycycle)

def resetHaptic():
    for i in [5, 6, 13, 19, 26]:
        pi.set_PWM_dutycycle(i, 0)

pi = pigpio.pi()
for i in [5, 6, 13, 19, 26]:
    pi.set_mode(i, pigpio.OUTPUT)
    pi.set_PWM_frequency(i, 80)

 
hapticControl(1000,20,60,20)