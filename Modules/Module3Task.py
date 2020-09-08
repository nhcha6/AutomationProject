import pigpio
import time
def servoCamPan():
    pi = pigpio.pi() #connect to local Pi.
    # pi.set_mode(23, pigpio.INPUT) #set pin 23 as input
    # pi.set_pull_up_down(23, pigpio.PUD_UP) #set internal pull up resistor for pin 23
    # print(pi.read(23)) #get the pin status, should print 1
    pi.set_mode(18, pigpio.OUTPUT)
    pi.set_mode(17, pigpio.OUTPUT)
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
    pi = pigpio.pi() #connect to local Pi.
    # pi.set_mode(23, pigpio.INPUT) #set pin 23 as input
    # pi.set_pull_up_down(23, pigpio.PUD_UP) #set internal pull up resistor for pin 23
    # print(pi.read(23)) #get the pin status, should print 1
    pi.set_mode(18, pigpio.OUTPUT)
    pi.set_mode(17, pigpio.OUTPUT)
    pi.set_servo_pulsewidth(18, 1500)
    pi.set_servo_pulsewidth(17, 1000)

def vibrateControl():
    pi = pigpio.pi() #connect to local Pi.
    pi.set_mode(23, pigpio.INPUT) #set pin 23 as input
    pi.set_pull_up_down(23, pigpio.PUD_UP) #set internal pull up resistor for pin 23
    print(pi.read(23)) #get the pin status, should print 1
    pi.set_mode(18, pigpio.OUTPUT)
    pi.set_PWM_frequency(18, 50)
    for i in range(0, 50, 5):
        pi.set_PWM_dutycycle(18, i)
        time.sleep(1)


servoCamPan()
servoCamCentre()