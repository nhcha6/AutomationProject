import time # Import module “time” to use sleep function
import RPi.GPIO as GPIO # Import module “RPi.GPIO” and gave it a nickname “GPIO”
import threading
import ctypes
import sys
import signal

# initialise delay global variable
delay = 0.5

# Define led pin numbers
led_pin_1 = 11
led_pin_2 = 13
led_pin_3 = 15

# Define button pin numbers
btn_pin_1 = 16
btn_pin_2 = 18

# initialise button edge bools to check if a button has been pressed
button_1_edge = False
button_2_edge = False

# Suppress warnings
GPIO.setwarnings(False)

# set pin numbering to board
GPIO.setmode(GPIO.BOARD)

# set up pins
GPIO.setup(led_pin_1, GPIO.OUT)
GPIO.setup(led_pin_2, GPIO.OUT)
GPIO.setup(led_pin_3, GPIO.OUT)
GPIO.setup(btn_pin_1, GPIO.IN)
GPIO.setup(btn_pin_2, GPIO.IN)

# turn all leds off to start.
GPIO.output(led_pin_1, GPIO.LOW)
GPIO.output(led_pin_2, GPIO.LOW)
GPIO.output(led_pin_3, GPIO.LOW)

# thread takes delay time, and the pin numbers of the outer LEDs as input
class thread(threading.Thread):
    def __init__(self, delay, led_pin_1, led_pin_3):
        threading.Thread.__init__(self)
        self.delay = delay
        self.led_pin_1 = led_pin_1
        self.led_pin_3 = led_pin_3

    def run(self):
        # target function of the thread class
        try:
            while True:
                GPIO.output(self.led_pin_1, GPIO.HIGH) # Turn outer LED on
                time.sleep(self.delay) #delay
                GPIO.output(self.led_pin_1, GPIO.LOW) # Turn outer LED off
                time.sleep(self.delay) # delay
                GPIO.output(13, GPIO.HIGH) # Turn LED2 on
                time.sleep(self.delay) # delay
                GPIO.output(13, GPIO.LOW) # Turn LED2 off
                time.sleep(self.delay) # delay
                GPIO.output(self.led_pin_3, GPIO.HIGH) # Turn outer LED on
                time.sleep(self.delay) # delay
                GPIO.output(self.led_pin_3, GPIO.LOW) # Turn outer LED off
                time.sleep(self.delay) # delay


        finally:
            print('ended')
            
    def get_id(self):
    # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        # stop the thread when the exception received
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
              ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')
            
def signal_handler(sig, frame):
    GPIO.cleanup()
    sys.exit(0)

# interrupt function of button 1 changes global boolean
def button_callback_1(channel):
    global button_1_edge
    button_1_edge = True

# interrupt function of button 2 changes global boolean
def button_callback_2(channel):
    global button_2_edge
    button_2_edge = True

if __name__ == '__main__':

    # checked for falling edge of either button (switch transitions closed to open)
    GPIO.add_event_detect(btn_pin_1, GPIO.FALLING, callback = button_callback_1, bouncetime=50)
    GPIO.add_event_detect(btn_pin_2, GPIO.FALLING, callback = button_callback_2, bouncetime=50)
    
    #signal.signal(signal.SIGINT, signal_handler)
    #signal.pause()
    
    blinking_thread = thread(delay,led_pin_1,led_pin_3)
    blinking_thread.start()

    while True:
        # if rising edge of button 1 was detected
        if button_1_edge:
            # exit from the run block of the old thread
            blinking_thread.raise_exception()
            # turn all LEDs off
            GPIO.output(11, GPIO.LOW)
            GPIO.output(13, GPIO.LOW)
            GPIO.output(15, GPIO.LOW)
            # change delay from 0.5 to 2 or 2 to 0.5
            delay=1/delay
            # start a new thread with new delay
            blinking_thread_temp = thread(delay,led_pin_1,led_pin_3)
            blinking_thread_temp.start()
            # terminate thread safely with join()
            #blinking_thread.join()
            # set boolean back to false
            button_1_edge = False
            # rename new thread as main thread
            blinking_thread = blinking_thread_temp

         # if rising edge of button 2 was detected           
        if button_2_edge:
            # exit from the run block of the old thread
            blinking_thread.raise_exception()
            # turn all LEDs off
            GPIO.output(11, GPIO.LOW)
            GPIO.output(13, GPIO.LOW)
            GPIO.output(15, GPIO.LOW)
            # switch button pin numbers to change sequence order
            led_pin_1, led_pin_3 = led_pin_3, led_pin_1
            # start a new thread with new pin numbers
            blinking_thread_temp = thread(delay,led_pin_1,led_pin_3)
            blinking_thread_temp.start()
            # terminate thread safely with join()
            #blinking_thread.join()
            # set boolean back to false
            button_2_edge = False
            # rename new thread as main thread
            blinking_thread = blinking_thread_temp


