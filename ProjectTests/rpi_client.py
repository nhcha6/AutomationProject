import io
import socket
import struct
import time
import picamera
import threading
import ctypes
import pigpio

image_port = 8001
result_port = 8080

# intial comment for dlib branch

def set_servo(pi, ut):
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

def servoCamCentre(pi):
    pi.set_servo_pulsewidth(18, 1500)
    pi.set_servo_pulsewidth(17, 1000)

# thread takes delay time, and the pin numbers of the outer LEDs as input
class thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        # target function of the thread class
        try:
            # create socket
            result_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result_client_socket.connect(('Nicolass-MacBook-Air.local', result_port))

            # declare servo control
            pi = pigpio.pi()  # connect to local Pi.
            pi.set_mode(18, pigpio.OUTPUT)
            pi.set_mode(17, pigpio.OUTPUT)
            servoCamCentre(pi)

            while True:
                result = result_client_socket.recv(4096)
                close_message = 'close'
                
                if len(result) == 8:
                    ut_received = struct.unpack('<2f', result)
                    print(ut_received)
                    if ut_received[0] == 1000:
                        continue
                    set_servo(pi, ut_received)

                elif result == close_message.encode():
                    print(result.decode())
                    global closeFlag
                    closeFlag = True

        finally:
            result_client_socket.close()
            servoCamCentre(pi)
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

# start thread to handle servo stuff
result_thread = thread()
result_thread.start()

# Connect a client socket to my_server:8000 (change my_server to the
# hostname of your server)
image_client_socket = socket.socket()
image_client_socket.connect(('Nicolass-MacBook-Air.local', image_port))

# Make a file-like object out of the connection
image_connection = image_client_socket.makefile('wb')

# flag for when close message recieved
closeFlag = False
try:
    camera = picamera.PiCamera()
    camera.resolution = (640, 480)
    # Start a preview and let the camera warm up for 2 seconds
    camera.start_preview()
    time.sleep(2)

    # turn off auto whiteness
    camera.awb_mode = 'off'
    camera.awb_gains = 1.3

    # Note the start time and construct a stream to hold image data
    # temporarily (we could write it directly to connection but in this
    # case we want to find out the size of each capture first to keep
    # our protocol simple)
    stream = io.BytesIO()
    for foo in camera.capture_continuous(stream, 'jpeg'):
        # Write the length of the capture to the stream and flush to
        # ensure it actually gets sent
        image_connection.write(struct.pack('<L', stream.tell()))
        image_connection.flush()
        # Rewind the stream and send the image data over the wire
        stream.seek(0)
        image_connection.write(stream.read())
        # If we've been capturing for more than 30 seconds, quit
        if closeFlag:
            break
        # Reset the stream for the next capture
        stream.seek(0)
        stream.truncate()
    # Write a length of zero to the stream to signal we're done
    image_connection.write(struct.pack('<L', 0))
finally:
    result_thread.raise_exception()
    result_thread.join()
    image_connection.close()
    image_client_socket.close()
