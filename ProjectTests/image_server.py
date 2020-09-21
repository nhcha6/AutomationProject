import io
import socket
import struct
import cv2
import numpy as np
import time

def track_color(image, range1, range2):
    # change image to hsv and create mask
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    maskFinger = cv2.inRange(hsv_image, range1, range2)

    ut = [1000, 1000]

    # calc moments and call PID
    try:
        M = cv2.moments(maskFinger)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), 4, 3)
        ut = PID_controller((cX, cY), DESIRED_POS)
    except ZeroDivisionError:
        pass

    return image, ut

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
        ut += Kd * derivative
    previous_error = error
    previous_time = current_time
    return ut

# Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
# all interfaces)
image_server_socket = socket.socket()
image_server_socket.bind(('0.0.0.0', 8000))
image_server_socket.listen(0)

# start new socket to send data back to the client
result_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result_server_socket.bind(('0.0.0.0', 8081))
result_server_socket.listen(0)

# Accept a single connection and make a file-like object out of it
image_connection = image_server_socket.accept()[0].makefile('rb')
result_connection = result_server_socket.accept()[0]

# declare global variables for control algorithm
range_1 = (170, 110, 50)
range_2 = (180, 230, 150)
DESIRED_POS = (320, 240)  # centre of screen
previous_error = None
previous_time = None

try:
    while True:
        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop
        image_len = struct.unpack('<L', image_connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        image_stream.write(image_connection.read(image_len))
        # Rewind the stream, open it as an image with PIL and do some
        # processing on it
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        ##### COMPUTER VISION #######
        img, ut = track_color(img, range_1, range_2)
        print(ut)
        ##############################

        cv2.imshow("Image", img)
        send_data = struct.pack('<2f', ut[0], ut[1])
        result_connection.send(send_data)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('yep')
            message = 'close'
            result_connection.send(message.encode())

finally:
    image_connection.close()
    image_server_socket.close()
    result_connection.close()
    result_server_socket.close()
