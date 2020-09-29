import io
import socket
import struct
import cv2
import numpy as np
from speaker_tracker import SpeakerTracker

image_port = 8001
result_port = 8081

# Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
# all interfaces)
image_server_socket = socket.socket()
image_server_socket.bind(('0.0.0.0', image_port))
image_server_socket.listen(0)

# start new socket to send data back to the client
result_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result_server_socket.bind(('0.0.0.0', result_port))

result_server_socket.listen(0)

# Accept a single connection and make a file-like object out of it
image_connection = image_server_socket.accept()[0].makefile('rb')
result_connection = result_server_socket.accept()[0]

speaker_tracker = SpeakerTracker()


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
        #img, ut = face_tracker_tensorflow(img, gaze)
        #img, ut = track_color(img, range_1, range_2)
        ##############################

        speaker_tracker.refresh(img)

        cv2.imshow("Image", speaker_tracker.img)
        send_data = struct.pack('<2f', speaker_tracker.ut[0], speaker_tracker.ut[1])
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
