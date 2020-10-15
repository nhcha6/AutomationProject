import numpy as np
import cv2
import tensorflow as tf
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

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
    cv2.imshow("image", image)

    if cv2.waitKey(1) & 0xFF == ord('r'):
        # Load the TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test the model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        print(input_data[0])
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)
        break

    rawCapture.truncate(0)

cv2.destroyAllWindows()

