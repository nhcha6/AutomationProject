import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import time
import urllib
from urllib.request import urlopen

# Convert the model.
# converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(keras_file)
# tflite_model = converter.convert()
#
# # Save the model.
# with open('model.tflite', 'wb') as f:
#   f.write(tflite_model)

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

types = ['ship', 'kangaroo', 'car']
while True:
    try:
        url=str(input())
        if url=='q':
            break
        req = urlopen(url)
        image = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(image, -1)


        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(gray_img, (128, 128))
        input_image = frame.reshape(-1, 128, 128, 1)
        input_image = input_image.astype(np.float32)


        # Test the model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_image)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)

        #font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, types[np.argmax(output_data)], (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('img', frame)
        
        cv2.waitKey(0)
        
    except urllib.error.HTTPError:
        print('cannot open image')
        
cv2.destroyAllWindows()
