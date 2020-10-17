import numpy as np
import cv2
import tensorflow as tf
import time
from tqdm import tqdm

# Convert the model.
# converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(keras_file)
# tflite_model = converter.convert()
#
# # Save the model.
# with open('model.tflite', 'wb') as f:
#   f.write(tflite_model)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

types = ['ship', 'kangaroo', 'car']
for type in types:
    print("quick_test/" + type + '.jpg')
    img = cv2.imread("quick_test/" + type + '.jpg', cv2.IMREAD_COLOR)
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

    cv2.imshow(type, gray_img)

    cv2.waitKey(0)

