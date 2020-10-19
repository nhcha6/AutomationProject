import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import time
import urllib
from urllib.request import urlopen

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# infinite loop waits for command line input of image url, and then runs the model on the image.
types = ['ship', 'kangaroo', 'car']
while True:
    # try to get url and run model
    try:
        # get input from user
        url=str(input())
        # break if 'q' recieved.
        if url=='q':
            break

        # open url image
        req = urlopen(url)
        image = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(image, -1)

        # convert to gray
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(gray_img, (128, 128))
        input_image = frame.reshape(-1, 128, 128, 1)
        input_image = input_image.astype(np.float32)

        # pass input image to model
        input_shape = input_details[0]['shape']
        interpreter.set_tensor(input_details[0]['index'], input_image)

        # run the model
        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # print raw and processed output data
        print(output_data)
        print(types[np.argmax(output_data)])

        # display the input image and the type
        cv2.putText(frame, types[np.argmax(output_data)], (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('img', frame)
        
        cv2.waitKey(0)

    # if there is a url upload error, inform the user.
    except urllib.error.HTTPError:
        print('cannot open image')

# remove windows once close signal recieved
cv2.destroyAllWindows()
