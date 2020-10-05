from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from tqdm import tqdm
import os
import cv2
from random import shuffle
import matplotlib.pyplot as plt



labelled_images = []
for vehicle in ['car','ship','plane']:
    for i in tqdm(os.listdir(vehicle)):
        path = os.path.join(vehicle, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        try:
            img = cv2.resize(img, (128,128))
            label = [0,0,0]
            ind = ['car', 'ship', 'plane'].index(vehicle)
            label[ind] = 1
            labelled_images.append([img, label])
        except cv2.error:
            print(path)
shuffle(labelled_images)
print(len(labelled_images))

training_images = labelled_images[:500]
testing_images = labelled_images[500:]

tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,128,128,1)
tr_lbl_data = np.array([i[1] for i in training_images])
tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,128,128,1)
tst_lbl_data = np.array([i[1] for i in testing_images])

model = Sequential()

model.add(InputLayer(input_shape=[128,128,1]))

model.add(Conv2D(filters=32, kernel_size=5,strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Conv2D(filters=50, kernel_size=5,strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Conv2D(filters=80, kernel_size=5,strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(3, activation='softmax'))
optimiser = Adam(lr=1e-3)

model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=tr_img_data, y=tr_lbl_data, epochs=50, batch_size=100)
model.summary()


fig = plt.figure(figsize=(14,14))
for cnt, data in enumerate(testing_images[10:40]):

    y = fig.add_subplot(6,5,cnt+1)
    img = data[0]
    data = img.reshape(1,128,128,1)
    model_out = model.predict([data])

    labels = ['car','ship','plane']
    str_label = labels[np.argmax(model_out)]

    y.imshow(img, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()