from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import Callback
from tqdm import tqdm
import os
import cv2
from random import shuffle
import matplotlib.pyplot as plt

class TrainingPlot(Callback):

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) == 25:

            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            #plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig('output/Epoch-{}.png'.format(epoch))
            plt.close()

labelled_images = []
for vehicle in ['ship','kangaroo','car']:
    for i in tqdm(os.listdir(vehicle)):
        path = os.path.join(vehicle, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        #img = cv2.imread(path)
        try:
            img = cv2.resize(img, (128,128))
            label = [0,0,0]
            ind = ['ship','kangaroo','car'].index(vehicle)
            label[ind] = 1
            labelled_images.append([img, label])
        except cv2.error:
            print(path)
shuffle(labelled_images)
print(len(labelled_images))

training_images = labelled_images[:500]
testing_images = labelled_images[500:]

tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,128,128,1)

print(tr_img_data[0])
tr_lbl_data = np.array([i[1] for i in training_images])
tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,128,128,1)
tst_lbl_data = np.array([i[1] for i in testing_images])

model = Sequential()

#model.add(InputLayer(input_shape=[128,128,1]))

model.add(Conv2D(filters=32, kernel_size=5,strides=1, padding='same', activation='relu', input_shape=(128,128,1)))
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
plot_losses = TrainingPlot()
model.fit(x=tr_img_data, y=tr_lbl_data, epochs=25, batch_size=100, callbacks=[plot_losses])
model.summary()

model.save('module_4_model')


# # Save tf.keras model in H5 format.
# keras_file = 'keras_model.h5'
# tf.keras.models.save_model(model, keras_file)
#
# # Convert the model.
# converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(keras_file)
# tflite_model = converter.convert()
#
# # Save the model.
# with open('model.tflite', 'wb') as f:
#   f.write(tflite_model)


fig = plt.figure(figsize=(14,14))
for cnt, data in enumerate(testing_images[10:40]):

    y = fig.add_subplot(6,5,cnt+1)
    img = data[0]
    data = img.reshape(1,128,128,1)
    model_out = model.predict([data])

    labels = ['ship','kangaroo','car']
    str_label = labels[np.argmax(model_out)]

    y.imshow(img, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()