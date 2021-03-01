import learn as learn
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import imutils
import numpy as np
import IPython
import functools
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3

#model = Sequential([
 #   Conv2D(100, (3, 3), activation='relu', input_shape=(150, 150, 3)),
  #  MaxPooling2D(2, 2),

   # Conv2D(100, (3, 3), activation='relu'),
    #MaxPooling2D(2, 2),

    #Flatten(),
    #Dropout(0.9),
    #Dense(63, activation='relu'),
    #Dense(2, activation='softmax')
#])
pre_trained_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(150, 150, 3))

pre_trained_model.summary()

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('Last layer output shape :', last_layer.output_shape)
last_output = last_layer.output

x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
#The Final layer with 3 outputs for 3 categories
x = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.models.Model(inputs=pre_trained_model.input, outputs=x)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

TRAINING_DIR = "C:/Users/silab/Desktop/MaskDetectionThesis/Datasets/observations/experiements/dest_folder/train"
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=10,
                                                    target_size=(150, 150))
VALIDATION_DIR = "C:/Users/silab/Desktop/MaskDetectionThesis/Datasets/observations/experiements/dest_folder/val"
validation_datagen = ImageDataGenerator(rescale=1.0/255)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                         batch_size=10,
                                                         target_size=(150, 150))

checkpoint = ModelCheckpoint(
'model2-{epoch:03d}.model',
 monitor='val_loss',verbose=0,
 save_best_only=True,
 mode='max')

#checkpoint = ModelCheckpoint(
    #filepath='model-{epoch:03d}.ckpt',
    #save_weights_only=True,
    #monitor='val_acc',
    #mode='max',
    #save_best_only=True,
    #verbose=0)




history = model.fit(train_generator,
                    validation_data = validation_generator,
                    epochs=3,
                    callbacks=[checkpoint],
                    batch_size=32,
                    )

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='model-{epoch:03d}.ckpt',
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True,
    verbose=0)




# Variational AutoEncoder (VAE)


print(history.history.keys())

plt.show()

plt.plot(history.history['accuracy'])

plt.plot(history.history['loss'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
#learn.export("2attempt")

#model.save('my_mask.h5')

import cv2
import numpy as np
from keras.models import load_model

face_clsfr = cv2.CascadeClassifier( 'C:/Users/silab/Desktop/Face-Mask-Detection-master/haarcascade_frontalface_default.xml')



labels_dict = {0: 'without_mask', 1: 'with_mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

size = 4
webcam = cv2.VideoCapture(0)  # Use camera 0

# We load the xml file
classifier = cv2.CascadeClassifier('C:/Users/silab/Desktop/Face-Mask-Detection-master/haarcascade_frontalface_default.xml')

while True:
    (rval, im) = webcam.read()
    im = cv2.flip(im, 1, 1)  # Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[0] // size, im.shape[1] // size))
    # mini = cv2.imread('C:/Users/silab/Desktop/Face-Mask-Detection-master/haarcascade_frontalface_default.xml')

    # detect MultiScale / faces
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f]  # Scale the shapesize backup
        # Save just the rectangle faces in SubRecFaces
        face_img = im[y:y + h, x:x + w]
        resized = cv2.resize(face_img, (150, 150))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 150, 150, 3))
        reshaped = np.vstack([reshaped])
        result = model.predict(reshaped)
        print(result)

        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(im, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(im, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(im, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    #  print("I work you stupid")

    # Show the image

    cv2.imshow("LIVE", im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop
    if key == 27:  # The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()
# print("I work you stupid")

