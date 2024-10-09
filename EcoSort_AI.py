import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers 


data_train_path = r'C:\Users\Om Chaudhari\OneDrive\Desktop\Project SEM V\Dataset\TRAIN'
data_test_path = r'C:\Users\Om Chaudhari\OneDrive\Desktop\Project SEM V\Dataset\TEST'

img_width = 180
img_height = 180

data_train = tf.keras.utils.image_dataset_from_directory(
    data_train_path,
    shuffle=True,
    image_size=32,
    validation_split=False)

#print(data_train.class_names)
data_cat = data_train.class_names

data_test = tf.keras.utils.image_dataset_from_directory(
    data_test_path,
    shuffle=True,
    image_size=32,
    validation_split=False)

#print(data_test.class_names)
data_cat = data_test.class_names

from tensorflow.keras.models import Sequential
model = Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(128),
    layers.Dense(len(data_cat))
])

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
epochs_size = 25
history = model.fit(data_train, epochs=epochs_size)



image ='organic.jpg'
image = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
img_arr = tf.keras.utils.array_to_img(image)
img_bat = tf.expand_dims (img_arr, 0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)

print('Waste in image is {} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)],np.max(score)*100))

