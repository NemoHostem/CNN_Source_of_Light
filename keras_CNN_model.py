# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:01:40 2019
Last Modified on TODO

@author: Matias Ijäs
"""

# %% Imports
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import numpy
import os
from PIL import Image


# %% Variables and seed
# Random seed for reproducing
numpy.random.seed(8)

train_folder = 'results/facedataset'
test_folder = 'results/facetestset'
save_folder = 'results/network'
model_name = 'keras_light_direction_trained_model.h5'

batch_size = 32
num_classes = 360
epochs = 10


# %% Setting Image files to correct format

while cur_imgs < num_imgs:
    # Read image name / path from given folder
    img_name = 'TODO'
    img = Image.open(img_name)
    arr = array(img)


# %% Creating training, validation and testing sets from the data
"""
# train_test_split ?
X_train =  # image in train_folder                      30000, 128, 128, 3
Y_train =  # computed value of angle in image name      30000, num_classes
"""
X_train = 
Y_train = 

"""
X_val =  # image in train_folder                        10000, 128, 128, 3
Y_val =  # computed value of angle in image name        10000, num_classes
"""
X_val = 
Y_val =
"""
X_test =  # image in test_folder                        5000, 128, 128, 3
Y_test_val =  # computed value of angle in image name   5000, num_classes
"""
X_test = 
Y_test_val = 


# %% Keras model for CNN

model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', data_format=None))
model.add(Conv2D(48, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', data_format=None))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', data_format=None))
model.add(Dropout(0.25))
model.add(Conv2D(96, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', data_format=None))
model.add(Flatten())
model.add(Dense(1024, activation='softmax'))
model.add(Dense(num_classes, activation='sigmoid'))
 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size)

# Make predictions
# Y_test = model.predict(X_test)

# Save model and weights
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
model_path = os.path.join(save_folder, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
training_scores = model.evaluate(X_test, Y_test_val, verbose=1)
print('Test loss:', training_scores[0])
print('Test accuracy:', training_scores[1])


# %% Possible issues

# Training data is too large - generator / train_on_batch
# Links: https://github.com/keras-team/keras/issues/2708 
# https://stackoverflow.com/questions/52851866/how-to-deal-with-thousands-of-images-for-cnn-training-keras

# Loss and accuracy scores
# Link: https://keras.io/examples/cifar10_cnn/