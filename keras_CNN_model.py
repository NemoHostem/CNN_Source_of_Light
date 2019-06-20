# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:01:40 2019
Created on Mon Jun  3 14:14:11 2019

@author: Matias Ijäs
"""

# %% Imports
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Reshape
import numpy as np
import os
import sys
import csv
from imageio import imread
from skimage.transform import rescale
# from PIL import Image


# %% Variables and seed
# Random seed for reproducing
np.random.seed(8)

print("Setting variables.")

train_folder = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/facedataset'
test_folder = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/facetestset'
train_file = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/traindata.csv'
test_file = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/testdata.csv'
save_folder = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/network'
model_name = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/keras_light_direction_trained_model.h5'

batch_size = 32
num_classes = 360
epochs = 10
num_training = 50000
num_validation = 4000
num_testing = 5000

# %% Creating training, validation and testing sets from the data

print("Setting training, validation and testing datasets.")

"""
# train_test_split ?
X_train =  # image in train_folder                      30000, 128, 128, 3
Y_train =  # computed value of angle in image name      30000, num_classes
"""
X_train = np.zeros((num_training, 49152))
Y_train = np.zeros((num_training, num_classes))

"""
X_val =  # image in train_folder                        5000, 128, 128, 3
Y_val =  # computed value of angle in image name        5000, num_classes
"""
X_val = np.zeros((num_validation, 49152))
Y_val = np.zeros((num_validation, num_classes))
"""
X_test =  # image in test_folder                        5000, 128, 128, 3
Y_test_val =  # computed value of angle in image name   5000, num_classes
"""
X_test = np.zeros((num_testing, 49152))
Y_test = np.zeros((num_testing, num_classes))

# %% Setting Image files to correct format

print("Generating data from training and testing files.")

with open(train_file, newline='') as csvfile:
    train_data = list(csv.reader(csvfile))
    
with open(test_file, newline='') as csvfile:
    test_data = list(csv.reader(csvfile))
    
print("Reading training data. \n")
cur_imgs = 0
while cur_imgs < num_training:
    # Read image name / path from given folder
    img_name = train_data[cur_imgs][0]
    img = imread(img_name)
    # Resize the image
    img = np.ravel(rescale(img, 1.0 / 2.0, anti_aliasing=False))
    # Add image data and answer to database
    X_train[cur_imgs] = img
    Y_train[cur_imgs][round(float(train_data[cur_imgs][2]))] = 1
    cur_imgs += 1
    if cur_imgs % 100 == 0:
        sys.stdout.write("\r" + "Training data " + str(cur_imgs*100 / num_training) + " % ready.")
        sys.stdout.flush()
    
print("Reading validation data. \n")
cur_imgs = 0
while cur_imgs < num_validation:
    # Read image name / path from given folder
    img_name = train_data[num_training + cur_imgs][0]
    #img = Image.open(img_name, 'r')
    img = imread(img_name)
    # Resize the image
    img = np.ravel(rescale(img, 1.0 / 2.0, anti_aliasing=False))
    # Add image data and answer to database
    #X_val.append(list(img.getdata()))
    X_val[cur_imgs] = img
    Y_val[cur_imgs][round(float(train_data[num_training + cur_imgs][2]))] = 1
    cur_imgs += 1
    if cur_imgs % 100 == 0:
        sys.stdout.write("\r" + "Validation data " + str(cur_imgs*100 / num_validation) + " % ready.")
        sys.stdout.flush()

print("Reading testing data. \n")
cur_imgs = 0
while cur_imgs < num_testing:
    # Read image name / path from given folder
    img_name = test_data[cur_imgs][0]
    img = imread(img_name)
    # Resize the image
    img = np.ravel(rescale(img, 1.0 / 2.0, anti_aliasing=False))
    # Add image data and answer to database
    X_test[cur_imgs] = img
    Y_test[cur_imgs][round(float(test_data[cur_imgs][2]))] = 1
    cur_imgs += 1
    if cur_imgs % 100 == 0:
        sys.stdout.write("\r" + "Testing data " + str(cur_imgs*100 / num_validation) + " % ready.")
        sys.stdout.flush()


# %% Keras model for CNN

print("Building Keras model.")
model = Sequential()
model.add(Reshape((128,128,3), input_shape=(49152,)))
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

print("Fitting Keras model") 
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size)

# Make predictions
# Y_testset = model.predict(X_test)

# %% Save model and weights

print("Saving Keras model and weights") 
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
model_path = os.path.join(save_folder, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
training_scores = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', training_scores[0])
print('Test accuracy:', training_scores[1])


# %% Testing on single images

real_angles = np.zeros(10)
img_np = np.zeros((10, 49152))
for i in range(10):
    index = np.random.randint(num_testing)
    imagepath = test_data[index][0]
    print(imagepath)
    real_angles[i] = test_data[index][2]
    
    img = imread(imagepath)
    img = np.ravel(rescale(img, 1.0 / 2.0, anti_aliasing=False))
    img_np[i] = img
    
evaluated = model.predict(img_np)
for i in range(10):
    val = np.amax(evaluated[i])
    print('Evaluated:', np.where(evaluated[i] == val),'Percentage:', val, 'Real', real_angles[i])


# %% Possible issues

# Training data is too large - generator / train_on_batch
# Links: https://github.com/keras-team/keras/issues/2708 
# https://stackoverflow.com/questions/52851866/how-to-deal-with-thousands-of-images-for-cnn-training-keras

# Loss and accuracy scores
# Link: https://keras.io/examples/cifar10_cnn/