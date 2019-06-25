# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:01:40 2019
Last Modified on Tue Jun  25 08:22:11 2019

@author: Matias Ijäs
"""

# %% Imports
from keras.models import Sequential, Model
from keras.layers import Flatten, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense # , Reshape
from keras.optimizers import Adam

import numpy as np
import os
import sys
import csv
from imageio import imread
from skimage.transform import rescale
import matplotlib.pyplot as plt
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
model_name = 'keras_light_direction_regr_model5.h5'

batch_size = 64
num_classes = 1 # 360 if using classification, 1 if using regression
epochs = 45
num_training = 10000
num_validation = 1000
num_testing = 1000
w, h, d = 64, 64, 3


# %% Creating training, validation and testing sets from the data

print("Setting training, validation and testing datasets.")

"""
# train_test_split ?
X_train =  # image in train_folder                      30000, 128, 128, 3
Y_train =  # computed value of angle in image name      30000, num_classes
"""
X_train = np.zeros((num_training, w*h*d)) # 12288 or 49152))
Y_train = np.zeros((num_training, num_classes))

"""
X_val =  # image in train_folder                        5000, 128, 128, 3
Y_val =  # computed value of angle in image name        5000, num_classes
"""
X_val = np.zeros((num_validation, w*h*d)) # 49152))
Y_val = np.zeros((num_validation, num_classes))
"""
X_test =  # image in test_folder                        5000, 128, 128, 3
Y_test_val =  # computed value of angle in image name   5000, num_classes
"""
X_test = np.zeros((num_testing, w*h*d)) # 49152))
Y_test = np.zeros((num_testing, num_classes))


# %% Setting Image files to correct format

print("Generating data from training and testing files.")

with open(train_file, newline='') as csvfile:
    train_data = list(csv.reader(csvfile))
    
with open(test_file, newline='') as csvfile:
    test_data = list(csv.reader(csvfile))
    
print("\nReading training data. \n")
cur_imgs = 0
while cur_imgs < num_training:
    # Read image name / path from given folder
    img_name = train_data[cur_imgs][0]
    img = imread(img_name)
    # Resize the image
    img = np.ravel(rescale(img, 1.0 / float(256.0/w), anti_aliasing=False))
    img = img / 255.0
    # Add image data and answer to database
    X_train[cur_imgs] = img
    # Y_train[cur_imgs][round(float(train_data[cur_imgs][2]))] = 1
    Y_train[cur_imgs] = float(train_data[cur_imgs][2])
    cur_imgs += 1
    if cur_imgs % 100 == 0:
        sys.stdout.write("\r" + "Training data " + str(round(cur_imgs*100 / num_training,2)) + " % ready.")
        sys.stdout.flush()
    
print("\nReading validation data. \n")
cur_imgs = 0
while cur_imgs < num_validation:
    # Read image name / path from given folder
    img_name = train_data[num_training + cur_imgs][0]
    img = imread(img_name)
    # Resize the image
    img = np.ravel(rescale(img, 1.0 / float(256.0/w), anti_aliasing=False))
    img = img / 255.0
    # Add image data and answer to database
    X_val[cur_imgs] = img
    # Y_val[cur_imgs][round(float(train_data[num_training + cur_imgs][2]))] = 1
    Y_val[cur_imgs] = float(train_data[num_training + cur_imgs][2])
    cur_imgs += 1
    if cur_imgs % 100 == 0:
        sys.stdout.write("\r" + "Validation data " + str(round(cur_imgs*100 / num_validation,2)) + " % ready.")
        sys.stdout.flush()

print("\nReading testing data. \n")
cur_imgs = 0
while cur_imgs < num_testing:
    # Read image name / path from given folder
    img_name = test_data[cur_imgs][0]
    img = imread(img_name)
    # Resize the image
    img = np.ravel(rescale(img, 1.0 / float(256.0/w), anti_aliasing=False))
    img = img / 255.0
    # Add image data and answer to database
    X_test[cur_imgs] = img
    # Y_test[cur_imgs][round(float(test_data[cur_imgs][2]))] = 1
    Y_test[cur_imgs] = float(test_data[cur_imgs][2])
    cur_imgs += 1
    if cur_imgs % 100 == 0:
        sys.stdout.write("\r" + "Testing data " + str(round(cur_imgs*100 / num_testing,2)) + " % ready.")
        sys.stdout.flush()

# Scale angles from -180 deg - 180 deg to [0, 1]
max_angle = 180
min_angle = -180
Y_train = (Y_train-min_angle)/(max_angle-min_angle)
Y_val = (Y_val-min_angle)/(max_angle-min_angle)
Y_test = (Y_test-min_angle)/(max_angle-min_angle)

X_train = X_train.reshape((num_training, 64, 64, 3))
X_val = X_val.reshape((num_validation, 64, 64, 3))
X_test = X_test.reshape((num_testing, 64, 64, 3))


# %% Keras classification model for CNN

"""
print("\nBuilding Keras classification model.")
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
"""

# %% Keras regression model for CNN

# Create_mlp and create_cnn functions copied from
# https://www.pyimagesearch.com/2019/01/21/regression-with-keras/
# Some modifications could be made

print("\nBuilding Keras regression model.")

def create_mlp(dim, regress=False):

    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))
 
    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))
 
    return model

def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    input_shape = (height, width, depth, )
    chan_dim = -1
    # define the model input
    inputs = Input(shape=input_shape)
 
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        if i == 0:
            x = inputs
         
        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(360)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chan_dim)(x)
    x = Dropout(0.25)(x)
 
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(36)(x)
    x = Activation("relu")(x)
    x = Dropout(0.25)(x)
 
    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)
 
    # construct the CNN
    model = Model(inputs, x)
 
    # return the CNN
    return model

model = create_cnn(64, 64, 3, regress=True)
opt = Adam(lr=1e-4, decay=1e-4 / 200)
model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])

# Training
model_history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size)


# %% Statistical part mean and std

# Predictions
preds = model.predict(X_test)

diff = np.abs(preds.flatten() - Y_test.flatten())
 
# Compute the mean and standard deviation of the absolute percentage difference
mean = np.mean(diff)
std = np.std(diff)

# Print statistics
print("Average angle: {}, std angle: {}".format(Y_test.mean(), Y_test.std()))
print("mean: {:.2f}%, std: {:.2f}%".format(mean, std))


# %% Save model and weights

print("Saving Keras model and weights") 
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
model_path = os.path.join(save_folder, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


# %% Cheats about model

model.summary()
model.get_config()
model.get_weights()


# %% Visualize training history of a model

print("Visualizing losses in history diagram")
print(model_history.history.keys())
# "Loss"
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()


# %% Score trained model.

"""
training_scores = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', training_scores[0])
print('Test accuracy:', training_scores[1])
"""


# %% Testing on single images from test data

def evaluate_accuracies(acc_count, sum_angle, est_val, real_val):
    
    err_val = abs(real_val - est_val)
    sum_angle += err_val
    
    if err_val < 1:
        acc_count[0] += 1
    elif err_val < 5:
        acc_count[1] += 1
    elif err_val < 10:
        acc_count[2] += 1
    elif err_val < 30:
        acc_count[3] += 1
    elif err_val < 90:
        acc_count[4] += 1
    else:
        acc_count[5] += 1
        
    return acc_count, sum_angle

def do_tests_test_file(num_tests):
    
    # Accuracy (in degrees) <=1, <=5, <=10, <=30, <=90, >90
    accuracies_count = [0, 0, 0, 0, 0, 0]
    sum_angle_error = 0
    real_angles = np.zeros(num_tests)
    img_np = np.zeros((num_tests, 64, 64, 3))
    
    for i in range(num_tests):
        index = np.random.randint(num_testing)
        imagepath = test_data[index][0]
        # print(imagepath)
        real_angles[i] = (float(test_data[index][2]) - min_angle) / (max_angle - min_angle)
        
        img = imread(imagepath)
        img = rescale(img, 1.0 / 4.0, anti_aliasing=False)
        img = img / 255.0
        img_np[i] = img.reshape((64, 64, 3))
        
    # Used only in regression
    # img_np = img_np.reshape((num_tests, 128, 128, 3))
    
    evaluated = model.predict(img_np)
    
    for i in range(num_tests):
        est_val = evaluated[i] * (max_angle - min_angle) + min_angle
        real_val = (real_angles[i]) * (max_angle - min_angle) + min_angle
        print('Evaluated:', est_val, ', Real:', real_val)
        accuracies_count, sum_angle_error = evaluate_accuracies(accuracies_count, sum_angle_error, est_val, real_val)
        
    print("Accuracy count, <1, <5, <10, <30, <90, >90")
    print(accuracies_count)
    print("Average angle error: ", sum_angle_error / num_tests)
    
    return evaluated


# Make predictions
num_tests = 100
ev = do_tests_test_file(num_tests)


# %% Testing on separate real image dataset

def do_tests_single_file(img):
        
    img = img.reshape(1,-1)
    evaluated = model.predict(img)
    val = np.amax(evaluated)
    print('Evaluated:', np.where(evaluated == val),'Percentage:', val)
        
def read_files_from_folder(folder):
    
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            
            if (file.endswith('.JPG') or file.endswith('.JPEG') or file.endswith('.jpg') or file.endswith('.jpeg')):
                print(file)
                filename = folder + '/' + file
                
                img = imread(filename)
                img = np.ravel(np.resize(img, (128, 128, 3)))
                do_tests_single_file(img)
  
"""
img_folder = 'C://Users/Matias Ijäs/Documents/Matias/data/realimgs'
read_files_from_folder(img_folder)
"""


# %% Possible issues

# Training data is too large - generator / train_on_batch
# Links: https://github.com/keras-team/keras/issues/2708 
# https://stackoverflow.com/questions/52851866/how-to-deal-with-thousands-of-images-for-cnn-training-keras

# Loss and accuracy scores do not behave right?
# Link: https://keras.io/examples/cifar10_cnn/

# Unstable validation loss - Lower the learning rate
# Learning rate too high - Lower the learning rate

# Network expected different size of input - Reshape the input to wanted

# Network gives same solution to all test examples -
# Predict gives values out of range [0, 1] - Train with similar data than test