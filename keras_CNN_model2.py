# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 08:21:40 2019
Last Modified on Tue Jul 16 08:22:11 2019

@author: Matias Ijäs
"""

# %% Imports
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense # , Reshape
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import models

import numpy as np
import os
import sys
import csv
import math
from imageio import imread
from skimage.transform import rescale
import matplotlib.pyplot as plt
# from PIL import Image


# %% Variables and seed

print("Setting variables.")

train_folder = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/face_PV_train_gray'
test_folder = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/face_PV_test_gray'
val_folder = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/face_PV_val_gray'
train_file = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/face_PV_train_gray.csv'
test_file = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/face_PV_test_gray.csv'
val_file = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/face_PV_val_gray.csv'
save_folder = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/network'
model_name = 'keras_regr_model_gray5.h5'
model_checkpoint_name = 'keras_regr_model_gray5_bw.h5'
model_checkpoint = 'keras_weights.hdf5'

batch_size = 64
num_classes = 1 # 360 if using classification, 1 if using regression
epochs = 40
num_training = 20000
num_validation = 5000
num_testing = 5000
w, h, d = 256, 256, 1


# %% Creating training, validation and testing sets from the data

print("Setting training, validation and testing datasets.")

"""
X_train =  # image in train_folder                      30000, 128, 128, 3
Y_train =  # computed value of angle in image name      30000, num_classes
"""
X_train = np.zeros((num_training, w*h*d))
Y_train = np.zeros((num_training, num_classes))

"""
X_val =  # image in train_folder                        5000, 128, 128, 3
Y_val =  # computed value of angle in image name        5000, num_classes
"""
X_val = np.zeros((num_validation, w*h*d))
Y_val = np.zeros((num_validation, num_classes))
 
"""
X_test =  # image in test_folder                        5000, 128, 128, 3
Y_test_val =  # computed value of angle in image name   5000, num_classes
"""
X_test = np.zeros((num_testing, w*h*d))
Y_test = np.zeros((num_testing, num_classes))

# %% Functions used

def mask_it(img, mmin, mmax):
    
    n_img = img.copy()
    xm, ym, zm = n_img.shape
    for x in range(0,xm):
        for y in range(0,ym):
            for z in range(0,zm):
                if n_img[x,y,z] < mmin or n_img[x,y,z] > mmax:
                    n_img[x,y,z] = 0

    return n_img

def read_and_mask_image(filename):
    img = plt.imread(filename)
    m_img = mask_it(img, 150, 256)
    
    """
    plt.figure()
    plt.imshow(m_img)
    """
    return m_img
    
def read_from_folder(folder):
    
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            
            if (file.endswith('.JPG') or file.endswith('.JPEG') or file.endswith('.jpg') or file.endswith('.jpeg')):
                print(file)
                filename = folder + '/' + file
                _ = read_and_mask_image(filename)

# %% Setting Image files to correct format

print("Generating data from training and testing files.")

with open(train_file, newline='') as csvfile:
    train_data = list(csv.reader(csvfile))
    
with open(val_file, newline='') as csvfile:
    val_data = list(csv.reader(csvfile))
   
with open(test_file, newline='') as csvfile:
    test_data = list(csv.reader(csvfile))
  
print("\nReading training data. \n")
cur_imgs = 0
while cur_imgs < num_training:
    # Read image name / path from given folder
    img_name = train_data[cur_imgs][0]
    img = plt.imread(img_name)
    #img = read_and_mask_image(img_name)
    # Resize the image
    img = np.ravel(rescale(img, 1.0 / float(256.0/w), anti_aliasing=False))
    img = img / 255.0
    # Add image data and answer to database
    X_train[cur_imgs] = img
    Y_train[cur_imgs] = float(train_data[cur_imgs][2])
    cur_imgs += 1
    if cur_imgs % 100 == 0:
        sys.stdout.write("\r" + "Training data " + str(round(cur_imgs*100 / num_training,2)) + " % ready.")
        sys.stdout.flush()
    
print("\nReading validation data. \n")
cur_imgs = 0
while cur_imgs < num_validation:
    # Read image name / path from given folder
    img_name = val_data[cur_imgs][0]
    img = plt.imread(img_name)
    #img = read_and_mask_image(img_name)
    # Resize the image
    img = np.ravel(rescale(img, 1.0 / float(256.0/w), anti_aliasing=False))
    img = img / 255.0
    # Add image data and answer to database
    X_val[cur_imgs] = img
    Y_val[cur_imgs] = float(val_data[cur_imgs][2])
    cur_imgs += 1
    if cur_imgs % 100 == 0:
        sys.stdout.write("\r" + "Validation data " + str(round(cur_imgs*100 / num_validation,2)) + " % ready.")
        sys.stdout.flush()

print("\nReading testing data. \n")
cur_imgs = 0
while cur_imgs < num_testing:
    # Read image name / path from given folder
    img_name = test_data[cur_imgs][0]
    img = plt.imread(img_name)
    #img = read_and_mask_image(img_name)
    # Resize the image
    img = np.ravel(rescale(img, 1.0 / float(256.0/w), anti_aliasing=False))
    img = img / 255.0
    # Add image data and answer to database
    X_test[cur_imgs] = img
    Y_test[cur_imgs] = float(test_data[cur_imgs][2])
    cur_imgs += 1
    if cur_imgs % 100 == 0:
        sys.stdout.write("\r" + "Testing data " + str(round(cur_imgs*100 / num_testing,2)) + " % ready.")
        sys.stdout.flush()

# %%
# Scale angles from [-180 deg, 180 deg] to [0, 1]
max_angle = 180
min_angle = -180

Y_train = (Y_train-min_angle)/(max_angle-min_angle)
Y_val = (Y_val-min_angle)/(max_angle-min_angle)
Y_test = (Y_test-min_angle)/(max_angle-min_angle)

X_train = X_train.reshape((num_training, w, h, d))
X_val = X_val.reshape((num_validation, w, h, d))
X_test = X_test.reshape((num_testing, w, h, d))


# %% Keras regression model for CNN

# Create_mlp and create_cnn functions copied from
# https://www.pyimagesearch.com/2019/01/21/regression-with-keras/
# Some modifications could be made

print("\nBuilding Keras regression model.")

checkpointer = ModelCheckpoint(filepath=model_checkpoint, 
                               monitor = 'val_loss', verbose=1, save_best_only=True)

def create_mlp(width, height, depth, filters=(16,32,48,64,80), regress=False):

    model = Sequential()
    
    model.add(Conv2D(16, (5, 5), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    for (i, f) in enumerate(filters):
        model.add(Conv2D(f, (3, 3), padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(300, activation="relu"))
    model.add(Dense(36, activation="relu"))
 
    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))
 
    return model

def create_cnn(width, height, depth, filters=(16, 32, 32, 64, 64), regress=False):
    
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

model = create_mlp(w, h, d, regress=True) # create_mlp(w, h, d, regress=True)
opt = Adam(lr=1e-4, decay=1e-4 / 400)
model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])

# Training
model_history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, callbacks=[checkpointer])


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

# Load best weights and save model
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
best_model_path = os.path.join(save_folder, model_checkpoint_name)
model.load_weights(model_checkpoint)
model.save(best_model_path)
print('Saved trained model at %s ' % best_model_path)


# %% Load a model

"""
load_model_folder = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/network'
load_model_file = 'keras_regr_model_gray4.h5' # 'keras_light_direction_regr_model10.h5'
model = load_model(load_model_folder + '/' + load_model_file)
"""

# %% Statistics about model structure

model.summary()

# %% Weights and cofiguration
"""
model.get_config()
model.get_weights()
"""

# %% Visualizing intermediate activations of convnet

"""
# This part is mostly copied from https://github.com/gabrielpierobon/cnnshapes/blob/master/README.md
img = img.reshape((1, w, h, d))
layer_outputs = [layer.output for layer in model.layers[:12]] 
# Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) 
# Creates a model that will return these outputs, given the model input

activations = activation_model.predict(img) 
# Returns a list of five Numpy arrays: one array per layer activation

first_layer_activation = activations[0]
print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')


layer_names = []
for layer in model.layers[:12]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
images_per_row = 16


for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
"""

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
    
    if real_val < -90 and est_val > 90:
        err_val = abs(real_val + 360 - est_val)
    elif real_val > 90 and est_val < -90:
        err_val = abs(est_val + 360 - real_val)
    else:
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
        print(est_val, real_val)
        
    return acc_count, sum_angle

def do_tests_test_file(num_tests):
    
    # Accuracy (in degrees) <=1, <=5, <=10, <=30, <=90, >90
    accuracies_count = [0, 0, 0, 0, 0, 0]
    sum_angle_error = 0
    real_angles = np.zeros(num_tests)
    img_np = np.zeros((num_tests, w, h, d))
    indices = np.zeros(num_tests)
    
    for i in range(num_tests):
        index = np.random.randint(0,num_testing)
        imagepath = test_data[index][0]
        # print(imagepath)
        real_angles[i] = (float(test_data[index][2]) - min_angle) / (max_angle - min_angle)
        
        img = imread(imagepath)
        img = rescale(img, 1.0 / float(256.0/w), anti_aliasing=False)
        img = img / 255.0
        img_np[i] = img.reshape((w, h, d))
        indices[i] = index
        
    # Used only in regression
    # img_np = img_np.reshape((num_tests, 128, 128, 3))
    
    evaluated = model.predict(img_np)
    last_sum_angle = 0
    
    for i in range(num_tests):
        est_val = evaluated[i] * (max_angle - min_angle) + min_angle
        real_val = (real_angles[i]) * (max_angle - min_angle) + min_angle
        # print('Evaluated:', est_val, ', Real:', real_val)
        last_sum_angle = float(sum_angle_error)
        accuracies_count, sum_angle_error = evaluate_accuracies(accuracies_count, sum_angle_error, est_val, real_val)
        if last_sum_angle + 90 < sum_angle_error:
            print("Over 90 at: ", i)
        
    print("Accuracy count, <1, <5, <10, <30, <90, >90")
    print(accuracies_count)
    print("Average angle error: ", sum_angle_error / num_tests)
    
    return evaluated, indices


# Make predictions
num_tests = 100
ev = do_tests_test_file(num_tests)


# %% Test and visualize a single image and estimated light direction

test_data_index = np.random.randint(0,num_testing)
sz = 256
img = plt.imread(test_data[test_data_index][0])
fig = plt.figure()
ax = fig.add_subplot(111)
# Plot the image as background
ax.imshow(img, extent=[-sz/2,sz/2,-sz/2,sz/2], cmap='gray')

img2 = rescale(img, 1.0 / float(256.0/w), anti_aliasing=False, anti_aliasing_sigma=None)
img2 = img2 / 255.0
img2 = img2.reshape((1, w, h, d))

# Compute evaluated angle and get real angle        
evaluated_angle = model.predict(img2)
evaluated_angle = evaluated_angle * (max_angle - min_angle) + min_angle
real_angle = float(test_data[test_data_index][2])

# Prepare x and y vectors to draw
if evaluated_angle < 90 and evaluated_angle > -90:
    x_range_e = np.array(range(0, int(sz/2)))
else:
    x_range_e = -1 * np.array(range(0, int(sz/2)))
    
if real_angle < 90 and real_angle > -90:
    x_range_r = np.array(range(0, int(sz/2)))  
else:
    x_range_r = -1 * np.array(range(0, int(sz/2)))

print("Estimated:", evaluated_angle, "Real:", real_angle)
y_range_e = np.array(math.tan((evaluated_angle / 180) * math.pi) * (x_range_e-0))
y_range_r = np.array(math.tan((real_angle / 180) * math.pi) * (x_range_r-0))

for i, y in enumerate(y_range_e):
    if y > sz / 2:
        y_range_e[i] = sz/2
        break
    elif y < -sz / 2:
        y_range_e[i] = -sz/2
        break
y_range_e = y_range_e[0:i+1]
x_range_e = x_range_e[0:i+1]

for i, y in enumerate(y_range_r):
    if y > sz / 2:
        y_range_r[i] = sz/2
        break
    elif y < -sz / 2:
        y_range_r[i] = -sz/2
        break
y_range_r = y_range_r[0:i+1]
x_range_r = x_range_r[0:i+1]
    
# Draw Estimated and Real lines on top of image
ax.plot(x_range_e, y_range_e, '--', linewidth=3, color='blue', label='Estimated')
ax.plot(x_range_r, y_range_r, '--', linewidth=3, color='green', label='Real')

ax.legend()


# %% Testing on separate real image dataset

def do_tests_single_file(img):
        
    # img = img.reshape(1,-1)
    evaluated = model.predict(img)
    val = np.amax(evaluated)
    print('Evaluated:', np.where(evaluated == val),'Percentage:', val)
        
def read_files_from_folder(folder):
    
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            
            if (file.endswith('.JPG') or file.endswith('.JPEG') or file.endswith('.jpg') or file.endswith('.jpeg')):
                print(file)
                filename = folder + '/' + file
                
                img = read_and_mask_image(filename)
                img = np.ravel(np.resize(img, (w, h, d)))
                img = img / 255.0
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