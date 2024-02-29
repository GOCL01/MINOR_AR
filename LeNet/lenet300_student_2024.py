# -*- coding: utf-8 -*-
"""
########################################################################################################
########################################################################################################
        
    Author       : GON0
    Organization : Fontys University of Applied Sciences
    Course       : MINOR Adaptive Robotics
    Source Name  : lenet300_student.py
    Description  : Main file for MNIST using Lenet300-100 (dense model)

########################################################################################################
########################################################################################################
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D
seed(1)

########################################################################################################
########################################################################################################
# Activate the following lines for GPU's usage

if False:
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # Currently, memory growth needs to be the same across GPUs
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    tf.config.experimental.set_visible_devices(gpus[0],'GPU')


########################################################################################################
########################################################################################################
#%% Load data  


(x_train, y_train), (x_test, y_test) = mnist.load_data()

#plot image
plt.imshow(x_train[0,:,:])
plt.plot()

#print shape
x_train.shape

"""
TODO1: Reshape x_train and x_test to to 1-D

"""

# normalize inputs from 0-255 to 0-1
"""
TODO2: Normalize x_train and x_test 

"""

# Divide dataset into training and validation
x_val = x_train[-5000:,:]
x_train = x_train[:-5000,:]

# one hot encode outputs
"""
TODO3: One hot encoding of y_train and y_test 

"""


y_val = y_train[-5000:,:]
y_train = y_train[:-5000,:]



########################################################################################################
########################################################################################################
#%% Define model

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense


model = Sequential()

"""
TODO4: Define model 

"""

model.summary()
# Compile model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

"""
TODO5: Train model 

model.fit( -add x data-, -add y data-,
          batch_size=50,
          epochs=10,
          verbose=1,
          validation_data=(x_val, y_val))


"""


"""
TODO6: Test model

results = model.evaluate(-add x data-, -add y data-, batch_size=16)
print(results)


"""

  






