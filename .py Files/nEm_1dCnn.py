# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 07:53:51 2024

@author: Samiran
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
import glob
import scipy.io as sio
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution1D, Dropout, Activation
from keras.optimizers import SGD
from keras.initializers import random_uniform
from sklearn.model_selection import train_test_split
#from keras.layers.convolutional import Conv1D    
from keras.models import Sequential
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input   
from keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

path='C:\\Users\\Samiran\\Downloads\\Dataset\\dataset_emwise\\'
X=np.load(path+'X.npy')
y=np.load(path+'Y.npy')
y=y.astype('int')
nClass=6

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# X_train=tf.keras.utils.normalize(X_train,axis=1)
# X_test=tf.keras.utils.normalize(X_test,axis=1)
X_train=np.expand_dims(X_train,axis=2)
X_test=np.expand_dims(X_test,axis=2)
sequence_length=X.shape[1]
num_classes=6


# Define the 1D CNN model
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv1D(16, kernel_size=3, activation='tanh', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        # layers.Conv1D(16, kernel_size=3, activation='tanh', input_shape=input_shape),
        # layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(32, kernel_size=3, activation='tanh', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(32, kernel_size=3, activation='tanh'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='tanh'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='tanh'),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Create the CNN model
input_shape = (sequence_length, 1)
model = create_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=30, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')