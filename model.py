import tensorflow as tf
import numpy as np
import matplotlib as plt
import wave, math

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, LeakyReLU, Conv2D, UpSampling2D, Reshape, BatchNormalization, Dropout

def printTFVersion():
    print(tf.__version__)

class DiscriminativeModel(Model):
    def __init__(self):
        super(DiscriminativeModel, self).__init__()
        self.conv2d = Conv2D(64, kernel_size=5, padding='same')
        self.activation = Activation(LeakyReLU(0.2))
        self.dropout = Dropout(0.3)
        self.conv2d1 = Conv2D(128, kernel_size=5, strides=2, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.conv2d(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2d1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


class GenerativeModel(Model):
    def __init__(self):
        super(GenerativeModel, self).__init__()
        self.dense = Dense(128*7*7, input_dim=100, activation=LeakyReLU(0.2))
        self.batch = BatchNormalization()
        self.reshape = Reshape((7, 7, 128))
        self.upS = UpSampling2D()
        self.conv2d = Conv2D(64, kernel_size=5, padding='same')
        self.activation = Activation(LeakyReLU(0.2))
        self.conv2d1 = Conv2D(1, kernel_size=5, padding='same', activation='tanh')

    def call(self, x):
        x = self.dense(x)
        x = self.batch(x)
        x = self.reshape(x)
        x = self.upS(x)
        x = self.conv2d(x)
        x = self.batch(x)
        x = self.activation(x)
        x = self.conv2d1(x)
        return x

EPOCHS = 100

modelD = DiscriminativeModel()
modelG = GenerativeModel()

modelD.compile(loss='binary_crossentropy', optimizer='adam')
modelG.compile(loss='binary_crossentropy', optimizer='adam')

modelD.trainable = False

#modelD + modelG = modelC
ginput = tf.keras.layers.Input(shape=(100,))
doutput = modelD(modelG(ginput))
modelC = tf.keras.Model(ginput, doutput)
modelC.compile(loss='binary_crossentropy', optimizer='adam')

modelD.save()
modelG.save()

def trainGan():
    x_train