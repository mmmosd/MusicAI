import tensorflow as tf
import numpy as np
import matplotlib as plt
import wave, math

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, LeakyReLU, Conv2D, UpSampling2D, Reshape, BatchNormalization, Dropout

def printTFVersion():
    print(tf.__version__)

Generator = tf.keras.Sequential([
    tf.keras.layers.Input(256,30),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(784, activation='sigmoid')
])

Discriminator = tf.keras.Sequential([
    tf.keras.layers.Input(784),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

Doptimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
Goptimizer = tf.keras.optimizers.Adam(learning_rate=0.001)