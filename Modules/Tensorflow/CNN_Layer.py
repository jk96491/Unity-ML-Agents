import tensorflow as tf
from tensorflow.keras import Model
from tensorflow import keras
import numpy as np


class CNN(Model):
    def __init__(self, env_info):
        super(CNN, self).__init__()

        self.shape = np.shape(env_info.visual_observations[0])
        self.width = self.shape[1]
        self.height = self.shape[2]
        self.channel = self.shape[3]

        model = keras.Sequential()
        model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu, padding='SAME',
                                      input_shape=(self.width, self.height, self.channel)))
        model.add(keras.layers.MaxPool2D(padding='SAME'))
        model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu, padding='SAME'))
        model.add(keras.layers.MaxPool2D(padding='SAME'))
        model.add(keras.layers.Flatten())

        self.model = model

    def call(self, x):
        x = self.model(x)
        return x