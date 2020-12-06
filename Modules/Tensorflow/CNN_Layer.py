import tensorflow as tf
from tensorflow.keras import Model
from tensorflow import keras


class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu, padding='SAME',
                                      input_shape=(480, 270, 3)))
        model.add(keras.layers.MaxPool2D(padding='SAME'))
        model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu, padding='SAME'))
        model.add(keras.layers.MaxPool2D(padding='SAME'))
        model.add(keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu, padding='SAME'))
        model.add(keras.layers.MaxPool2D(padding='SAME'))
        model.add(keras.layers.Flatten())

        self.model = model

    def call(self, x):
        x = self.model(x)
        return x