import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from Modules.Tensorflow.CNN_Layer import CNN
from tensorflow.keras import optimizers as optim
import numpy as np


class visual_obs_critic(Model):
    def __init__(self, args, action_space, learning_rate, device, env_info, hidden):
        super(visual_obs_critic, self).__init__()
        self.learning_rate = learning_rate
        self.action_space = action_space
        self.device = device
        self.env_info = env_info

        self.cnnLayer = CNN(self.env_info)

        weight_init = tf.keras.initializers.RandomNormal()
        self.model = tf.keras.Sequential()
        self.model.add(self.cnnLayer)
        self.model.add(Dense(self.action_space, use_bias=True, kernel_initializer=weight_init))

        self.optimizers = optim.Adam(learning_rate=learning_rate)

        self.model.build(input_shape=[None, 480, 270, 3])

        print(self.model.summary())

    def call(self, obs):
        obs = np.array(obs, dtype=np.float)
        obs = (obs - (255.0 / 2)) / (255.0 / 2)
        x = self.model(obs)
        return x

    def predict(self, obs):
        q_val = self.call(obs)
        return q_val

    def Learn(self, Data):
        loss = Data[0]
        tape = Data[1]
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizers.apply_gradients(zip(gradients, self.trainable_variables))

        return loss.numpy()

    def get_loss(self,  states, td_targets):
        loss = None

        states = np.asarray(states).squeeze(1)

        with tf.GradientTape() as tape:
            predict = self.predict(states)
            loss = tf.reduce_mean(tf.square(td_targets - predict))

        return [loss, tape]



