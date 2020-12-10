import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from Modules.Tensorflow.CNN_Layer import CNN
from tensorflow.keras import optimizers as optim
import numpy as np


class visual_obs_actor(Model):
    def __init__(self, args,  action_space, learning_rate, device, env_info, hidden):
        super(visual_obs_actor, self).__init__()
        self.learning_rate = learning_rate
        self.action_space = action_space
        self.device = device
        self.env_info = env_info

        self.cnnLayer = CNN(self.env_info)

        weight_init = tf.keras.initializers.RandomNormal()
        self.model = tf.keras.Sequential()
        self.model.add(self.cnnLayer)
        self.model.add(Dense(self.action_space, use_bias=True, kernel_initializer=weight_init, activation=tf.nn.softmax))

        self.optimizers = optim.Adam(learning_rate=learning_rate)

        self.model.build(input_shape=[None, 480, 270, 3])

        print(self.model.summary())

    def call(self, obs):
        obs = np.array(obs, dtype=np.float)
        obs = (obs - (255.0 / 2)) / (255.0 / 2)
        x = self.model(obs)
        return x

    def get_action(self, obs):
        q_val = self.call(obs)
        action = tf.argmax(q_val[0]).numpy()
        return action

    def Learn(self, obs, actions, advantages):
        obs = np.asarray(obs).squeeze(1)
        advantages = np.asarray(advantages).squeeze(1)
        actions = np.asarray(actions).squeeze(1)

        advantages = tf.gather(advantages, actions, axis=1)
        loss = None
        with tf.GradientTape() as Tape:
            policy = self.call(obs)
            log_policy = tf.math.log(policy)

            loss = tf.reduce_mean(- log_policy * advantages)

        gradients = Tape.gradient(loss, self.trainable_variables)
        self.optimizers.apply_gradients(zip(gradients, self.trainable_variables))

        return loss.numpy()



