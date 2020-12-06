import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from Modules.Tensorflow.CNN_Layer import CNN
from tensorflow.keras import optimizers as optim
import numpy as np


class visual_obs_dqn(Model):
    def __init__(self, action_space, learning_rate, device):
        super(visual_obs_dqn, self).__init__()

        self.learning_rate = learning_rate
        self.action_space = action_space
        self.device = device

        self.cnnLayer = CNN()

        weight_init = tf.keras.initializers.RandomNormal()
        self.model = tf.keras.Sequential()
        self.model.add(self.cnnLayer)
        self.model.add(Dense(512, use_bias=True, kernel_initializer=weight_init, activation="relu"))
        self.model.add(Dense(128, use_bias=True, kernel_initializer=weight_init, activation="relu"))
        self.model.add(Dense(self.action_space, use_bias=True, kernel_initializer=weight_init))

        self.optimizers = optim.Adam(learning_rate=0.001)

        self.model.build(input_shape=[None, 480, 270, 3])

    def call(self, x):
        x = np.array(x, dtype=np.float)
        x = self.model(x)
        return x

    def get_action(self, obs):
        q_val = self.call(obs)
        action = tf.argmax(q_val[0]).numpy()
        return action

    def Learn(self, target_model, train_batch, dis):

        loss = []
        for state, action, reward, next_state, done in train_batch:
            with tf.GradientTape() as Tape:
                q_val = self.call(state)
                target_q_val = target_model.call(state).numpy()

                maxQ1 = np.amax(q_val)
                action = np.asscalar(action)

                if done:
                    target_q_val[0][action] = reward
                else:
                    target_q_val[0][action] = reward + maxQ1 * dis

                loss.append(tf.reduce_mean(tf.square(q_val - target_q_val)))

            gradients = Tape.gradient(loss, self.trainable_variables)
            self.optimizers.apply_gradients(zip(gradients, self.trainable_variables))

        return tf.reduce_mean(loss)



