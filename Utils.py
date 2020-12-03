import numpy as np
import torch
import yaml
from types import SimpleNamespace as SN
from Modules.Discrete import Actor
from Modules.Discrete import Critic
from Modules.Discrete import DQN


def get_state_by_visual(data):
    data = np.uint8(255 * np.array(data))

    state = []
    for i in range(data.shape[3]):
        state.append(data[:, :, :, i])

    return np.array(state).reshape(data.shape[0], data.shape[3], data.shape[1], data.shape[2])


def advantage_td_target(reward, v_value, next_v_value, done, GAMMA):
    if done:
        y_k = v_value
        advantage = y_k - v_value
    else:
        y_k = reward + GAMMA * next_v_value
        advantage = y_k - v_value

    return advantage, y_k


def unpack_batch(batch):
    unpack = batch[0]
    for idx in range(len(batch) - 1):
        unpack = np.append(unpack, batch[idx + 1], axis=0)

    return unpack


def get_discrete_actor(state_dim, action_dim, ACTOR_LEARNING_RATE, device):
    if state_dim is None:
        actor = Actor.visual_obs_actor(action_dim, ACTOR_LEARNING_RATE, device)
    else:
        actor = Actor.vector_obs_actor(state_dim, action_dim, ACTOR_LEARNING_RATE, device)

    return actor


def get_discrete_critic(state_dim, action_dim, ACTOR_LEARNING_RATE, device):
    if state_dim is None:
        critic = Critic.visual_obs_critic(action_dim, ACTOR_LEARNING_RATE, device)
    else:
        critic = Critic.vector_obs_critic(state_dim, action_dim, ACTOR_LEARNING_RATE, device)

    return critic


def get_discrete_dqn(state_dim, action_dim, LEARNING_RATE, device):
    if state_dim is None:
        critic = DQN.visual_obs_dqn(action_dim, LEARNING_RATE, device)
    else:
        critic = DQN.vector_obs_dqn(state_dim, action_dim, LEARNING_RATE, device)

    return critic


def convertToTensorInput(input, input_size, batsize=1):
    input = np.reshape(input, [batsize, input_size])
    return torch.FloatTensor(input)


def get_config(algorithm):
    config_dir = '{0}/{1}'

    with open(config_dir.format('config', "{}.yaml".format(algorithm)), "r") as f:
        try:
            config = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    return SN(**config)


class OU_noise:
    def __init__(self, action_size):
        self.reset()
        self.action_size = action_size
        self.mu = 0.6
        self.theta = 1e-5
        self.sigma = 1e-2

    def reset(self):
        self.X = np.ones(self.action_size) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X) + self.sigma * np.random.randn(len(self.X))
        self.X += dx
        return self.X







