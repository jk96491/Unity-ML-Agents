import numpy as np
import torch
import yaml
from types import SimpleNamespace as SN
from Modules.Pytoch.Discrete import Actor as torchActor, Critic as torchCritic, DQN as torchDQN
from Modules.Tensorflow.discrete import DQN as tensorflowDQN, Actor as tensorflowActor, Critic as tensorflowCritic
import copy


def get_state_by_visual(data, framework):

    if framework == 'torch':
        data = np.uint8(255 * np.array(data))

        state = []
        for i in range(data.shape[3]):
            state.append(data[:, :, :, i])

        return np.array(state).reshape(data.shape[0], data.shape[3], data.shape[1], data.shape[2])
    else:
        return data


def advantage_td_target(reward, v_value, next_v_value, done, GAMMA, framework, device):
    reward = (reward + 8) / 8

    if framework == 'torch':
        reward = torch.FloatTensor(reward).to(device)

    if done:
        y_k = v_value
        advantage = y_k - v_value
    else:
        y_k = reward + GAMMA * next_v_value
        advantage = y_k - v_value

    return advantage, y_k


def get_discrete_actor(state_dim, action_dim, ACTOR_LEARNING_RATE, device, framework, env_info, hidden):
    if state_dim is None:
        if framework == 'torch':
            actor = torchActor.visual_obs_actor(action_dim, ACTOR_LEARNING_RATE, device, env_info, hidden)
        else:
            actor = tensorflowActor.visual_obs_actor(action_dim, ACTOR_LEARNING_RATE, device, env_info, hidden)
    else:
        actor = torchActor.vector_obs_actor(state_dim, action_dim, ACTOR_LEARNING_RATE, device, hidden)

    return actor


def get_discrete_critic(state_dim, action_dim, ACTOR_LEARNING_RATE, device, framework, env_info, hidden):
    if state_dim is None:
        if framework == 'torch':
            critic = torchCritic.visual_obs_critic(action_dim, ACTOR_LEARNING_RATE, device, env_info, hidden)
        else:
            critic = tensorflowCritic.visual_obs_critic(action_dim, ACTOR_LEARNING_RATE, device, env_info, hidden)
    else:
        critic = torchCritic.vector_obs_critic(state_dim, action_dim, ACTOR_LEARNING_RATE, device, hidden)

    return critic


def get_discrete_dqn(state_dim, action_dim, LEARNING_RATE, device, framework, env_info, hidden):
    if state_dim is None:
        if framework == 'torch':
            dqn = torchDQN.visual_obs_dqn(action_dim, LEARNING_RATE, device, env_info, hidden)
        else:
            dqn = tensorflowDQN.visual_obs_dqn(action_dim, LEARNING_RATE, device, env_info, hidden)
    else:
        dqn = torchDQN.vector_obs_dqn(state_dim, action_dim, LEARNING_RATE, device, hidden)

    return dqn


def init_target_network(framework, model, env_info, hidden):
    if framework == 'torch':
        target_model = copy.deepcopy(model)
    else:
        target_model = get_discrete_dqn(None, model.action_space, model.learning_rate, model.device, framework, env_info)
        target_model.set_weights(model.get_weights())

    return target_model


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


def get_device(device_name):
    device = device_name if torch.cuda.is_available() else 'cpu'
    return device


def update_target(mainDQN, targetDQN, framework):
    if framework == 'torch':
        targetDQN.load_state_dict(mainDQN.state_dict())
    else:
        targetDQN.set_weights(mainDQN.get_weights())


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









