import torch
import Utils
import numpy as np
import copy
from Memory.Replay_Memory import Replay_buffer


class dqn_agents:
    def __init__(self, env, args):
        self.args = args
        self.env = env
        self.GAMMA = args.GAMMA

        self.LEARNING_RATE = args.LEARNING_RATE
        self.max_episode_num = args.max_episode
        self.train_mode = args.train_mode
        self.buffer_size = args.buffer_size
        self.start_train_episode = args.start_train_episode
        self.target_update_interval = args.target_update_interval

        self.device1 = args.device1 if torch.cuda.is_available() else 'cpu'

        self.default_brain = self.env.brain_names[0]
        self.env_info = self.env.reset(train_mode=self.train_mode)[self.default_brain]

        self.state_dim = 0
        self.action_dim = self.env_info.action_masks.shape[1]

        self.env = env
        self.model = Utils.get_discrete_dqn(None, self.action_dim, self.LEARNING_RATE, self.device1).to(self.device1)
        self.target_model = copy.deepcopy(self.model)

        self.save_epi_reward = []

    def train(self):
        loss = 0
        replay_buffer = Replay_buffer(self.buffer_size)
        total_step = 0
        for ep in range(self.max_episode_num):
            epsilon = 1 / ((ep / 10) + 1)

            env_info = self.env.reset(train_mode=self.train_mode)[self.default_brain]
            state = Utils.get_state_by_visual(env_info.visual_observations[0])

            episode_rewards = 0
            done = False
            step_count = 0

            while not done:
                if np.random.rand(1) < epsilon:
                    action = np.random.randint(0, self.action_dim)
                else:
                    action = self.model.get_action(state)

                env_info = self.env.step(action)[self.default_brain]
                next_state = Utils.get_state_by_visual(env_info.visual_observations[0])
                reward = env_info.rewards[0]
                episode_rewards += reward
                done = env_info.local_done[0]

                action = np.reshape(action, [1, 1])
                reward = np.reshape(reward, [1, 1])

                replay_buffer.append((state, action, reward, next_state, done))

                state = next_state
                step_count += 1
                total_step += 1

                if total_step > self.start_train_episode:
                    minibatch = replay_buffer.random_sample(self.start_train_episode)
                    loss = self.model.Learn(self.target_model, minibatch, self.GAMMA)

                    if (total_step % self.target_update_interval) == 0:
                        self.update_target(self.model, self.target_model)
                        print("target updated!!!")

            print("episode: {} / step: {} / reward: {:.2f} / loss: {:.4f} / epsilon: {:.3f}".format
                  (ep, step_count, episode_rewards, loss, epsilon))

    def update_target(self, mainDQN, targetDQN):
        targetDQN.load_state_dict(mainDQN.state_dict())
