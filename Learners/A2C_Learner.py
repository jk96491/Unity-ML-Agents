import numpy as np
import Utils
import torch
import torch.nn as nn
import copy


class a2c_agent():
    def __init__(self, env):
        self.GAMMA = 0.95
        self.env = env
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.max_episode_num = 50000
        self.train_mode = True

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.default_brain = self.env.brain_names[0]
        self.env_info = self.env.reset(train_mode=self.train_mode)[self.default_brain]

        self.state_dim = 0
        self.action_dim =self. env_info.action_masks.shape[1]

        self.env = env

        self.actor = Utils.get_discrete_actor(None, self.action_dim, self.ACTOR_LEARNING_RATE, self.device).to(self.device)
        self.critic = Utils.get_discrete_critic(None, self.action_dim, self.ACTOR_LEARNING_RATE, self.device).to(self.device)

        self.save_epi_reward = []

    def train(self):
        actor_loss = 0
        critic_loss = 0
        for ep in range(self.max_episode_num):
            batch_state, batch_action, batch_td_target, batch_advantage = [], [], [], []

            time, episode_reward, done = 0, 0, False

            env_info = self.env.reset(train_mode=self.train_mode)[self.default_brain]
            state = Utils.get_state_by_visual(env_info.visual_observations[0])

            episode_rewards = 0
            done = False

            while not done:
                action = np.asscalar(torch.argmax(self.actor.get_action(state)[0]).detach().cpu().clone().numpy())

                env_info = self.env.step(action)[self.default_brain]
                next_state = Utils.get_state_by_visual(env_info.visual_observations[0])
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                #state = np.reshape(state, [1, self.state_dim])
               #next_state = np.reshape(next_state, [1, self.state_dim])
                action = np.reshape(action, [1, 1])
                reward = np.reshape(reward, [1, 1])

                v_value = self.critic.predict(state)[0]
                next_v_value = self.critic.predict(next_state)[0]

                train_reward = torch.FloatTensor((reward + 8) / 8).to(self.device)
                advantage, y_i = Utils.advantage_td_target(train_reward, v_value, next_v_value, done, self.GAMMA)

                batch_state.append(state)
                batch_action.append(action)
                batch_td_target.append(y_i)
                batch_advantage.append(advantage)

                if len(batch_state) < self.BATCH_SIZE:
                    state = next_state
                    episode_reward += reward[0]
                    time += 1
                    continue

                states = Utils.unpack_batch(batch_state)
                actions = Utils.unpack_batch(batch_action)
                td_targets = torch.stack(batch_td_target, dim=0)
                advantages = torch.stack(batch_advantage, dim=0)
                #td_targets = Utils.unpack_batch(batch_td_target)
                #advantages = Utils.unpack_batch(batch_advantage)

                batch_state, batch_action, batch_td_target, batch_advantage = [], [], [], []

                critic_loss = self.critic.Learn(states, td_targets)
                actor_loss = self.actor.Learn(states, actions, advantages)

                state = next_state
                episode_reward += reward[0]
                time += 1

            print('Episode: ', ep + 1, 'Time: ', time, 'Reward: ', episode_reward, 'actor loss', actor_loss.item(),
                  'critic loss', critic_loss.item())

            self.save_epi_reward.append(episode_reward)

            if ep % 10 == 0:
                self.actor.save_weights('pendulum_actor.th')
                self.critic.save_weights('pendulum_critic.th')