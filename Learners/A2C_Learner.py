import Utils
import numpy as np


class a2c_agent():
    def __init__(self, env, args):
        self.GAMMA = args.GAMMA
        self.args = args
        self.env = env
        self.BATCH_SIZE = args.BATCH_SIZE
        self.ACTOR_LEARNING_RATE = args.ACTOR_LEARNING_RATE
        self.CRITIC_LEARNING_RATE = args.CRITIC_LEARNING_RATE
        self.max_episode_num = args.max_episode
        self.train_mode = args.train_mode

        if self.args.framework == 'torch':
            self.device1 = Utils.get_device(args.device1)
            self.device2 = Utils.get_device(args.device2)
        else:
            self.device1 = Utils.get_device(args.device1)
            self.device2 = Utils.get_device(args.device2)

        self.default_brain = self.env.brain_names[0]
        self.env_info = self.env.reset(train_mode=self.train_mode)[self.default_brain]

        self.state_dim = 0
        self.action_dim = self.env_info.action_masks.shape[1]

        self.env = env

        self.actor = Utils.get_discrete_actor(None, self.action_dim, self.ACTOR_LEARNING_RATE, self.device1)
        self.critic = Utils.get_discrete_critic(None, self.action_dim, self.CRITIC_LEARNING_RATE, self.device2)

        self.save_epi_reward = []

    def train(self):
        actor_loss = 0
        critic_loss = 0
        for ep in range(self.max_episode_num):
            batch_state, batch_action, batch_td_target, batch_advantage = [], [], [], []

            time, episode_reward, done = 0, 0, False

            env_info = self.env.reset(train_mode=self.train_mode)[self.default_brain]
            state = Utils.get_state_by_visual(env_info.visual_observations[0], self.args.framework)

            episode_rewards = 0
            done = False

            while not done:
                action = self.actor.get_action(state)

                env_info = self.env.step(action)[self.default_brain]
                next_state = Utils.get_state_by_visual(env_info.visual_observations[0], self.args.framework)
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                action = np.reshape(action, [1, 1])
                reward = np.reshape(reward, [1, 1])

                v_value = self.critic.predict(state)[0]
                next_v_value = self.critic.predict(next_state)[0]

                advantage, y_i = Utils.advantage_td_target(reward, v_value, next_v_value, done, self.GAMMA, self.device2)

                batch_state.append(state)
                batch_action.append(action)
                batch_td_target.append(y_i)
                batch_advantage.append(advantage)

                if len(batch_state) < self.BATCH_SIZE:
                    state = next_state
                    episode_reward += reward[0]
                    time += 1
                    continue

                states = batch_state
                actions = batch_action
                td_targets = batch_td_target
                advantages = batch_advantage

                batch_state, batch_action, batch_td_target, batch_advantage = [], [], [], []

                critic_loss = self.critic.Learn(states, td_targets)
                actor_loss = self.actor.Learn(states, actions, advantages)

                state = next_state
                episode_rewards += reward[0]
                time += 1

            print('Episode: ', ep + 1, 'Time: ', time, 'Reward: ', episode_rewards, 'actor loss', actor_loss.item(),
                  'critic loss', critic_loss.item())

            self.save_epi_reward.append(episode_reward)

            if ep % 10 == 0:
                self.actor.save_weights('aircombat_actor.th')
                self.critic.save_weights('aircombat_critic.th')