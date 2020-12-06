from mlagents.envs.environment import UnityEnvironment
import Utils
from agents import get_agents

game = "AirCombat/Aircombat"
#game = "AirHockey/AirHockey"
env_name = 'Games/{}'.format(game)

if __name__ == '__main__':
    args = Utils.get_config('DQN')
    env = UnityEnvironment(file_name=env_name)

    default_brain = env.brain_names[0]
    agent = get_agents(env, args)

    if agent is not None:
        agent.train()

    env.close()