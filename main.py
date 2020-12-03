from mlagents.envs.environment import UnityEnvironment
from Learners.A2C_Learner import a2c_agent

game = "AirCombat/Aircombat"
env_name = 'Games/' + game

run_episode = 500000
test_episode = 100000

start_train_episode = 10

print_interval = 5
save_interval = 100

if __name__ == '__main__':
    env = UnityEnvironment(file_name=env_name)
    default_brain = env.brain_names[0]
    agent = a2c_agent(env)

    agent.train()

    env.close()