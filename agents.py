from Learners.A2C_Learner import a2c_agent
from Learners.DQN_Learner import dqn_agents


def get_agents(env, args):
    agent = None
    if args.agent == 'A2C':
        agent = a2c_agent(env, args)
    elif args.agent == 'DQN':
        agent = dqn_agents(env, args)

    return agent
