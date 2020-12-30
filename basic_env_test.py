import os
os.environ['LANG']='en_US'

import gfootball.env as football_env
'''
Environments:
1. academy_empty_goal - 2 players with a empty goal. Player (agent) is spawned at the centerline.
2. 11_vs_11_stochastic - 11 players a team doing stochastic movements
'''
env = football_env.create_environment(env_name= '11_vs_11_stochastic', representation='pixels', render=True)
state = env.reset()

while True:
    '''
    This is a basic code that generates random states for each player agent.
    1. State is defined using vectors that determine the direction of motion along with a set of actions
    2. Each action is associated with an observation, reward and completion indicator.
    We use this as training parameters in a reward function for any RL problem.
    '''
    action = env.action_space.sample()
    obs, rewards,done, info = env.step(action)

    if done:
        env.reset()