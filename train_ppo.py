import os
import torch
from torch.distributions import Normal
import numpy as np
import gfootball.env as football_env
import gym

from models.actor import Actor
from models.critic import Critic
from models.ActorCritic import ActorCritic

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # This is the main function
    env = football_env.create_environment(env_name='academy_empty_goal', representation='pixels', render=True)
    state = env.reset()

    state_dims = env.observation_space.shape
    print(state_dims)

    n_actions = env.action_space.n 
    print(n_actions)

    ppo_steps = 128

    states = []
    actions = []
    values = []
    masks = []
    rewards = []
    actions_probs = []
    entropy = 0

    model = ActorCritic(n_actions, 0.002, (0.9, 0.999))
    model.to(device)
    model.eval()

    optim = torch.optim.Adam(model.parameters(), lr=0.002)

    print(model)    

    for itr in range(ppo_steps):
        state = torch.from_numpy(state).float().to(device)
        state = torch.unsqueeze(state, 0)
        state = state.permute(0, 3, 1, 2)
        print(state.shape)
        
        # jointly forward pass on actor and critic models
        action_dist, q_value = model(state)
                     
        # Perform an action
        action = action_dist.sample()
        print(action)
        next_state, reward, done, info = env.step(action.item())

        log_probs = action_dist.log_prob(action)
        entropy += action_dist.entropy().mean()
        print(entropy)
        
        print(" Iter : {} State shape : {} Reward: {} Mask : {} ".format(itr, state.shape, reward, done))

        '''
        # NOTE : only if multiple envs are present
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        mask = torch.FloatTensor(1 - done).unsqueeze(1).to(device)
        '''
        
        reward = torch.FloatTensor([reward])
        mask = torch.FloatTensor([1 - done])

        states.append(state)
        actions.append(action)
        actions_probs.append(log_probs)
        values.append(q_value)
        masks.append(mask)
        rewards.append(reward)
        actions_probs.append(action_dist)

        state = next_state # update to next state

        if done:
            env.reset()

    env.close()
