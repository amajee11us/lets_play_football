import os
import torch
import numpy as np
import gfootball.env as football_env

from models.actor import Actor
from models.critic import Critic

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
    actions_onehot = []

    actor = Actor(n_actions, 0.002, (0.9, 0.999))
    m_actor = actor.get_model()

    critic = Critic(0.002, (0.9, 0.999))
    m_critic = critic.get_model()

    # evvauation
    m_actor.to(device)
    m_critic.to(device)
    m_critic.eval()
    m_actor.eval()

    print(m_actor)
    print(m_critic)

    for itr in range(ppo_steps):
        state = torch.from_numpy(state).float().to(device)
        state = torch.unsqueeze(state, 0)
        state = state.permute(0, 3, 1, 2)
        print(state.shape)

        with torch.no_grad():
            action_dist = m_actor(state)
            q_value = m_critic(state)
        
        print(action_dist.shape)

        action_dist = action_dist.cpu().numpy()
        q_value = q_value.cpu().numpy()

        action = np.random.choice(n_actions, p=action_dist[0, :])
        action_onehot = np.zeros(n_actions)
        action_onehot[action] = 1 # set  value of onehot vector

        observation, reward, done, info = env.step(action)
        mask = not done

        states.append(state)
        actions.append(action)
        actions_onehot.append(action_onehot)
        values.append(q_value)
        masks.append(mask)
        rewards.append(reward)
        actions_probs.append(action_dist)


        state = observation

        if done:
            env.reset()

    env.close()
