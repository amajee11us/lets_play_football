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

        # Perform an action
        observation, reward, done, info = env.step(action)
        mask = torch.FloatTensor(1 - done).unsqueeze(1).to(device)

        states.append(state)
        actions.append(action)
        actions_onehot.append(action_onehot)
        values.append(q_value)
        masks.append(mask)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        actions_probs.append(action_dist)

        state = observation # update to next state

        if done:
            env.reset()

    env.close()

def calculate_advantage(masks, q_values, rewards):
    gae = 0
    returns_per_state = [] # this is tracked per step to achieve a goal

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + (gamma * q_values[i + 1] * masks[i]) - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns_per_state.insert(0, gae + values[i])
    
    adv = np.array(returns_per_state) - q_values[:-1]
    return returns_per_state, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

class PPO:

    def __init__(self, learn_rate= 0.01, betas= (0.9, 0.999), eps_clip=0.2, gamma=0.99, lmbda=0.95):
        # Advantage calculation
        self.gamma = gamma # discount factor
        self.lmbda = lmbda # smoothing factor

        #PPO loss constants
        self.lr = learn_rate
        self.betas = betas
        self.eps_clip = eps_clip

        self.MSEloss = torch.nn.MSELoss()

    def calc_advantage(self, masks, q_values, rewards):
        '''
        Genearlized Advantage Estimation Algorithm
        '''
        gae = 0
        returns_per_state = [] # this is tracked per step to achieve a goal

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + (self.gamma * q_values[i + 1] * masks[i]) - values[i]
            gae = delta + self.gamma * self.lmbda * masks[i] * gae
            returns_per_state.insert(0, gae + values[i])
        
        adv = np.array(returns_per_state) - q_values[:-1]
        return returns_per_state, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    
    def _calc_entropy(self, state_values):
        '''
        Calculate the entropy of the state predictions
        '''
        entropy = torch.mean(state * torch.log(state + 1e-10))
        return entropy

    def calc_loss(self):

