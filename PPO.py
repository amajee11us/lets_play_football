import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np 

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

        print(masks, q_values, rewards)

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + (self.gamma * q_values[i + 1] * masks[i]) - q_values[i]
            gae = delta + self.gamma * self.lmbda * masks[i] * gae
            returns_per_state.insert(0, gae + q_values[i])
        
        adv = np.array(returns_per_state) - q_values[:-1]
        return returns_per_state, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    
    def _calc_entropy(self, action_dist):
        '''
        Calculate the entropy of the state predictions
        '''
        entropy = action_dist.entropy()
        return entropy

    def calc_loss(self):
        pass
