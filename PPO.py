import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np 

class PPO:

    def __init__(self, epochs=4, batch_size=5, learn_rate= 0.01, betas= (0.9, 0.999), eps_clip=0.2, gamma=0.99, lmbda=0.95):
        # Advantage calculation
        self.gamma = gamma # discount factor
        self.lmbda = lmbda # smoothing factor

        #PPO loss constants
        self.lr = learn_rate
        self.betas = betas
        self.eps_clip = eps_clip
        self.ppo_epochs = epochs
        self.batch_size = batch_size

        self.MSEloss = torch.nn.MSELoss()

    def calc_returns(self, masks, q_values, rewards, next_value):
        '''
        Genearlized Advantage Estimation Algorithm
        '''
        q_values = q_values + [next_value] # decouple it so that the final value is not affected
        gae = 0
        returns_per_state = [] # this is tracked per step to achieve a goal

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + (self.gamma * q_values[i + 1] * masks[i]) - q_values[i]
            gae = delta + self.gamma * self.lmbda * masks[i] * gae
            returns_per_state.insert(0, gae + q_values[i])
                
        return returns_per_state
    
    def _calc_entropy(self, action_dist):
        '''
        Calculate the entropy of the state predictions
        '''
        entropy = action_dist.entropy().mean()
        return entropy

    def __do_iter(self, batch, states, actions, log_probs, returns, advantages):
        batch_size = states.size(0)
        for _ in range(batch_size // batch):
            rand_ids = np.random.randint(0, batch_size, batch)
            # return states for each batch
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantages[rand_ids, :]

    def update_backward(self, model, optimizer, states, actions, action_probs, returns, advantages):
        for _ in range(self.ppo_epochs):
            for state, action, prev_log_prob, return_, advantage in self.__do_iter(self.batch_size, states, actions, action_probs, returns, advantages):
                pdist, q_value = model(state)
                entropy = self._calc_entropy(pdist)
                next_log_prob = pdist.log_prob(action)

                ratio = (next_log_prob - prev_log_prob).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantage

                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - q_value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
