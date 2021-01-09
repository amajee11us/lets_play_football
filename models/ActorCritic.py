import torch
import torch.nn as nn 
from torch.distributions import Normal
from .actor import Actor
from .critic import Critic

class ActorCritic(nn.Module):
    def __init__(self, out_dim_actor, learn_rate, betas, std=0.0):
        '''
        Model : The Actor critic model
        '''
        self.actor_model = Actor(out_dim_actor, learn_rate, betas)
        self.critic_model = Critic(learn_rate, betas)

        # logarithmic standard dev
        self.log_std = nn.Parameter(torch.ones(1, out_dim_actor) * std)

    def forward(self, x):
        '''
        Here x is the state
        '''
        value = self.critic_model(x)
        mu = self.actor_model(x)

        # now create a action distribution
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std) # following normal distribution
        return dist, value