import torch
import torch.nn as nn 
from torch.distributions import Categorical

from .actor import Actor
from .critic import Critic

class ActorCritic(nn.Module):
    def __init__(self, out_dim_actor, learn_rate, betas, std=0.0):
        '''
        Model : The Actor critic model
        '''
        super(ActorCritic, self).__init__()

        # initialize the actor and critic classes
        actor = Actor(out_dim_actor, learn_rate, betas)
        critic = Critic(learn_rate, betas)

        self.actor_model = actor.get_model()
        self.critic_model = critic.get_model()

        self.optimizer = actor.get_optimizer()

        # logarithmic standard dev
        #self.log_std = nn.Parameter(torch.ones(1, out_dim_actor) * std)

        self.apply(init_weights)

    def forward(self, x):
        '''
        Here x is the state
        '''
        value = self.critic_model(x)
        mu = self.actor_model(x)

        # now create a action distribution
        dist  = Categorical(mu) # following normal distribution
        return dist, value
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)