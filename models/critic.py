import torch
import torch.nn as nn
from .mobilenet import mobilenet_v2

class Critic:
    def __init__(self, learn_rate, betas, model_tag="mobilenetV2", state_dict_path = None):
        '''
            Create the actor model
        '''
        self.critic_channels = 1 # Outpt is a loss function
        self.latent_var = 64

        self.lr = learn_rate
        self.betas= betas

        self.model_name = model_tag
        self.model_path = state_dict_path

    def get_model(self):
        # TODO: Fix the model initialization 
        model = mobilenet_v2(pretrained=True)

        # 1. pickle the final classifier
        state_dim = model.classifier.in_features
        model.classifier = nn.Sequential(
                nn.Linear(state_dim, self.latent_var),
                nn.Tanh(),
                nn.Linear(self.latent_var, self.latent_var),
                nn.Tanh(),
                nn.Linear(self.latent_var, self.critic_channels)
                )
           
        
        # 2. Set the feature extractor to fixed
        for param in model.features.parameters():
            param.requires_grad = False
        
        # 3. Now set the first conv layer to trainable
        for param in model.features[0][0].parameters():
            param.requires_grad = True

        self.model = model

        return model

    def _init_models(self):
        pass
    
    def get_loss_func(self):
        return torch.nn.MSELoss()
    
    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=self.betas)
