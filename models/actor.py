import torch
import torch.nn as nn
from .mobilenet import mobilenet_v2

class Actor:
    def __init__(self, out_dim, learn_rate, betas, model_tag="mobilenetV2", state_dict_path = None):
        '''
            Create the actor model
        '''
        self.action_dim = out_dim[-1] # HWC format
        self.latent_var = 64;

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
                nn.Linear(self.latent_var, self.action_dim),
                nn.Softmax(dim=-1)
                )
        
        # 2. Set the feature extractor to fixed
        for param in model.features.parameters():
            param.requires_grad = False
        
        # 3. Now set the first conv layer to trainable
        model.features[0][0].parameters().requires_grad = True

        self.model = model

        return model

    def _init_models(self):
        pass
    
    def get_loss_func(self):
        return torch.nn.MSELoss()
    
    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=self.betas)
