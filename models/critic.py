import torch
import torch.nn as NN
from .mobilenet import mobilenet_v2

class Actor:
    def __init__(self, model_tag="mobilenetV2", state_dict_path = None):
        '''
            Create the actor model
        '''
        self.out_channels = 1 # Outpt is a loss function
        self.model_name = model_tag
        self.model_path = state_dict_path

    def get_model(self):
        # TODO: Fix the model initialization 
        model = mobilenet_v2(pretrained=True)

        # 1. pickle the final classifier
        final_layer_channels = model.classifier.in_features
        model.classifier = NN.Linear(in_features=final_layer_channels,
                                           out_features=self.out_channels)   
        
        # 2. Set the feature extractor to fixed
        for param in model.features.parameters():
            param.requires_grad = False
        
        # 3. Now set the first conv layer to trainable
        model.features[0][0].parameters().requires_grad = True

        return model

    def _init_models(self):
        pass
