# Implements a wrapper to compute an representation output for the U2PL method

import torch.nn.functional as F
from torch import nn

# Wrapper for an representation output
class ModelRepresentation(nn.Module):
    def __init__(self, model, rep, rep_forward, x_tmp_transform=None, *args, **kwargs):
        super(ModelRepresentation, self).__init__(*args, **kwargs)

        self.model = model

        if self.training:
            # Representation output
            self.rep = rep
            # Optional additional transform of the output
            self.x_tmp_transform = x_tmp_transform

            # Register hook for a layer of the model
            self.last_activation = {}
            rep_forward.register_forward_hook(self.getActivation('x_tmp'))

    # Returns a hook function that saves an output into an dictionary
    def getActivation(self, name):
        # the hook signature
        def hook(model, input, output):
            self.last_activation[name] = output
        return hook

    def forward(self, x):
        h, w = x.size()[-2:]
        output = self.model(x)

        if self.training:
            # Get last output of an intermediate layer
            x_tmp = self.last_activation["x_tmp"]
            self.last_activation = {}

            # Optional transform of the output
            if self.x_tmp_transform:
                x_tmp = self.x_tmp_transform(x_tmp)

            # Compute representation output
            rep = self.rep(x_tmp)
            if (rep.shape[2] != h) or (rep.shape[3] != w):
               rep = F.interpolate(rep, size=(h, w), mode='bilinear', align_corners=True)

            output["rep"] = rep
            return output
        else:
            return output