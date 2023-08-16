# Model classes for DeepLabv3

import torch
import torch.nn.functional as F
from torch import nn
import torchvision

from model.wrapper import ModelRepresentation

# Implements the DeepLabv3 model
class DeepLabv3(nn.Module):
    def __init__(self, hparams, *args, **kwargs):
        super(DeepLabv3, self).__init__()

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet' + str(hparams.layers), pretrained=hparams.pretrained)

        # Replaces classifier and auxiliary classifier with different output dimensions
        self.model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, hparams.classes)
        self.model.aux_classifier = torchvision.models.segmentation.fcn.FCNHead(1024, hparams.classes)

    def forward(self, x):
        h, w = x.size()[-2:]
        output = self.model(x)
        pred = output["out"]
        aux = output["aux"]

        if self.training:
            if (aux.shape[2] != h) or (aux.shape[3] != w):
               aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)

            return {"pred": pred, "aux": aux}
        else:
            return {"pred": pred}

# Wrapper for the DeepLabv3 backbone that outputs the prediction of the model
class OutTransformModule(nn.Module):
    def __init__(self, model):
        super(OutTransformModule, self).__init__()

        self.model = model

    def forward(self, x):
        return self.model(x)["out"]
    

# DeepLabv3 for frame interpolation
class FlowDeepLabv3(nn.Module):
    def __init__(self, hparams, *args, **kwargs):
        super(FlowDeepLabv3, self).__init__()

        model = DeepLabv3(hparams=hparams, *args, **kwargs)
        
        self.encoder = OutTransformModule(model.model.backbone)
        self.decoder = model.model.classifier

# DeepLabv3 for U2PL method with representation outputs
def DeepLabv3Semi(hparams, *args, **kwargs):
    model = DeepLabv3(hparams, *args, **kwargs)
    modules_back = [model.model.backbone]
    modules_head = [model.model.classifier, model.model.aux_classifier]
    
    if hparams.semisupervised:
        rep = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, 256, kernel_size=1),
        )
        model = ModelRepresentation(model, rep, model.model.backbone, lambda x_tmp: x_tmp["out"])
        modules_head.append(rep)

    return model, modules_head, modules_back