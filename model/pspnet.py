# Sources:
# - https://github.com/hszhao/semseg/blob/master/model/pspnet.py

# Model classes for PSPNet

import torch
import torch.nn.functional as F
from torch import nn

import model.resnet as models
from model.wrapper import ModelRepresentation


# Pyramid Parsing Module
# Source: https://github.com/hszhao/semseg/blob/master/model/pspnet.py
class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

# Implements the PSPNet model
# Source: https://github.com/hszhao/semseg/blob/master/model/pspnet.py
class PSPNet(nn.Module):
    def __init__(self, hparams):
        super(PSPNet, self).__init__()
        self.dropout=0.1
        bins=(1, 2, 3, 6)
        self.zoom_factor = 8

        if hparams.layers == 50:
            resnet = models.resnet50(pretrained=hparams.pretrained)
        elif hparams.layers == 101:
            resnet = models.resnet101(pretrained=hparams.pretrained)
        else:
            resnet = models.resnet152(pretrained=hparams.pretrained)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
        fea_dim *= 2
        
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),
            nn.Conv2d(512, hparams.classes, kernel_size=1)
        )

        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=self.dropout),
                nn.Conv2d(256, hparams.classes, kernel_size=1)
            )

    def forward(self, x):
        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            return {"pred": x, "aux": aux}
        else:
            return {"pred": x}


# PSPNet for frame interpolation
class FlowPSPNet(nn.Module):
    def __init__(self, hparams, *args, **kwargs):
        super(FlowPSPNet, self).__init__()

        model = PSPNet(hparams=hparams, *args, **kwargs)

        # Necessary for Backwards compatibility
        self.layer0 = model.layer0
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.layers = nn.Sequential(
            model.layer0,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )

        self.ppm = model.ppm

        self.encoder = nn.Sequential(
            self.layers,
            model.ppm
        )
        
        self.decoder = model.cls


# PSPNet for U2PL method with representation outputs
def PSPNetSemi(hparams, *args, **kwargs):
    model = PSPNet(hparams=hparams, *args, **kwargs)
    modules_head = [model.ppm, model.cls, model.aux]
    modules_back = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
    
    if hparams.semisupervised:
        rep = nn.Sequential(
            nn.Conv2d(4096, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=model.dropout),
            nn.Conv2d(256, 256, kernel_size=1),
        )
        model = ModelRepresentation(model, rep, model.ppm)
        modules_head.append(rep)

    return model, modules_head, modules_back
