# Sources:
# - https://github.com/sud0301/semisup-semseg/blob/master/train_s4GAN.py

import numpy as np
import torch
from torch import nn

# The discriminator model for the s4GAN approach
# Source: https://github.com/sud0301/semisup-semseg/blob/master/train_s4GAN.py
class s4GAN_discriminator(nn.Module):
    def __init__(self, num_classes, ndf = 64):
        super(s4GAN_discriminator, self).__init__()
        leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        drop_out = nn.Dropout2d(0.5)

        self.layers = nn.Sequential(
            nn.Conv2d(num_classes+3, ndf, kernel_size=4, stride=2, padding=1),
            leaky_relu,
            drop_out,
            nn.Conv2d(  ndf, ndf*2, kernel_size=4, stride=2, padding=1),
            leaky_relu,
            drop_out,
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1),
            leaky_relu,
            drop_out,
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1),
            leaky_relu,
            nn.AdaptiveAvgPool2d(1)
        )

        self.final = nn.Sequential(
            nn.Linear(ndf*8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):        
        maps = self.layers(x)
        conv4_maps = maps
        out = maps.view(maps.size(0), -1)
        out = self.final(out)
        
        return out, conv4_maps

# Applies argmax to generate a segmentation mask
# Source: https://github.com/sud0301/semisup-semseg/blob/master/train_s4GAN.py
def compute_argmax_map(output):
    output = output.detach().cpu().numpy()
    output = output.transpose((1,2,0))
    output = np.asarray(np.argmax(output, axis=2), dtype=np.int32)
    output = torch.from_numpy(output).to(dtype=torch.long)
    return output

# Identify the segmentation masks above the threshold to use for self-training
# Source: https://github.com/sud0301/semisup-semseg/blob/master/train_s4GAN.py
def find_good_maps(D_outs, pred_all, threshold_st):
    count = 0
    for i in range(D_outs.size(0)):
        if D_outs[i] > threshold_st:
            count +=1

    # If reliable predictions exist
    if count > 0:

        # Setup pseudo-labels
        pred_sel = torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3)).to(dtype=pred_all.dtype)
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3)).to(dtype=torch.long)
        num_sel = 0 
        for j in range(D_outs.size(0)):
            if D_outs[j] > threshold_st:
                pred_sel[num_sel] = pred_all[j]
                label_sel[num_sel] = compute_argmax_map(pred_all[j])
                num_sel +=1
        return  pred_sel.to(device=pred_all.device), label_sel.to(device=pred_all.device), count  
    else:
        return 0, 0, count 

# Returns a one hot representation for a segmentation mask
# Source: https://github.com/sud0301/semisup-semseg/blob/master/train_s4GAN.py
def one_hot(l, classes):
    label = l.cpu().numpy()
    one_hot = np.zeros((label.shape[0], classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(classes):
        one_hot[:,i,...] = (label==i)
    
    return torch.Tensor(one_hot).to(dtype=l.dtype, device=l.device)
