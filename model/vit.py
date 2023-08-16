# Implements the Segmenter architecture (VIT)

from torch import nn
import segm.model.decoder
import segm.model.segmenter
import segm.model.vit

import torch.nn.functional as F

from model.wrapper import ModelRepresentation

# VIT model class
class VITSegmentModel(nn.Module):
    def __init__(self, num_classes, image_size, dropout=0.1, *args, **kwargs):
        super(VITSegmentModel, self).__init__()

        self.patch_size = 32

        self.image_size = image_size
        self.num_classes = num_classes
        self.dropout = dropout

        self.d_model = 768

        # VisionTransformer as the encoder
        encoder = segm.model.vit.VisionTransformer(
            image_size=(image_size, image_size),
            patch_size=self.patch_size,
            n_layers=12,
            d_model=self.d_model,
            d_ff=4*self.d_model,
            n_heads=self.d_model//64,
            n_cls=num_classes,
            dropout=dropout,
            drop_path_rate=0.0,
            distilled=False,
            channels=3,
        )

        # MaskTransformer as the decoder
        decoder = segm.model.decoder.MaskTransformer(
            n_cls=num_classes,
            patch_size=encoder.patch_size,
            d_encoder=self.d_model,
            n_layers=2,
            n_heads=self.d_model // 64, # 12
            d_model=self.d_model,
            d_ff=4*self.d_model,
            drop_path_rate=0.0,
            dropout=dropout,
        )

        self.model = segm.model.segmenter.Segmenter(encoder, decoder, n_cls=num_classes)

    def forward(self, x):
        return {"pred": self.model(x)}


# Representation layers for the VIT model
class VITRepModel(nn.Module):
    def __init__(self, model, *args, **kwargs):
        super(VITRepModel, self).__init__()

        self.image_size = model.image_size
        self.distilled = model.model.encoder.distilled

        # MaskTransformer as additional representation layers
        self.rep_model = segm.model.decoder.MaskTransformer(
            n_cls=256,
            patch_size=model.patch_size,
            d_encoder=model.d_model,
            n_layers=1,
            n_heads=model.d_model // 64, # 12
            d_model=model.d_model,
            d_ff=4*model.d_model,
            drop_path_rate=0.0,
            dropout=model.dropout,
        )

    def forward(self, x):
        h, w = x.size()[-2:]
        num_extra_tokens = 1 + self.distilled
        x = x[:, num_extra_tokens:]
        x = self.rep_model(x, (self.image_size, self.image_size))
        if (x.shape[2] != h) or (x.shape[3] != w):
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x

# VIT model for the U2PL method with representation outputs
def VITSemi(classes, image_size, semisupervised, *args, **kwargs):
    model = VITSegmentModel(classes, image_size, *args, **kwargs)
    modules_head = [model.model.decoder]
    modules_back = [model.model.encoder]
    
    if semisupervised:
        rep = VITRepModel(model)
        model = ModelRepresentation(model, rep, model.model.encoder)
        modules_head.append(rep)

    return model, modules_head, modules_back

