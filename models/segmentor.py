import os
import sys

import torch.nn as nn
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import MixVisionTransformer
from models.decoder import AllMLPDecoder


class SegFormer(nn.Module):
    def __init__(self, img_size=224, num_classes=150, encoder_pretrained=True,
                 encoder_config=None, decoder_config=None):
        super().__init__()

        assert encoder_config is not None, "encoder_config is required"
        assert decoder_config is not None, "decoder_config is required"

        self.encoder = MixVisionTransformer(
            img_size=img_size,
            embed_dims=encoder_config["embed_dims"],
            num_heads=encoder_config["num_heads"],
            depths=encoder_config["depths"],
            mlp_ratios=encoder_config["mlp_ratios"],
            sr_ratios=encoder_config.get("sr_ratios", [8, 4, 2, 1]),
            drop_path=encoder_config.get("drop_path_rate", 0.1),
            encoder_pretrained=encoder_pretrained,
        )

        self.decoder = AllMLPDecoder(
            encoder_dims=encoder_config["embed_dims"],
            decoder_dim=decoder_config["decoder_dim"],
            num_classes=num_classes,
            dropout_ratio=decoder_config.get("dropout_ratio", 0.1),
            context_gate=decoder_config.get("context_gate", False),
            context_reduction=decoder_config.get("context_reduction", 4),
            refine=decoder_config.get("refine", False),
            refine_dilations=decoder_config.get("refine_dilations", [1, 3, 5]),
            edge_head=decoder_config.get("edge_head", False),
        )

    def forward(self, x, return_aux=False):
        feats = self.encoder(x)
        decoded = self.decoder(feats, return_aux=return_aux)

        if isinstance(decoded, dict):
            decoded["seg"] = F.interpolate(
                decoded["seg"], size=(x.shape[2], x.shape[3]),
                mode="bilinear", align_corners=False,
            )
            if "edge" in decoded:
                decoded["edge"] = F.interpolate(
                    decoded["edge"], size=(x.shape[2], x.shape[3]),
                    mode="bilinear", align_corners=False,
                )
            return decoded

        return F.interpolate(
            decoded, size=(x.shape[2], x.shape[3]),
            mode="bilinear", align_corners=False,
        )
