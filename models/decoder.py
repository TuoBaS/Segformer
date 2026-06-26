import torch
import torch.nn as nn
from torch.nn import functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0,
                 groups=1, dilation=1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class SemanticContextGate(nn.Module):
    """Use the deepest semantic feature to recalibrate all decoder scales."""

    def __init__(self, decoder_dim, num_scales=4, reduction=4):
        super().__init__()
        hidden_dim = max(decoder_dim // reduction, 32)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(decoder_dim, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, decoder_dim, kernel_size=1),
                nn.Sigmoid(),
            )
            for _ in range(num_scales)
        ])

    def forward(self, features, semantic_feature):
        gated = []
        for feature, gate in zip(features, self.gates):
            gated.append(feature * (1.0 + gate(semantic_feature)))
        return gated


class DepthwiseContextRefine(nn.Module):
    """Lightweight multi-dilation depthwise refinement at 1/4 resolution."""

    def __init__(self, channels, dilations=(1, 3, 5)):
        super().__init__()
        self.branches = nn.ModuleList([
            ConvBNAct(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                groups=channels,
                dilation=dilation,
            )
            for dilation in dilations
        ])
        self.project = ConvBNAct(channels * (len(dilations) + 1), channels, kernel_size=1)

    def forward(self, x):
        context = [x]
        context.extend(branch(x) for branch in self.branches)
        return x + self.project(torch.cat(context, dim=1))


class AllMLPDecoder(nn.Module):
    def __init__(
        self,
        encoder_dims=(32, 64, 160, 256),
        decoder_dim=256,
        num_classes=19,
        dropout_ratio=0.1,
        context_gate=False,
        context_reduction=4,
        refine=False,
        refine_dilations=(1, 3, 5),
        edge_head=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.edge_head_enabled = edge_head

        self.conv1 = nn.Conv2d(encoder_dims[0], decoder_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(encoder_dims[1], decoder_dim, kernel_size=1)
        self.conv3 = nn.Conv2d(encoder_dims[2], decoder_dim, kernel_size=1)
        self.conv4 = nn.Conv2d(encoder_dims[3], decoder_dim, kernel_size=1)

        self.context_gate = (
            SemanticContextGate(decoder_dim, num_scales=4, reduction=context_reduction)
            if context_gate else None
        )

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
        )

        self.refine = (
            DepthwiseContextRefine(decoder_dim, dilations=tuple(refine_dilations))
            if refine else None
        )

        self.linear_pred = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

        if edge_head:
            self.edge_head = nn.Sequential(
                ConvBNAct(decoder_dim, decoder_dim, kernel_size=3, padding=1, groups=decoder_dim),
                nn.Conv2d(decoder_dim, 1, kernel_size=1),
            )
        else:
            self.edge_head = None

    def forward(self, encoder_outs, return_aux=False):
        c1, c2, c3, c4 = encoder_outs
        _, _, target_h, target_w = c1.shape

        c1 = self.conv1(c1)
        c2 = F.interpolate(self.conv2(c2), size=(target_h, target_w),
                           mode="bilinear", align_corners=False)
        c3 = F.interpolate(self.conv3(c3), size=(target_h, target_w),
                           mode="bilinear", align_corners=False)
        c4 = F.interpolate(self.conv4(c4), size=(target_h, target_w),
                           mode="bilinear", align_corners=False)

        features = [c1, c2, c3, c4]
        if self.context_gate is not None:
            features = self.context_gate(features, c4)

        fused = self.linear_fuse(torch.cat(features, dim=1))
        if self.refine is not None:
            fused = self.refine(fused)

        logits = self.linear_pred(self.dropout(fused))
        if return_aux and self.edge_head is not None:
            return {"seg": logits, "edge": self.edge_head(fused)}
        return logits
