"""PyTorch re-implementation of DR(eye)VE (Palazzi et al., TPAMI 2018).

Reference implementation (Keras 1 / Theano):
    https://github.com/ndrplz/dreyeve/blob/master/experiments/train/models.py

Architecture of one saliency branch (``SaliencyBranch`` in the reference code)::

    clip  [B, C, T, H, W]
      |-- last frame (full res) ---------------------------------------+
      |-- resize to (H/4, W/4) --> coarse 3D-CNN encoder               |
      |       --> temporal collapse --> bilinear up x8 --> conv3x3(1)  |
      |       --> ReLU --> bilinear up x4 --> [B, 1, H, W] ----- concat
      |                                                                |
      |                         refinement: conv32-conv16-conv8-conv1 (LeakyReLU 0.001)
      |                                         --> ReLU --> fine map [B, 1, H, W]
      |
      +-- (training only) random crop clip [B, C, T, H/4, W/4]
              --> *shared* coarse encoder --> up x8 --> conv3x3(1) --> ReLU
              --> crop map [B, 1, H/4, W/4]

The coarse encoder is C3D up to ``conv4b`` (as in the paper) or, optionally, a
torchvision video ResNet (``r3d_18`` / ``mc3_18`` / ``r2plus1d_18``).

Differences from the reference implementation (intentional):
    * Upsampling to the target size is done with ``F.interpolate(size=...)`` instead of fixed
      x8/x4 factors so that backbones with a different output stride (/16 for the ResNets)
      and non-square inputs work transparently. For C3D with H, W divisible by 32 this is
      the same x8/x4 bilinear upsampling as the original.
    * The temporal dimension is collapsed with a max over time. For C3D with T=16 this is
      exactly the reference ``pool4`` (pool_size=(4,1,1)) followed by the reshape, but it
      also accepts other clip lengths.
    * Sports-1M C3D weights (Keras ``.h5``) are not converted; C3D starts from scratch.
      Kinetics-400 weights are available for the torchvision backbones.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

TORCHVISION_BACKBONES = ("r3d_18", "mc3_18", "r2plus1d_18")
BACKBONES = ("c3d",) + TORCHVISION_BACKBONES


# --------------------------------------------------------------------------------------
# Coarse encoders
# --------------------------------------------------------------------------------------
class C3DEncoder(nn.Module):
    """C3D up to ``conv4b`` as used by ``CoarseSaliencyModel`` in the reference code.

    Spatial stride 8, output 512 channels; the temporal axis is pooled by 4 (T=16 -> 4).
    """

    out_channels = 512

    def __init__(self, in_channels: int = 3):
        super().__init__()

        def conv(cin, cout):
            return nn.Sequential(nn.Conv3d(cin, cout, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        self.features = nn.Sequential(
            conv(in_channels, 64),                              # conv1
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # pool1
            conv(64, 128),                                      # conv2
            nn.MaxPool3d(kernel_size=2, stride=2),              # pool2
            conv(128, 256),                                     # conv3a
            conv(256, 256),                                     # conv3b
            nn.MaxPool3d(kernel_size=2, stride=2),              # pool3
            conv(256, 512),                                     # conv4a
            conv(512, 512),                                     # conv4b
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class TorchvisionVideoEncoder(nn.Module):
    """torchvision video ResNet-18 family (stem .. ``out_layer``) used as coarse encoder."""

    def __init__(self, name: str, in_channels: int = 3, pretrained: bool = False,
                 out_layer: str = "layer4"):
        super().__init__()
        from torchvision.models import video as tv_video

        if name not in TORCHVISION_BACKBONES:
            raise ValueError(f"Unknown torchvision backbone: {name}")
        layers = ("layer1", "layer2", "layer3", "layer4")
        if out_layer not in layers:
            raise ValueError(f"out_layer must be one of {layers}, got {out_layer}")

        weights = "KINETICS400_V1" if pretrained else None
        net = getattr(tv_video, name)(weights=weights)

        if in_channels != 3:
            # Re-create the first conv for non-RGB inputs (e.g. optical flow / semseg).
            first = net.stem[0]
            new = nn.Conv3d(in_channels, first.out_channels, kernel_size=first.kernel_size,
                            stride=first.stride, padding=first.padding, bias=first.bias is not None)
            if pretrained:
                with torch.no_grad():
                    mean_w = first.weight.mean(dim=1, keepdim=True)
                    new.weight.copy_(mean_w.repeat(1, in_channels, 1, 1, 1))
            net.stem[0] = new

        n_layers = layers.index(out_layer) + 1
        self.stem = net.stem
        self.layers = nn.Sequential(*[getattr(net, l) for l in layers[:n_layers]])
        self.out_channels = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}[out_layer]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(self.stem(x))


def build_encoder(backbone: str, in_channels: int, pretrained: bool, out_layer: str) -> nn.Module:
    if backbone == "c3d":
        if pretrained:
            raise ValueError("Pretrained C3D (Sports-1M) weights are not provided in this port; "
                             "set pretrained: false or use a torchvision backbone.")
        return C3DEncoder(in_channels)
    return TorchvisionVideoEncoder(backbone, in_channels, pretrained, out_layer)


# --------------------------------------------------------------------------------------
# Saliency branch
# --------------------------------------------------------------------------------------
class CoarseSaliencyModel(nn.Module):
    """3D encoder + temporal collapse + bilinear upsampling to the input clip resolution."""

    def __init__(self, backbone: str = "c3d", in_channels: int = 3, pretrained: bool = False,
                 out_layer: str = "layer4"):
        super().__init__()
        self.encoder = build_encoder(backbone, in_channels, pretrained, out_layer)
        self.out_channels = self.encoder.out_channels

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        h, w = clip.shape[-2:]
        feat = self.encoder(clip)            # [B, F, T', h', w']
        feat = feat.amax(dim=2)              # squeeze out time (== C3D pool4 for T=16)
        return F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)


class SaliencyBranch(nn.Module):
    """Coarse-to-fine saliency branch of DR(eye)VE (one input modality)."""

    def __init__(self, in_channels: int = 3, backbone: str = "c3d", pretrained: bool = False,
                 out_layer: str = "layer4", coarse_scale: float = 0.25):
        super().__init__()
        self.coarse_scale = coarse_scale
        self.coarse = CoarseSaliencyModel(backbone, in_channels, pretrained, out_layer)
        f = self.coarse.out_channels

        # coarse head used for the full-frame path
        self.coarse_conv = nn.Conv2d(f, 1, kernel_size=3, padding=1)
        # head used for the crop path (``{branch}_crop_final_conv``)
        self.crop_conv = nn.Conv2d(f, 1, kernel_size=3, padding=1)

        self.refine = nn.Sequential(
            nn.Conv2d(1 + in_channels, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.001),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.LeakyReLU(0.001),
            nn.Conv2d(16, 8, kernel_size=3, padding=1), nn.LeakyReLU(0.001),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )
        self._init_weights()

    def _init_weights(self):
        # Reference: he_normal for refine_conv1..3, glorot_uniform for refine_conv4 / crop conv.
        for m in list(self.refine)[:-1]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu", a=0.001)
                nn.init.zeros_(m.bias)
        for m in (self.refine[-1], self.crop_conv, self.coarse_conv):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def coarse_size(self, h: int, w: int) -> Tuple[int, int]:
        return max(1, round(h * self.coarse_scale)), max(1, round(w * self.coarse_scale))

    def forward(self, clip: torch.Tensor, crop_clip: Optional[torch.Tensor] = None):
        """
        :param clip: full-resolution clip [B, C, T, H, W]. The last frame is the one whose
            saliency is predicted (as in the reference code).
        :param crop_clip: optional crop clip [B, C, T, h, w] (training-time auxiliary task).
        :return: fine map [B, 1, H, W] (non-negative), plus crop map [B, 1, h, w] if
            ``crop_clip`` is given.
        """
        if clip.dim() != 5:
            raise ValueError(f"Expected clip of shape [B, C, T, H, W], got {tuple(clip.shape)}")
        b, c, t, h, w = clip.shape
        hs, ws = self.coarse_size(h, w)

        last_frame = clip[:, :, -1]                                            # [B, C, H, W]
        small = F.interpolate(clip, size=(t, hs, ws), mode="trilinear", align_corners=False)

        coarse = F.relu(self.coarse_conv(self.coarse(small)))                  # [B, 1, hs, ws]
        coarse = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)
        fine = F.relu(self.refine(torch.cat([coarse, last_frame], dim=1)))     # [B, 1, H, W]

        if crop_clip is None:
            return fine
        crop = F.relu(self.crop_conv(self.coarse(crop_clip)))
        return fine, crop


# --------------------------------------------------------------------------------------
# Full models
# --------------------------------------------------------------------------------------
class DrEyeVEBaseline(SaliencyBranch):
    """Single-branch (RGB) DR(eye)VE: input [B, C, T, H, W] -> saliency map [B, 1, H, W]."""


class DreyeveNet(nn.Module):
    """Multi-branch DR(eye)VE (e.g. image + optical flow + semantic segmentation).

    Branch predictions are summed and passed through a ReLU, as in the reference
    ``DreyeveNet``. Inputs are given as a dict ``{branch_name: clip}``.
    """

    def __init__(self, branches: Dict[str, int], backbone: str = "c3d", pretrained: bool = False,
                 out_layer: str = "layer4", coarse_scale: float = 0.25):
        super().__init__()
        self.branches = nn.ModuleDict({
            name: SaliencyBranch(in_ch, backbone, pretrained, out_layer, coarse_scale)
            for name, in_ch in branches.items()
        })

    def forward(self, clips: Dict[str, torch.Tensor],
                crop_clips: Optional[Dict[str, torch.Tensor]] = None):
        fines, crops = [], []
        for name, branch in self.branches.items():
            if crop_clips is None:
                fines.append(branch(clips[name]))
            else:
                f, c = branch(clips[name], crop_clips[name])
                fines.append(f)
                crops.append(c)
        fine = F.relu(torch.stack(fines).sum(0))
        if crop_clips is None:
            return fine
        return fine, F.relu(torch.stack(crops).sum(0))


def build_model(cfg: dict) -> nn.Module:
    """Build a model from the ``model`` section of a config dict."""
    kwargs = dict(
        backbone=cfg.get("backbone", "c3d"),
        pretrained=cfg.get("pretrained", False),
        out_layer=cfg.get("out_layer", "layer4"),
        coarse_scale=cfg.get("coarse_scale", 0.25),
    )
    if kwargs["backbone"] not in BACKBONES:
        raise ValueError(f"backbone must be one of {BACKBONES}, got {kwargs['backbone']}")
    branches: Union[None, Dict[str, int]] = cfg.get("branches")
    if branches:
        return DreyeveNet(branches=dict(branches), **kwargs)
    return DrEyeVEBaseline(in_channels=cfg.get("in_channels", 3), **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = DrEyeVEBaseline(backbone="c3d")
    x = torch.randn(1, 3, 16, 448, 448)
    with torch.no_grad():
        y = net(x)
    print(f"{tuple(x.shape)} -> {tuple(y.shape)}  params={count_parameters(net) / 1e6:.2f}M")
