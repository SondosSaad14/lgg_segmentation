from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def build_deeplabv3_resnet50():
    try:
        w = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        m = models.segmentation.deeplabv3_resnet50(weights=w)
    except Exception:
        m = models.segmentation.deeplabv3_resnet50(weights=None)
    in_ch = m.classifier[-1].in_channels if hasattr(m.classifier[-1], "in_channels") else 256
    m.classifier[-1] = nn.Conv2d(in_ch, 1, kernel_size=1)
    if hasattr(m, "aux_classifier") and m.aux_classifier is not None:
        in_ch_aux = m.aux_classifier[-1].in_channels if hasattr(m.aux_classifier[-1], "in_channels") else 256
        m.aux_classifier[-1] = nn.Conv2d(in_ch_aux, 1, kernel_size=1)
    return m


def build_deeplabv3_resnet101():
    try:
        w = models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
        m = models.segmentation.deeplabv3_resnet101(weights=w)
    except Exception:
        m = models.segmentation.deeplabv3_resnet101(weights=None)
    in_ch = m.classifier[-1].in_channels if hasattr(m.classifier[-1], "in_channels") else 256
    m.classifier[-1] = nn.Conv2d(in_ch, 1, kernel_size=1)
    if hasattr(m, "aux_classifier") and m.aux_classifier is not None:
        in_ch_aux = m.aux_classifier[-1].in_channels if hasattr(m.aux_classifier[-1], "in_channels") else 256
        m.aux_classifier[-1] = nn.Conv2d(in_ch_aux, 1, kernel_size=1)
    return m

def build_fcn_resnet50():
    try:
        w = models.segmentation.FCN_ResNet50_Weights.DEFAULT
        m = models.segmentation.fcn_resnet50(weights=w)
    except Exception:
        m = models.segmentation.fcn_resnet50(weights=None)
    in_ch = m.classifier[-1].in_channels if hasattr(m.classifier[-1], "in_channels") else 512
    m.classifier[-1] = nn.Conv2d(in_ch, 1, kernel_size=1)
    if hasattr(m, "aux_classifier") and m.aux_classifier is not None:
        in_ch_aux = m.aux_classifier[-1].in_channels if hasattr(m.aux_classifier[-1], "in_channels") else 256
        m.aux_classifier[-1] = nn.Conv2d(in_ch_aux, 1, kernel_size=1)
    return m

def build_lraspp_mobilenet_v3():
    try:
        w = models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        m = models.segmentation.lraspp_mobilenet_v3_large(weights=w)
    except Exception:
        m = models.segmentation.lraspp_mobilenet_v3_large(weights=None)
    if hasattr(m.classifier, "high_classifier") and hasattr(m.classifier, "low_classifier"):
        try:
            in_high = m.classifier.high_classifier[-1].in_channels
            m.classifier.high_classifier[-1] = nn.Conv2d(in_high, 1, kernel_size=1)
        except Exception:
            pass
        try:
            in_low = m.classifier.low_classifier[-1].in_channels
            m.classifier.low_classifier[-1] = nn.Conv2d(in_low, 1, kernel_size=1)
        except Exception:
            pass
    else:
        # fallback(older versions) - best effort ☺☺ 
        try:
            in_ch = m.classifier[-1].in_channels
            m.classifier[-1] = nn.Conv2d(in_ch, 1, 1)
        except Exception:
            pass
    return m

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.proj = nn.Identity() if in_ch==out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
    def forward(self, x):
        idt = self.proj(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x = F.relu(x + idt, inplace=True)
        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = ResBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        x = self.block(x); return x, self.pool(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.block = ResBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1]!=skip.shape[-1] or x.shape[-2]!=skip.shape[-2]:
            dx = skip.shape[-1]-x.shape[-1]; dy = skip.shape[-2]-x.shape[-2]
            x = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class ResUNet(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.in_conv = ResBlock(3, base)
        self.d1 = Down(base, base*2)
        self.d2 = Down(base*2, base*4)
        self.d3 = Down(base*4, base*8)
        self.bot = ResBlock(base*8, base*16)
        self.u3 = Up(base*16, base*8)
        self.u2 = Up(base*8, base*4)
        self.u1 = Up(base*4, base*2)
        self.u0 = Up(base*2, base)
        self.out = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        x0 = self.in_conv(x)
        x1, p1 = self.d1(x0)
        x2, p2 = self.d2(p1)
        x3, p3 = self.d3(p2)
        xb = self.bot(p3)
        x = self.u3(xb, x3)
        x = self.u2(x, x2)
        x = self.u1(x, x1)
        x = self.u0(x, x0)
        return self.out(x)

def freeze_backbone(model):
    for n,p in model.named_parameters():
        if any(k in n for k in ["classifier", "aux_classifier", "out"]):
            p.requires_grad=True
        else:
            p.requires_grad=False
def unfreeze_all(model):
    for p in model.parameters(): p.requires_grad=True
def forward_logits(model, x):
    out = model(x)
    if isinstance(out, dict) and "out" in out: return out["out"]
    return out
