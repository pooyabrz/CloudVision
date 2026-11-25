"""Four independently trained architectures. SegFormer uses native PyTorch attention."""

import torch
from torch import nn
from torch.nn import functional as F


class CloudCNN(nn.Module):
    def __init__(self, width=16, dropout=0.12):
        super().__init__()
        layers = []
        incoming = 2
        for outgoing in (width, width * 2, width * 3):
            layers.extend(
                [
                    nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                    nn.BatchNorm2d(outgoing),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            incoming = outgoing
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(incoming, 11)
        )

    def forward(self, x):
        return self.head(self.features(x))


class Block(nn.Module):
    def __init__(self, left, right, residual):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(left, right, 3, padding=1, bias=False),
            nn.BatchNorm2d(right),
            nn.ReLU(),
            nn.Conv2d(right, right, 3, padding=1, bias=False),
            nn.BatchNorm2d(right),
        )
        self.shortcut = nn.Conv2d(left, right, 1, bias=False) if residual else None

    def forward(self, x):
        out = self.conv(x)
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return F.relu(out)


class UNet(nn.Module):
    def __init__(self, width=12, dropout=0.1, residual=False):
        super().__init__()
        self.enc1 = Block(2, width, residual)
        self.enc2 = Block(width, 2 * width, residual)
        self.bridge = nn.Sequential(Block(2 * width, 4 * width, residual), nn.Dropout2d(dropout))
        self.dec2 = Block(6 * width, 2 * width, residual)
        self.dec1 = Block(3 * width, width, residual)
        self.out = nn.Conv2d(width, 14, 1)

    def merge(self, x, skip):
        return torch.cat(
            (F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip),
            dim=1,
        )

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        z = self.bridge(F.max_pool2d(b, 2))
        return self.out(self.dec1(self.merge(self.dec2(self.merge(z, b)), a)))


class MixBlock(nn.Module):
    def __init__(self, width, heads, reduction, dropout):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.reduce = nn.Conv2d(width, width, reduction, reduction) if reduction > 1 else None
        self.sr_norm = nn.LayerNorm(width) if reduction > 1 else nn.Identity()
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, width * 2)
        self.attention_out = nn.Linear(width, width)
        self.expand = nn.Conv2d(width, width * 3, 1)
        self.depthwise = nn.Conv2d(width * 3, width * 3, 3, padding=1, groups=width * 3)
        self.contract = nn.Conv2d(width * 3, width, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        norm = self.norm1(t)
        reduced = norm
        if self.reduce is not None:
            reduced = self.sr_norm(
                self.reduce(norm.transpose(1, 2).reshape(b, c, h, w)).flatten(2).transpose(1, 2)
            )

        def heads(t):
            return t.reshape(b, -1, self.heads, c // self.heads).transpose(1, 2)

        k, v = self.kv(reduced).chunk(2, -1)
        att = F.scaled_dot_product_attention(heads(self.q(norm)), heads(k), heads(v))
        t = t + self.drop(self.attention_out(att.transpose(1, 2).reshape(b, h * w, c)))
        z = self.norm2(t).transpose(1, 2).reshape(b, c, h, w)
        return t.transpose(1, 2).reshape(b, c, h, w) + self.drop(
            self.contract(self.drop(F.gelu(self.depthwise(self.expand(z)))))
        )


class SegFormer(nn.Module):
    def __init__(self, widths, heads, depths, reductions, decoder=32, dropout=0.1):
        super().__init__()
        self.patches, self.stages, self.norms, self.proj = (nn.ModuleList() for _ in range(4))
        inc = 2
        for i, width in enumerate(widths):
            kernel, stride = (7, 4) if i == 0 else (3, 2)
            self.patches.append(nn.Conv2d(inc, width, kernel, stride, padding=kernel // 2))
            self.stages.append(
                nn.Sequential(
                    *(MixBlock(width, heads[i], reductions[i], dropout) for _ in range(depths[i]))
                )
            )
            self.norms.append(nn.LayerNorm(width))
            self.proj.append(nn.Conv2d(width, decoder, 1))
            inc = width
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder * len(widths), decoder, 1, bias=False),
            nn.BatchNorm2d(decoder),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder, 14, 1),
        )

    def forward(self, x):
        size = x.shape[-2:]
        fs = []
        for patch, stage, norm, proj in zip(self.patches, self.stages, self.norms, self.proj):
            x = patch(x)
            x = stage(norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())
            fs.append(proj(x))
        features = torch.cat(
            [
                F.interpolate(v, size=fs[0].shape[-2:], mode="bilinear", align_corners=False)
                for v in fs
            ],
            1,
        )
        return F.interpolate(self.fuse(features), size=size, mode="bilinear", align_corners=False)


def build(name, settings):
    if name == "cnn":
        return CloudCNN(**settings)
    if name in ("unet", "unet_plus"):
        return UNet(**settings, residual=name == "unet_plus")
    if name == "segformer":
        return SegFormer(**settings)
    raise ValueError("Unknown architecture.")
