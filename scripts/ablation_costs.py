#!/usr/bin/env python
"""Architecture cost ablations: params / MFLOPs / receptive field per variant.

Instantiates SeismicUNet variants (block type, encoder depth, channel widths,
kernel size, bottleneck dilation) and measures each with the same counters as
scripts/inspect_model.py. Runnable with no data and no checkpoint — these are
properties of the architecture, not of trained weights. This script backs the
"Design rationale" tables in the README.

    python scripts/ablation_costs.py
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seismic_edge_picker.model import (  # noqa: E402
    DSConv1d,
    SeismicUNet,
    count_macs,
    count_parameters,
)

WINDOW = 6000  # 60 s @ 100 Hz


class RegularConvBlock(nn.Module):
    """Standard Conv1d + BN + ReLU with the same interface as DSConv1d."""

    def __init__(self, in_ch, out_ch, k=7, stride=1, dilation=1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=pad,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class FlexUNet(nn.Module):
    """SeismicUNet generalized to a variable number of encoder levels and a
    pluggable conv block, so cost variants can be instantiated and measured.
    With block=DSConv1d and 4 levels it is weight-for-weight identical to
    SeismicUNet (asserted in main())."""

    def __init__(self, block, in_channels=3, stem_channels=16,
                 encoder_channels=(16, 32, 64, 96), kernel_size=7,
                 bottleneck_dilations=(2, 4), out_channels=3):
        super().__init__()
        k = kernel_size
        c = list(encoder_channels)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, k, stride=2,
                      padding=(k - 1) // 2, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
        )
        encs, prev = [], stem_channels
        for ch in c:
            encs.append(block(prev, ch, k, stride=2))
            prev = ch
        self.encs = nn.ModuleList(encs)
        self.bottleneck = nn.Sequential(
            *[block(c[-1], c[-1], k, dilation=d) for d in bottleneck_dilations]
        )
        skips = [stem_channels] + c[:-1]  # skip widths at each decoder stage
        decs = []
        for i in range(len(c) - 1, -1, -1):
            out_ch = skips[i] if i > 0 else stem_channels
            decs.append(block(c[i] + skips[i], out_ch, k))
        self.decs = nn.ModuleList(decs)
        self.head = nn.Conv1d(stem_channels, out_channels, 1)

    @staticmethod
    def _up_cat(x, skip):
        x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        input_len = x.shape[-1]
        s0 = self.stem(x)
        feats, h = [s0], s0
        for enc in self.encs:
            h = enc(h)
            feats.append(h)
        h = self.bottleneck(h)
        for i, dec in enumerate(self.decs):
            h = dec(self._up_cat(h, feats[len(self.encs) - 1 - i]))
        h = F.interpolate(h, size=input_len, mode="nearest")
        return torch.sigmoid(self.head(h))


def bottleneck_receptive_field(kernel=7, n_enc=4, dilations=(2, 4)):
    """Receptive field (input samples) of one bottleneck-output sample,
    tracked along the encoder path (stem + strided depthwise + dilated
    bottleneck; pointwise 1x1 convs add nothing)."""
    rf, jump = 1, 1
    rf += (kernel - 1) * jump  # stem k, s2
    jump *= 2
    for _ in range(n_enc):     # each encoder level: depthwise k, s2
        rf += (kernel - 1) * jump
        jump *= 2
    for d in dilations:        # bottleneck: depthwise k, dilation d, s1
        rf += d * (kernel - 1) * jump
    return rf


def bottleneck_len(n_enc, length=WINDOW):
    n = (length + 1) // 2      # stem
    for _ in range(n_enc):
        n = (n + 1) // 2
    return n


def measure(model, n_enc=4, kernel=7, dilations=(2, 4)):
    params = count_parameters(model)
    mflops = 2 * count_macs(model, input_shape=(1, 3, WINDOW)) / 1e6
    rf = bottleneck_receptive_field(kernel, n_enc, dilations)
    stride = 2 ** (n_enc + 1)  # stem s2 + n_enc stride-2 encoder levels
    return params, mflops, rf, stride, bottleneck_len(n_enc)


def print_table(title, rows):
    print(f"\n### {title}\n")
    print("| variant | params | MFLOPs / window | bottleneck RF | bottleneck (stride, len) |")
    print("|---|---:|---:|---:|---|")
    for name, (params, mflops, rf, stride, blen) in rows:
        print(f"| {name} | {params:,} | {mflops:.1f} | "
              f"{rf} samp ({rf / 100:.1f} s) | x{stride}, {blen} |")


def main():
    # Sanity: the 4-level DS FlexUNet must be weight-identical to the repo model.
    ref, flex = SeismicUNet(), FlexUNet(DSConv1d)
    assert count_parameters(ref) == count_parameters(flex)
    assert count_macs(ref) == count_macs(flex)
    print(f"sanity: FlexUNet(DS, 4 levels) == SeismicUNet "
          f"({count_parameters(ref):,} params) PASS")

    print_table("Block type (4 levels, channels 16/32/64/96, k=7)", [
        ("depthwise-separable (shipped)", measure(FlexUNet(DSConv1d))),
        ("regular Conv1d, same widths", measure(FlexUNet(RegularConvBlock))),
    ])

    print_table("Encoder depth (DS blocks, k=7, dilations 2/4)", [
        ("3 levels (16,32,64)",
         measure(FlexUNet(DSConv1d, encoder_channels=(16, 32, 64)), n_enc=3)),
        ("4 levels (16,32,64,96) (shipped)", measure(FlexUNet(DSConv1d))),
        ("5 levels (16,32,64,96,128)",
         measure(FlexUNet(DSConv1d, encoder_channels=(16, 32, 64, 96, 128)),
                 n_enc=5)),
    ])

    print_table("Bottleneck dilation (DS blocks, 4 levels, k=7)", [
        ("no dilation (1,1)",
         measure(FlexUNet(DSConv1d, bottleneck_dilations=(1, 1)),
                 dilations=(1, 1))),
        ("dilations (2,4) (shipped)", measure(FlexUNet(DSConv1d))),
    ])

    print_table("Channel widths (DS blocks, 4 levels, k=7)", [
        ("half (8,16,32,48), stem 8",
         measure(FlexUNet(DSConv1d, stem_channels=8,
                          encoder_channels=(8, 16, 32, 48)))),
        ("shipped (16,32,64,96), stem 16", measure(FlexUNet(DSConv1d))),
        ("uniform (64,64,64,64), stem 64",
         measure(FlexUNet(DSConv1d, stem_channels=64,
                          encoder_channels=(64, 64, 64, 64)))),
        ("double (32,64,128,192), stem 32",
         measure(FlexUNet(DSConv1d, stem_channels=32,
                          encoder_channels=(32, 64, 128, 192)))),
    ])

    print_table("Kernel size (DS blocks, shipped widths)", [
        (f"k={k}" + (" (shipped)" if k == 7 else ""),
         measure(FlexUNet(DSConv1d, kernel_size=k), kernel=k))
        for k in (3, 5, 7, 11)
    ])


if __name__ == "__main__":
    main()
