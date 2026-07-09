"""1D U-Net phase picker — compact and quantization-friendly.

Only INT8-friendly ops are used: Conv1d, BatchNorm1d, ReLU, and nearest-neighbor
upsampling (F.interpolate). No LSTM, no attention, no transposed convolution.

Shapes (100 Hz, 60 s window):
    input                     (B, 3, 6000)
    stem  conv k7 s2          (B, 16, 3000)
    enc1  ds-block   s2       (B, 16, 1500)   <- skip s0 is stem out (3000)
    enc2  ds-block   s2       (B, 32, 750)
    enc3  ds-block   s2       (B, 64, 375)
    enc4  ds-block   s2       (B, 96, 188)
    bottleneck x2 (dil 2,4)   (B, 96, 188)
    dec1  up->375  +enc3      (B, 64, 375)
    dec2  up->750  +enc2      (B, 32, 750)
    dec3  up->1500 +enc1      (B, 16, 1500)
    dec4  up->3000 +stem      (B, 16, 3000)
    final up->6000, head 1x1  (B, 3, 6000) -> sigmoid
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv1d(nn.Module):
    """Depthwise-separable 1D conv: depthwise (k, groups=in) -> pointwise (1x1)
    -> BN -> ReLU. Optional stride (downsample) and dilation.

    When ``causal=True`` the depthwise conv uses left-only padding
    (``dilation*(k-1)`` on the left, 0 on the right) instead of symmetric
    padding, so output[t] depends only on input[<=t]. The left pad is chosen so
    the output length is IDENTICAL to the symmetric-padding case (e.g. stride-2
    6000->3000), which keeps every parameter shape unchanged -> an acausal
    checkpoint warm-starts a causal model with strict=True (padding is not a
    learned parameter). Only Conv1d / BatchNorm1d / ReLU / F.pad ops are used,
    all INT8-quantization friendly."""

    def __init__(self, in_ch, out_ch, k=7, stride=1, dilation=1, causal=False):
        super().__init__()
        self.causal = causal
        # Symmetric "same" padding for the acausal path; left-only for causal.
        if causal:
            self.left_pad = dilation * (k - 1)
            pad = 0
        else:
            self.left_pad = 0
            pad = dilation * (k - 1) // 2
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, k, stride=stride, padding=pad,
            dilation=dilation, groups=in_ch, bias=False,
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.left_pad:
            x = F.pad(x, (self.left_pad, 0))
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class SeismicUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        stem_channels=16,
        encoder_channels=(16, 32, 64, 96),
        kernel_size=7,
        bottleneck_dilations=(2, 4),
        out_channels=3,
        causal=False,
        lookahead=0,
    ):
        super().__init__()
        k = kernel_size
        c = list(encoder_channels)
        assert len(c) == 4, "expected 4 encoder channel widths"
        self.causal = causal
        # Fixed streaming delay (samples): output[t] may use input[<=t+lookahead].
        # 0 = strictly causal (the primary configuration). Only meaningful when
        # causal=True. Implemented as a post-hoc left-shift of the causal output.
        self.lookahead = int(lookahead) if causal else 0

        # Stem: conv k7 s2, 6000 -> 3000. Causal uses left-only pad (applied in
        # forward) so the Conv1d weight shape is unchanged vs the acausal model.
        stem_pad = 0 if causal else (k - 1) // 2
        self.stem_left_pad = (k - 1) if causal else 0
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, k, stride=2,
                      padding=stem_pad, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
        )

        # Encoder (each downsamples x2 via stride-2 depthwise)
        self.enc1 = DSConv1d(stem_channels, c[0], k, stride=2, causal=causal)  # ->1500
        self.enc2 = DSConv1d(c[0], c[1], k, stride=2, causal=causal)           # ->750
        self.enc3 = DSConv1d(c[1], c[2], k, stride=2, causal=causal)           # ->375
        self.enc4 = DSConv1d(c[2], c[3], k, stride=2, causal=causal)           # ->188

        # Bottleneck: 2 depthwise-separable blocks, dilations 2 and 4
        self.bottleneck = nn.Sequential(
            DSConv1d(c[3], c[3], k, dilation=bottleneck_dilations[0], causal=causal),
            DSConv1d(c[3], c[3], k, dilation=bottleneck_dilations[1], causal=causal),
        )

        # Decoder: nearest upsample + DS-conv, with encoder skips concatenated.
        self.dec1 = DSConv1d(c[3] + c[2], c[2], k, causal=causal)  # +enc3
        self.dec2 = DSConv1d(c[2] + c[1], c[1], k, causal=causal)  # +enc2
        self.dec3 = DSConv1d(c[1] + c[0], c[0], k, causal=causal)  # +enc1
        self.dec4 = DSConv1d(c[0] + stem_channels, stem_channels, k, causal=causal)  # +stem

        # Head: 1x1 conv to out streams + sigmoid (applied in forward)
        self.head = nn.Conv1d(stem_channels, out_channels, 1)

    def _up_cat(self, x, skip):
        """Upsample x to skip's length and concat along channels.

        Acausal: nearest (F.interpolate). Causal: floor-aligned upsample where
        fine index j samples coarse index floor(j*Ls/Lf) <= its own nominal
        time, so no future coarse sample is ever read (verified by the impulse
        unit test). Both are INT8-friendly gather/resize ops."""
        x = self._upsample(x, skip.shape[-1])
        return torch.cat([x, skip], dim=1)

    def _upsample(self, x, size):
        if not self.causal:
            return F.interpolate(x, size=size, mode="nearest")
        src = x.shape[-1]
        # floor(j * src / size) for each fine index j -> past-only gather.
        idx = (torch.arange(size, device=x.device) * src) // size
        idx = idx.clamp_max(src - 1)
        return x.index_select(-1, idx)

    def forward(self, x):
        input_len = x.shape[-1]
        if self.stem_left_pad:
            x = F.pad(x, (self.stem_left_pad, 0))
        s0 = self.stem(x)          # 3000
        e1 = self.enc1(s0)         # 1500
        e2 = self.enc2(e1)         # 750
        e3 = self.enc3(e2)         # 375
        e4 = self.enc4(e3)         # 188
        b = self.bottleneck(e4)    # 188

        d = self.dec1(self._up_cat(b, e3))   # 375
        d = self.dec2(self._up_cat(d, e2))   # 750
        d = self.dec3(self._up_cat(d, e1))   # 1500
        d = self.dec4(self._up_cat(d, s0))   # 3000

        d = self._upsample(d, input_len)     # 6000
        out = self.head(d)
        if self.lookahead > 0:
            # Fixed streaming delay: output[t] := causal_out[t+L], tail replicated.
            L = self.lookahead
            out = F.pad(out[..., L:], (0, L), mode="replicate")
        return torch.sigmoid(out)


def build_model(cfg) -> SeismicUNet:
    m = cfg.model
    return SeismicUNet(
        in_channels=cfg.data.n_channels,
        stem_channels=m.stem_channels,
        encoder_channels=tuple(m.encoder_channels),
        kernel_size=m.kernel_size,
        bottleneck_dilations=tuple(m.bottleneck_dilations),
        out_channels=m.out_channels,
        # New (optional) knobs; default false/0 so existing configs are unchanged.
        causal=bool(getattr(m, "causal", False)),
        lookahead=int(getattr(m, "lookahead", 0)),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(model: nn.Module, input_shape=(1, 3, 6000)) -> int:
    """Count multiply-accumulates for all Conv1d layers via forward hooks.

    FLOPs ~= 2 * MACs. Reported as MFLOPs by callers.
    """
    macs = {"n": 0}
    hooks = []

    def hook(module, inp, out):
        out_len = out.shape[-1]
        macs["n"] += (
            module.out_channels
            * out_len
            * (module.in_channels // module.groups)
            * module.kernel_size[0]
        )

    for mod in model.modules():
        if isinstance(mod, nn.Conv1d):
            hooks.append(mod.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(*input_shape))
    if was_training:
        model.train()
    for h in hooks:
        h.remove()
    return macs["n"]


def model_summary(model: nn.Module, input_shape=(1, 3, 6000)) -> dict:
    params = count_parameters(model)
    macs = count_macs(model, input_shape)
    return {
        "parameters": params,
        "macs": macs,
        "mflops": 2 * macs / 1e6,
    }
