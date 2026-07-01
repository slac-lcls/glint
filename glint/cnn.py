"""PeakNet-style 3D U-Net peakfinder on the FFT volume (Perlmutter / torch + A100).

Guarded import: the rest of fftindex works without torch (this module just exposes
nothing torch-y until torch is present). The model maps the (1,n,n,n) normalized
|F(x)| volume to a (1,n,n,n) peakness heatmap; CNNPeakFinder extracts heatmap maxima
as candidate real-space lattice vectors and matches the index_shot peakfinder
interface, so it drops in via index_shot(..., peakfinder=CNNPeakFinder(model)).

CNNPeakFinder resamples the input volume to the model's training size and maps peak
voxels back through the input xcoords range, so it tolerates index_shot's auto/escalated
grid sizes. Peak extraction reuses peakfind.find_peaks_classical on the heatmap.
"""

from __future__ import annotations

import numpy as np

from .peakfind import find_peaks_classical

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:                                  # repo stays importable without torch
    HAS_TORCH = False


if HAS_TORCH:

    from .cnn_dataset import make_cnn_sample

    class _ConvBlock(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(cin, cout, 3, padding=1), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
                nn.Conv3d(cout, cout, 3, padding=1), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class UNet3D(nn.Module):
        """Small 3-level 3D U-Net, 1-channel sigmoid heatmap output."""

        def __init__(self, base=16):
            super().__init__()
            self.e1 = _ConvBlock(1, base)
            self.e2 = _ConvBlock(base, base * 2)
            self.e3 = _ConvBlock(base * 2, base * 4)
            self.pool = nn.MaxPool3d(2)
            self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
            self.d2 = _ConvBlock(base * 4, base * 2)
            self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
            self.d1 = _ConvBlock(base * 2, base)
            self.out = nn.Conv3d(base, 1, 1)

        def forward(self, x):
            c1 = self.e1(x)
            c2 = self.e2(self.pool(c1))
            c3 = self.e3(self.pool(c2))
            u2 = self.d2(torch.cat([self.up2(c3), c2], 1))
            u1 = self.d1(torch.cat([self.up1(u2), c1], 1))
            return torch.sigmoid(self.out(u1))

    def dice_bce_loss(pred, target, eps=1.0):
        """BCE + soft Dice -- robust to the heavy peak/background imbalance."""
        bce = F.binary_cross_entropy(pred, target)
        num = 2 * (pred * target).sum() + eps
        den = pred.sum() + target.sum() + eps
        return bce + (1 - num / den)

    class VolumeHeatmapDataset(torch.utils.data.Dataset):
        """On-the-fly (volume, heatmap) pairs -- a full materialized set is ~GBs.

        Deterministic per index (seeded by base_seed+idx) so epochs are reproducible.
        """

        def __init__(self, length, n=96, base_seed=0, **sample_kw):
            self.length, self.n, self.base_seed, self.kw = length, n, base_seed, sample_kw

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            rng = np.random.default_rng(self.base_seed + idx)
            v, y = make_cnn_sample(rng, n=self.n, **self.kw)
            return torch.from_numpy(v)[None], torch.from_numpy(y)[None]


class CNNPeakFinder:
    """Wrap a trained UNet3D as an index_shot peakfinder.

    __call__(vol, xcoords, g, qmax, min_len) -> (vecs, amps), matching the
    find_peaks interface. The input volume is resampled to the model's training
    size; peak voxels map back through the physical extent [xcoords[0], xcoords[-1]],
    so any index_shot grid size works.
    """

    def __init__(self, model, n_model=96, device="cpu", thresh=0.3):
        if not HAS_TORCH:
            raise RuntimeError("CNNPeakFinder requires torch")
        self.model = model.to(device).eval()
        self.n_model, self.device, self.thresh = n_model, device, thresh

    def __call__(self, vol, xcoords, g=None, qmax=None, min_len=3.0):
        with torch.no_grad():
            vol = np.asarray(vol, np.float32)
            vol = vol / max(vol.max(), 1e-9)          # match training normalization
            v = torch.as_tensor(vol, dtype=torch.float32,
                                device=self.device)[None, None]
            if v.shape[-1] != self.n_model:
                v = F.interpolate(v, size=(self.n_model,) * 3, mode="trilinear",
                                  align_corners=False)
            heat = self.model(v)[0, 0].cpu().numpy()
        # physical coordinates of the model-grid voxels (preserves input extent)
        xc = np.linspace(xcoords[0], xcoords[-1], self.n_model)
        return find_peaks_classical(heat, xc, min_len=min_len, rel_thresh=self.thresh)
