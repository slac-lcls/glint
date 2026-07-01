"""Train the 3D-CNN peakfinder (PeakNet-style) on Perlmutter (torch + A100).

Generates (volume, heatmap) pairs on the fly from the simulator, trains a 3D U-Net
to predict the peakness heatmap, and saves a checkpoint. Designed for a GPU; runs on
CPU only for the tiny --smoke validation.

Usage:
    python train_cnn.py --smoke                 # tiny CPU end-to-end check
    srun ... python train_cnn.py --epochs 40    # real run on a Perlmutter GPU node
See docs/perlmutter_cnn.md for the NERSC recipe.
"""

import argparse
import sys
import time

sys.path.insert(0, "..")

import torch
from torch.utils.data import DataLoader

from glint.cnn import UNet3D, VolumeHeatmapDataset, dice_bce_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny CPU sanity run")
    ap.add_argument("--n", type=int, default=96, help="grid size")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--train", type=int, default=2000, help="samples per epoch")
    ap.add_argument("--val", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="cnn_peakfinder.pt")
    a = ap.parse_args()

    if a.smoke:                                      # shrink everything for a CPU check
        a.n, a.epochs, a.train, a.val, a.batch, a.workers = 32, 1, 16, 8, 2, 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  n={a.n}  epochs={a.epochs}  train/epoch={a.train}")

    tr = DataLoader(VolumeHeatmapDataset(a.train, n=a.n, base_seed=0),
                    batch_size=a.batch, num_workers=a.workers, shuffle=True, drop_last=True)
    va = DataLoader(VolumeHeatmapDataset(a.val, n=a.n, base_seed=10_000_000),
                    batch_size=a.batch, num_workers=a.workers)

    model = UNet3D().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    for ep in range(a.epochs):
        t0 = time.time()
        model.train()
        tl = 0.0
        for v, y in tr:
            v, y = v.to(device), y.to(device)
            opt.zero_grad()
            loss = dice_bce_loss(model(v), y)
            loss.backward()
            opt.step()
            tl += loss.item() * len(v)
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for v, y in va:
                v, y = v.to(device), y.to(device)
                vl += dice_bce_loss(model(v), y).item() * len(v)
        print(f"epoch {ep:3d}  train {tl/a.train:.4f}  val {vl/a.val:.4f}  "
              f"[{time.time()-t0:.0f}s]")
        torch.save({"model": model.state_dict(), "n": a.n}, a.out)

    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
