"""Train the powder autoencoder + cell/system head. Denoising AE (noisy input -> clean target) plus a
supervised head. Loss = MSE(recon,clean) + lam_cell*MSE(cell) + lam_sys*CE(system).

  python train.py                 # default: ~12k train / 2k val, 60 epochs
  python train.py 2000 30 1       # n_train=2000, epochs=30, overfit-check flag
Saves ae.pt (+ prints per-epoch val recon-RMS, system-acc, cell-MAE-Angstrom).
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_dataset import make_dataset, norm_cell, denorm_cell, CELL_LO, CELL_HI  # noqa: E402
from model import PowderAE  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
LAM_CELL, LAM_SYS, NOISE = 5.0, 0.5, 0.04
# supervised-contrastive term on the normalised latent (pull same-system latents together). Off by default
# (LAM_CON=0 reproduces the original AE); env LAM_CON>0 turns it on. No new params -> ae.pt still loads.
LAM_CON = float(os.environ.get("LAM_CON", "0"))
CON_TEMP = 0.1
AE_PT = os.environ.get("AE_PT", "ae.pt")


def supcon(z, labels, temp=CON_TEMP):
    """Supervised contrastive loss (Khosla 2020) on L2-normalised latents; batch must have >1 per class."""
    z = torch.nn.functional.normalize(z, dim=1)
    N = z.size(0)
    logits = z @ z.T / temp - torch.eye(N, device=z.device) * 1e9      # mask self-similarity
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    lab = labels.view(-1, 1)
    pos = (lab == lab.T).float() - torch.eye(N, device=z.device)       # same-class, excluding self
    denom = pos.sum(1).clamp(min=1)
    return -(pos * logp).sum(1).div(denom).mean()


def tensors(n, seed):
    X, cells, sysid, _ = make_dataset(n, seed=seed)
    Xn = np.stack([norm_cell(c) for c in cells])
    return (torch.tensor(X).unsqueeze(1), torch.tensor(Xn), torch.tensor(sysid))


def evaluate(net, Xv, Cv, Sv):
    net.eval()
    with torch.no_grad():
        xr, cp, sl, _ = net(Xv.to(DEV))
        rms = float(((xr.cpu() - Xv) ** 2).mean().sqrt())
        acc = float((sl.argmax(1).cpu() == Sv).float().mean())
        # cell MAE in Angstrom on the 3 edges (denormalised)
        span = torch.tensor(CELL_HI - CELL_LO)
        mae = float((torch.abs(cp.cpu() - Cv) * span)[:, :3].mean())
    return rms, acc, mae


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    overfit = len(sys.argv) > 3
    print("device:", DEV, "n_train:", n_train, "epochs:", epochs, "overfit:", overfit)
    Xt, Ct, St = tensors(n_train, seed=1)
    Xv, Cv, Sv = tensors(2000, seed=999)
    net = PowderAE().to(DEV)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()
    bs = 128 if not overfit else n_train
    idx = np.arange(n_train)
    t0 = time.time()
    for ep in range(epochs):
        net.train(); np.random.shuffle(idx)
        for b in range(0, n_train, bs):
            j = idx[b:b + bs]
            x = Xt[j].to(DEV); c = Ct[j].to(DEV); s = St[j].to(DEV)
            xin = x + NOISE * torch.randn_like(x) if not overfit else x       # denoising
            xr, cp, sl, z = net(xin)
            loss = mse(xr, x) + LAM_CELL * mse(cp, c) + LAM_SYS * ce(sl, s)
            if LAM_CON > 0:
                loss = loss + LAM_CON * supcon(z, s)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            rms, acc, mae = evaluate(net, Xv, Cv, Sv)
            print("ep %3d  loss=%.4f  val: recon_rms=%.4f  sys_acc=%.3f  edge_MAE=%.2f A  [%.0fs]"
                  % (ep, float(loss), rms, acc, mae, time.time() - t0))
    torch.save({"state": net.state_dict()}, os.path.join(os.path.dirname(__file__), AE_PT))
    print("saved", AE_PT, "(LAM_CON=%.2f)" % LAM_CON)


if __name__ == "__main__":
    main()
