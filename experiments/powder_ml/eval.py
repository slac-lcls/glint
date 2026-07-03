"""Evaluate the trained powder autoencoder + head:
  (1) denoising RMS vs input noise
  (2) latent clustering by crystal system (silhouette)
  (3) system accuracy + cell edge-MAE on held-out
  (4) HEAD-TO-HEAD vs the classical index_powder (same profile in): accuracy + wall-time, clean vs noisy.
Run after train.py (needs ae.pt).
"""
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_dataset import make_dataset, norm_cell, QGRID, SYSTEMS, CELL_LO, CELL_HI  # noqa: E402
from model import PowderAE  # noqa: E402
from powder_index import index_powder, peaks_from_profile  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HIGHSYM = {"cubic", "tetragonal", "hexagonal", "orthorhombic"}   # classical fast path for the timed head-to-head


AE_PT = os.environ.get("AE_PT", "ae.pt")   # which weights to evaluate (ae.pt baseline vs ae_con.pt contrastive)


def load():
    net = PowderAE().to(DEV)
    net.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), AE_PT), map_location=DEV)["state"])
    net.eval(); return net


def main():
    net = load()
    X, cells, sysid, cens = make_dataset(2000, seed=12345)
    Xt = torch.tensor(X).unsqueeze(1)

    print("=== (1) denoising RMS vs input noise (recon vs CLEAN) ===")
    for nz in [0.0, 0.02, 0.05, 0.1]:
        with torch.no_grad():
            xin = Xt + nz * torch.randn_like(Xt)
            xr, _, _, _ = net(xin.to(DEV))
            print("  noise=%.2f: recon_rms=%.4f  (input_rms=%.4f)" %
                  (nz, float(((xr.cpu() - Xt) ** 2).mean().sqrt()), float((nz * torch.randn_like(Xt)).std())))

    print("=== (2) latent clustering by system (silhouette; higher=better separated) ===")
    with torch.no_grad():
        z = net.encode(Xt.to(DEV)).cpu().numpy()
    try:
        from sklearn.metrics import silhouette_score
        print("  silhouette(z, system) = %.3f  (chance ~0)" % silhouette_score(z[:1500], sysid[:1500]))
    except Exception as e:
        print("  sklearn unavailable:", e)

    print("=== (3) head accuracy on held-out ===")
    with torch.no_grad():
        _, cp, sl, _ = net(Xt.to(DEV))
    acc = float((sl.argmax(1).cpu().numpy() == sysid).mean())
    span = (CELL_HI - CELL_LO)
    Cn = np.stack([norm_cell(c) for c in cells])
    edge_mae = float((np.abs(cp.cpu().numpy() - Cn) * span)[:, :3].mean())
    print("  system_acc=%.3f  edge_MAE=%.2f A  (per-system acc: %s)" % (
        acc, edge_mae, {SYSTEMS[s]: round(float((sl.argmax(1).cpu().numpy()[sysid == s] == s).mean()), 2)
                        for s in range(7)}))

    print("=== (4) head-to-head vs classical index_powder (high-sym subset, clean vs noisy input) ===")
    sel = [i for i in range(len(X)) if SYSTEMS[sysid[i]] in HIGHSYM][:40]
    for nz in [0.0, 0.05]:
        ae_ok = cl_ok = 0; t_ae = t_cl = 0.0
        for i in sel:
            prof = X[i] + (nz * np.random.default_rng(i).standard_normal(len(QGRID))).astype(np.float32)
            true_sys = SYSTEMS[sysid[i]]
            # AE head
            t = time.perf_counter()
            with torch.no_grad():
                _, cp, sl, _ = net(torch.tensor(prof).view(1, 1, -1).to(DEV))
            t_ae += time.perf_counter() - t
            ae_ok += int(SYSTEMS[int(sl.argmax(1))] == true_sys)
            # classical: profile -> peaks -> index_powder
            t = time.perf_counter()
            try:
                pk = peaks_from_profile(QGRID, prof, max_peaks=25)
                sols = index_powder(pk, units="q", amin=2, amax=20) if len(pk) >= 4 else []
                cl_ok += int(bool(sols) and sols[0].system == true_sys)
            except Exception:
                pass
            t_cl += time.perf_counter() - t
        n = len(sel)
        print("  noise=%.2f | AE sys-acc=%.2f (%.2f ms/pat) | classical sys-acc=%.2f (%.0f ms/pat)" %
              (nz, ae_ok / n, 1e3 * t_ae / n, cl_ok / n, 1e3 * t_cl / n))
    print("DONE")


if __name__ == "__main__":
    main()
