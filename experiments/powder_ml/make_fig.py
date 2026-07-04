"""Thread-A deliverable figure: (a) denoising, (b) latent PCA by system, (c) cell-regression scatter.
  conda activate ana-4.0.58-py3-minipytorch ; python make_fig.py"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim_dataset import make_dataset, QGRID, SYSTEMS, CELL_LO, CELL_HI, denorm_cell
from model import PowderAE

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
net = PowderAE().to(DEV)
net.load_state_dict(torch.load(os.path.join(HERE, "ae.pt"), map_location=DEV)["state"]); net.eval()

X, cells, sysid, cens = make_dataset(1500, seed=123)
Xt = torch.tensor(X).unsqueeze(1)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

# (a) denoising
i = next(k for k in range(len(X)) if SYSTEMS[sysid[k]] == "tetragonal")
clean = X[i]; nz = 0.08
noisy = clean + (nz * np.random.default_rng(1).standard_normal(len(clean))).astype(np.float32)
with torch.no_grad():
    recon = net(torch.tensor(noisy).view(1, 1, -1).to(DEV))[0].cpu().numpy().ravel()
ax[0].plot(QGRID, noisy, color="0.75", lw=0.7, label=f"noisy input (σ={nz})")
ax[0].plot(QGRID, clean, color="C0", lw=1.3, label="clean truth")
ax[0].plot(QGRID, recon, color="C3", lw=1.3, ls="--", label="AE reconstruction")
ax[0].set_title(f"(a) Denoising — {SYSTEMS[sysid[i]]}"); ax[0].set_xlabel("q (Å$^{-1}$)")
ax[0].set_ylabel("I (norm)"); ax[0].legend(fontsize=8)

# (b) latent PCA by system
with torch.no_grad():
    z = net.encode(Xt.to(DEV)).cpu().numpy()
zc = z - z.mean(0); _, _, Vt = np.linalg.svd(zc, full_matrices=False); z2 = zc @ Vt[:2].T
for s in range(7):
    m = sysid == s
    ax[1].scatter(z2[m, 0], z2[m, 1], s=6, alpha=0.5, label=SYSTEMS[s])
ax[1].set_title("(b) Latent (PCA) by system — overlap ≈ chance"); ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
ax[1].legend(fontsize=7, ncol=2, markerscale=1.6)

# (c) cell-regression scatter (edge a)
with torch.no_grad():
    _, cp, sl, _ = net(Xt.to(DEV))
pred = denorm_cell(cp.cpu().numpy()); mae = float(np.abs(pred[:, :3] - cells[:, :3]).mean())
ax[2].scatter(cells[:, 0], pred[:, 0], s=6, alpha=0.4, c="C2")
lim = [float(CELL_LO[0]), float(CELL_HI[0])]; ax[2].plot(lim, lim, "k--", lw=1)
ax[2].set_xlim(lim); ax[2].set_ylim(lim)
ax[2].set_title(f"(c) Cell regression — edge MAE {mae:.1f} Å"); ax[2].set_xlabel("true a (Å)"); ax[2].set_ylabel("predicted a (Å)")

plt.tight_layout()
out = os.path.join(HERE, "powder_ae.png"); plt.savefig(out, dpi=130)
acc = float((sl.argmax(1).cpu().numpy() == sysid).mean())
print(f"wrote {out}  sys_acc={acc:.3f}  edge_MAE={mae:.2f} A")
