"""Paper figure: CBXD blind-solvability phase diagram (deliverable 3, issue #11).

Three synchronized 3D surfaces (COM / PTS / TAN) over (NA, streak-point noise), height = %
solved, surface color = observed median #streaks at that NA (per Stefano's scoped plan: NA x
noise is the swept grid, #streaks/length is an OBSERVED statistic on top, not an independent
axis). Reads directly from the checkpointed run (results_grid.jsonl) and the cached orientation
pools/streak-length stats -- no hand-copied numbers, so this stays correct if the grid is
extended (more NA values, a finer noise ladder, etc.) without editing this file.

Web-artifact attempts at this figure (Chart.js, then hand-rolled SVG, then Plotly.js) all failed
to render reliably across a few tries -- this script sidesteps that whole failure class: a
committed, reproducible script producing a real static file is also just the right deliverable
for an actual paper figure, not a workaround.

  python plot_phase_diagram.py
"""
import collections
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

YUAN_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, YUAN_DIR)
sys.path.insert(0, os.path.join(YUAN_DIR, ".."))
RESULTS = os.path.join(YUAN_DIR, "results_grid.jsonl")
GRID_DIR = os.path.join(YUAN_DIR, "data", "grid")
OUTDIR = os.path.dirname(__file__)

from generate_dataset_grid import NA_GRID  # single source of truth -- extend there, not here

NA_LIST = list(NA_GRID)
NOISE_LADDER = [2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
ARMS = ["COM", "PTS", "TAN"]
_NA_CMAP = LinearSegmentedColormap.from_list("na_ramp", ["#aebbef", "#262c82"])
NA_COLORS = [matplotlib.colors.to_hex(_NA_CMAP(i / max(len(NA_LIST) - 1, 1)))
            for i in range(len(NA_LIST))]


def load_grid():
    """(arm -> {na -> [pct_solved or nan per noise rung]}), plus na -> median #streaks."""
    rows = [json.loads(l) for l in open(RESULTS)]
    by_cell = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by_cell[(r["na"], r["noise"])][r["arm"]].append(r["frac"])

    z = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for na in NA_LIST:
            row = []
            for noise in NOISE_LADDER:
                fracs = by_cell.get((na, noise), {}).get(arm)
                row.append(100.0 * sum(f > 0.7 for f in fracs) / len(fracs) if fracs else np.nan)
            z[arm][na] = row
    return z


def load_na_streaks():
    out = {}
    for na in NA_LIST:
        key = f"{na:.4f}".rstrip("0").rstrip(".")
        path = os.path.join(GRID_DIR, f"orientations_na{key}.npz")
        out[na] = float(np.median(np.load(path)["n_streaks"]))
    return out


def plot_surface(ax, na_streaks, z_by_na, color):
    """One arm's surface: X=noise index (log-labeled), Y=NA index, Z=%solved, flat per-row
    color (since the observed #streaks statistic is constant within a NA row in this grid)."""
    X, Y = np.meshgrid(np.arange(len(NOISE_LADDER)), np.arange(len(NA_LIST)))
    Z = np.array([z_by_na[na] for na in NA_LIST])

    facecolors = np.empty(X.shape + (4,))
    for i, na in enumerate(NA_LIST):
        facecolors[i, :, :] = matplotlib.colors.to_rgba(color[i], alpha=0.92)

    Zm = np.ma.masked_invalid(Z)
    ax.plot_surface(X, Y, Zm, facecolors=facecolors, rstride=1, cstride=1,
                    linewidth=0.4, edgecolor="white", antialiased=True, shade=False)

    ax.set_xticks(range(len(NOISE_LADDER)))
    ax.set_xticklabels([f"{n:.0e}".replace("e-0", "e-") for n in NOISE_LADDER],
                       fontsize=6.5, rotation=30, ha="right")
    ax.set_yticks(range(len(NA_LIST)))
    ax.set_yticklabels([f"{na:.3f}" for na in NA_LIST], fontsize=7)
    ax.set_zlim(0, 100)
    ax.set_xlabel("noise (Å⁻¹)", fontsize=8, labelpad=6)
    ax.set_ylabel("NA", fontsize=8, labelpad=2)
    ax.set_zlabel("% solved", fontsize=8, labelpad=2)
    ax.view_init(elev=22, azim=-55)
    ax.tick_params(axis="z", labelsize=7)


def main():
    z = load_grid()
    na_streaks = load_na_streaks()

    fig = plt.figure(figsize=(13.5, 4.6))
    fig.suptitle("Blind CBXD solvability: COM / PTS / TAN over (NA, streak-point noise)",
                fontsize=12, y=1.02)
    for i, arm in enumerate(ARMS):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        plot_surface(ax, na_streaks, z[arm], NA_COLORS)
        ax.set_title(arm, fontsize=10, pad=0)

    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=NA_COLORS[i], markersize=9)
              for i in range(len(NA_LIST))]
    labels = [f"NA={na:.3f}  ({na_streaks[na]:.0f} streaks)" for na in NA_LIST]
    fig.legend(handles, labels, loc="lower center", ncol=len(NA_LIST), fontsize=7.5, frameon=False,
              bbox_to_anchor=(0.5, -0.06))

    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    png_path = os.path.join(OUTDIR, "phase_diagram.png")
    pdf_path = os.path.join(OUTDIR, "phase_diagram.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")

    plot_tan_vs_pts(z)


def plot_tan_vs_pts(z, noise_focus=1e-3):
    """Companion 2D figure: the specific, mechanistic finding Stefano's review asked for --
    does TAN survive where PTS (which leans on the centroid-adjacent point-pooling) struggles?
    Success% vs NA at one representative noise level, PTS and TAN side by side. This is the
    plain, direct view of the divergence that's easy to miss inside the 3D surfaces."""
    j = NOISE_LADDER.index(noise_focus)
    na_arr = np.array(NA_LIST)
    pts = np.array([z["PTS"][na][j] for na in NA_LIST])
    tan = np.array([z["TAN"][na][j] for na in NA_LIST])
    com = np.array([z["COM"][na][j] for na in NA_LIST])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(na_arr, pts, "o-", color="#7b8dde", label="PTS", linewidth=2, markersize=6)
    ax.plot(na_arr, tan, "o-", color="#262c82", label="TAN", linewidth=2, markersize=6)
    ax.plot(na_arr, com, "o--", color="#b5502e", label="COM", linewidth=1.5, markersize=5, alpha=0.8)
    ax.set_xlabel("NA")
    ax.set_ylabel("% solved")
    ax.set_title(f"PTS vs TAN vs COM at noise={noise_focus:.0e} Å⁻¹")
    ax.set_ylim(-3, 103)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    png_path = os.path.join(OUTDIR, "tan_vs_pts.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
