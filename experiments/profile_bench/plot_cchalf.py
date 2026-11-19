"""CC1/2 against resolution, box vs profile, from cchalf.py's JSON (merged and anomalous panels).

    python plot_cchalf.py cchalf.json cchalf_anom.json -o cchalf.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = {"box": "#2a78d6", "profile": "#eb6834"}     # categorical slots 1 and 2
INK, MUTED, GRID = "#1a1a19", "#6b6a64", "#e4e3de"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged")
    ap.add_argument("anomalous")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, path, title in zip(axes, (a.merged, a.anomalous), ("Friedel mates merged", "Friedel mates separate")):
        d = json.load(open(path))
        x = 1 / np.array(d["d_mid"]) ** 2
        for name, col in SERIES.items():
            cc = d["integrators"][name]["cchalf"]
            ax.plot(x, cc, color=col, lw=2, marker="o", ms=4, label=name)
            ax.annotate(name, (x[-2], cc[-2]), xytext=(4, 0), textcoords="offset points",
                        color=INK, fontsize=9, va="center")
        ticks = np.array([6, 4, 3, 2.5, 2.2, 2.0])
        ax.set_xticks(1 / ticks ** 2, [f"{t:g}" for t in ticks])
        ax.set_xlabel("resolution (Å)", color=MUTED)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.grid(True, color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("CC½ (merge-free, 1/mean(I) frame scale)", color=MUTED)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("CXIDB 33 Dataset A, 69,361 frames: box vs profile integration", color=INK, x=0.06, ha="left")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)


if __name__ == "__main__":
    main()
