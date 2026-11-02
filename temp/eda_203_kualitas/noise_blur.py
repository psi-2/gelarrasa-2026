"""EDA 203: noise (Immerkaer, resolusi asli), blur (variance Laplacian pada 256 px), kontras, blockiness JPEG.

Fitur dipakai ulang dari eda_107 visual.csv. Montage citra terburuk disimpan untuk dilihat manusia.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from eda_common import DATA, VISUAL, kelompok, meta, montage, out_dir

OUT = out_dir(__file__)
v = pd.read_csv(VISUAL).merge(meta()[["split", "image_id", "fam", "ext", "w", "h"]], on=["split", "image_id"])
v["kontras"] = v.v_std
v["log_lap"] = np.log10(v.lap_var + 1e-6)
AMBANG = {"noise": ("noise", ">", .03), "buram": ("lap_var", "<", 5e-4), "kontras_rendah": ("kontras", "<", .10), "blok": ("blk8", ">", 1.5)}

for s, d in v.groupby("split"):
    print(f"\n== {s}")
    for c in ("noise", "lap_var", "kontras", "blk8"):
        print(f"{c:8s} q05/q50/q95: {np.percentile(d[c], [5, 50, 95]).round(5)}")
    for k, (c, op, t) in AMBANG.items():
        f = d[c] > t if op == ">" else d[c] < t
        print(f"  {k:15s} ({c}{op}{t}): {f.mean():.3f} ({int(f.sum())})")
tr, te = v[v.split == "train"], v[v.split == "test"]
for c in ("noise", "log_lap", "kontras", "blk8"):
    print(f"KS {c}: {ks_2samp(tr[c], te[c]).statistic:.3f}")

g = kelompok(8)
t = tr.merge(g, on="image_id")
print("\nmedian per cluster K=8 (label lama dominan):")
dom = t.groupby("cluster").label_lama.agg(lambda x: x.value_counts().index[0])
tab = t.groupby("cluster")[["noise", "lap_var", "kontras", "blk8"]].median().round(4)
tab["dominan"] = dom
tab["n_noise"] = t.groupby("cluster").noise.apply(lambda x: int((x > .03).sum()))
print(tab.to_string())
print("\nnoise>0.03 per keluarga:", v[v.noise > .03].groupby(["fam", "split"]).size().unstack(fill_value=0).to_dict())

fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for s, c in (("train", "C0"), ("test", "C3")):
    d = v[v.split == s]
    ax[0].hist(np.log10(d.noise + 1e-5), 40, alpha=.5, density=True, color=c, label=s)
    ax[1].hist(d.log_lap, 40, alpha=.5, density=True, color=c, label=s)
    ax[2].hist(d.kontras, 40, alpha=.5, density=True, color=c, label=s)
ax[0].axvline(np.log10(.03), c="k", ls="--"); ax[0].set(title="log10 noise")
ax[1].axvline(np.log10(5e-4), c="k", ls="--"); ax[1].set(title="log10 var Laplacian (blur)")
ax[2].axvline(.10, c="k", ls="--"); ax[2].set(title="kontras (std V)")
ax[3].boxplot([t[t.cluster == k].noise for k in range(8)], tick_labels=[f"{k}\n{dom[k][:5]}" for k in range(8)], showfliers=False)
ax[3].set(title="noise per cluster K=8 (train)")
for a in ax[:3]:
    a.legend()
fig.tight_layout()
fig.savefig(OUT / "noise_blur.png", dpi=120)

def pth(r):
    return next((DATA / r.split).glob(f"{r.image_id:0{4 if r.split == 'train' else 3}d}.*"))
for nama, d in (("noise_tertinggi", v.nlargest(16, "noise")), ("paling_buram", v.nsmallest(16, "lap_var"))):
    montage([pth(r) for r in d.itertuples()], [f"{r.split[:2]} {r.image_id} {r.fam[2:9]}" for r in d.itertuples()], OUT / f"montage_{nama}.png")
print("simpan noise_blur.png, montage_noise_tertinggi.png, montage_paling_buram.png di", OUT)
