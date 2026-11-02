"""EDA 201: resolusi dan aspect ratio, train vs test. Sumber: meta.csv (eda_101), tanpa membuka piksel."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from eda_common import meta, out_dir

OUT = out_dir(__file__)
m = meta()
m["pendek"] = m[["w", "h"]].min(axis=1)
m["panjang"] = m[["w", "h"]].max(axis=1)
m["ar"] = m.w / m.h

for s, d in m.groupby("split"):
    print(f"\n== {s} n={len(d)}")
    print("sisi pendek q05/q25/q50/q75/q95:", np.percentile(d.pendek, [5, 25, 50, 75, 95]).round(0))
    print("aspect w/h  q05/q50/q95:", np.percentile(d.ar, [5, 50, 95]).round(3))
    print(f"persegi (|ar-1|<0.02): {np.mean(abs(d.ar - 1) < .02):.3f}  | ar>1.5 atau <0.67: {np.mean((d.ar > 1.5) | (d.ar < .67)):.3f}")
    for t in (224, 256, 384, 448, 518):
        print(f"  sisi pendek < {t}: {np.mean(d.pendek < t):.3f}")
    print("top ukuran:", d.groupby(["w", "h"]).size().sort_values(ascending=False).head(6).to_dict())

tr, te = m[m.split == "train"], m[m.split == "test"]
for c in ("pendek", "ar", "mp"):
    r = ks_2samp(tr[c], te[c])
    print(f"KS {c}: D={r.statistic:.3f} p={r.pvalue:.2e}")
print("\nsisi pendek per keluarga sumber (median, n):")
print(m.groupby(["fam", "split"]).pendek.agg(["median", "size"]).unstack().to_string())

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for s, c in (("train", "C0"), ("test", "C3")):
    d = m[m.split == s]
    ax[0].scatter(d.w, d.h, s=6, alpha=.4, c=c, label=s)
    ax[1].hist(np.log2(d.pendek), bins=40, alpha=.5, density=True, color=c, label=s)
    ax[2].hist(np.clip(d.ar, .3, 2.5), bins=50, alpha=.5, density=True, color=c, label=s)
ax[0].set(xscale="log", yscale="log", xlabel="lebar", ylabel="tinggi", title="ukuran (log)")
for t in (224, 384, 448):
    ax[1].axvline(np.log2(t), ls="--", c="k", lw=.8)
ax[1].set(xlabel="log2 sisi pendek (garis: 224/384/448)", title="sisi pendek")
ax[2].set(xlabel="w/h (clip 0.3-2.5)", title="aspect ratio")
for a in ax:
    a.legend()
fig.tight_layout()
fig.savefig(OUT / "resolusi.png", dpi=120)
print("simpan", OUT / "resolusi.png")
