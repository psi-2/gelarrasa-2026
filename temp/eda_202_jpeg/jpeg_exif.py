"""EDA 202: format, kualitas JPEG (estimasi IJG), subsampling, progressive, EXIF. Sumber: meta.csv eda_101."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from eda_common import meta, out_dir

OUT = out_dir(__file__)
m = meta()
m["q"] = m.q_ijg
for s, d in m.groupby("split"):
    print(f"\n== {s} n={len(d)}")
    print("format/mode:", d.groupby(["fmt", "mode"]).size().to_dict())
    print("ekstensi:", d.ext.value_counts().to_dict())
    print("q IJG q05/q25/q50/q75/q95:", np.nanpercentile(d.q, [5, 25, 50, 75, 95]).round(1))
    print(f"q<=60: {np.mean(d.q <= 60):.3f}  q<=75: {np.mean(d.q <= 75):.3f}  q>=90: {np.mean(d.q >= 90):.3f}")
    print("subsampling:", d.subsamp.value_counts().to_dict())
    print(f"progressive: {d.progressive.mean():.3f}  ICC: {d.icc.notna().mean() if d.icc.dtype == object else d.icc.mean():.3f}")
    print(f"EXIF ada: {(d.exif_len > 0).mean():.3f}  Orientation!=1: {int((d.exif_Orientation.fillna(1) != 1).sum())}")
    print("bpp q05/q50/q95:", np.nanpercentile(d.bpp, [5, 50, 95]).round(3))
    print("qhash top3 porsi:", (d.qhash.value_counts(normalize=True).head(3)).round(3).to_dict())
tr, te = m[m.split == "train"], m[m.split == "test"]
print(f"\nKS q: {ks_2samp(tr.q.dropna(), te.q.dropna()).statistic:.3f}  KS bpp: {ks_2samp(tr.bpp.dropna(), te.bpp.dropna()).statistic:.3f}")
print("EXIF kamera/software (non-kosong):")
print(m[["exif_Make", "exif_Model", "exif_Software"]].dropna(how="all").value_counts().head(10).to_string())
print("\nq IJG per keluarga (median):")
print(m.groupby(["fam", "split"]).q.median().unstack().round(0).to_string())

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for s, c in (("train", "C0"), ("test", "C3")):
    d = m[m.split == s]
    ax[0].hist(d.q.dropna(), bins=np.arange(30, 101, 2), alpha=.5, density=True, color=c, label=s)
    ax[1].hist(np.log10(d.bpp.dropna() + 1e-3), bins=40, alpha=.5, density=True, color=c, label=s)
ax[0].set(xlabel="estimasi kualitas IJG", title="kualitas JPEG")
ax[1].set(xlabel="log10 bit per piksel", title="kompresi")
fam = m.groupby(["fam", "split"]).size().unstack(fill_value=0)
(fam / fam.sum()).plot.barh(ax=ax[2], color=["C3", "C0"])
ax[2].set(title="porsi keluarga sumber per split", xlabel="porsi")
for a in ax[:2]:
    a.legend()
fig.tight_layout()
fig.savefig(OUT / "jpeg_exif.png", dpi=120)
print("simpan", OUT / "jpeg_exif.png")
