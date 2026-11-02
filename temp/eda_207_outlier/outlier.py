"""EDA 207: outlier train dari tiga sudut: embedding (jarak kNN + LOF, ViT-L cache), metadata ekstrem, kualitas.

Outlier embedding adalah kandidat citra yang tidak cocok dengan cluster mana pun. Montage untuk dilihat manusia.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from eda_common import DATA, VISUAL, emb, kelompok, meta, montage, out_dir

OUT = out_dir(__file__)
tr, _ = emb("vit")
S = tr @ tr.T
np.fill_diagonal(S, -1)
knn = 1 - np.sort(S, 1)[:, -10:].mean(1)
lof = -LocalOutlierFactor(20, metric="cosine").fit(tr).negative_outlier_factor_
m = meta(); m = m[m.split == "train"].sort_values("image_id").reset_index(drop=True)
v = pd.read_csv(VISUAL); v = v[v.split == "train"].sort_values("image_id").reset_index(drop=True)
g = kelompok(8)
d = pd.DataFrame({"image_id": np.arange(1, 1921), "knn10": knn, "lof": lof, "fam": m.fam, "pendek": np.minimum(m.w, m.h),
                  "ar": m.w / m.h, "noise": v.noise, "lap_var": v.lap_var, "label_lama": g.label_lama, "suara": g.suara})
z = lambda x: (x - x.median()) / (1.4826 * (x - x.median()).abs().median())
d["z_knn"], d["z_lof"] = z(d.knn10), z(d.lof)
d["out_emb"] = (d.z_knn > 3.5) | (d.z_lof > 3.5)
d["out_meta"] = (d.pendek < 128) | (d.ar > 2.2) | (d.ar < 0.45)
d["out_kual"] = (d.noise > .05) | (d.lap_var < 2e-4)
print("knn10 jarak q50/q95/q99/maks:", np.percentile(knn, [50, 95, 99, 100]).round(3))
for c in ("out_emb", "out_meta", "out_kual"):
    print(f"{c}: {int(d[c].sum())}")
print("irisan emb & kualitas:", int((d.out_emb & d.out_kual).sum()), "| emb & meta:", int((d.out_emb & d.out_meta).sum()))
o = d[d.out_emb]
print("\noutlier embedding per keluarga:", o.fam.value_counts().to_dict())
print("outlier embedding per label lama:", o.label_lama.value_counts().to_dict())
print(f"suara label lama median: outlier {o.suara.median():.2f} vs semua {d.suara.median():.2f}")
print(f"korelasi Spearman knn10 vs suara label lama: {d[['knn10', 'suara']].corr('spearman').iloc[0, 1]:.3f}")
d.to_csv(OUT / "outlier.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(knn, 60); ax[0].axvline(d.knn10[d.z_knn > 3.5].min() if d.out_emb.any() else knn.max(), c="r", ls="--")
ax[0].set(title="jarak cos rata-rata ke 10 NN (train)")
ax[1].scatter(d.knn10, d.suara, s=4, alpha=.4); ax[1].set(xlabel="jarak kNN", ylabel="porsi suara label lama", title="outlier vs keyakinan label")
fig.tight_layout(); fig.savefig(OUT / "outlier.png", dpi=120)
top = d.nlargest(24, "knn10")
montage([next((DATA / "train").glob(f"{i:04d}.*")) for i in top.image_id], [f"{r.image_id} {r.label_lama[:5]} {r.suara:.2f}" for r in top.itertuples()], OUT / "montage_outlier_emb.png")
print("simpan outlier.png, outlier.csv, montage_outlier_emb.png di", OUT)
