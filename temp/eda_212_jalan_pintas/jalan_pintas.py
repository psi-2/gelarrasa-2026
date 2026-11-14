"""EDA 212: apakah cluster Run 1 terbentuk oleh atribut non-isi (latar, sumber berkas, kualitas JPEG, pita hitam, ukuran)?

Bila sebuah cluster didominasi satu nilai atribut yang jarang secara global, cluster itu mungkin dikumpulkan oleh gaya
foto, bukan material. Cluster seperti ini perlu diperiksa paling teliti saat penamaan. Kekuatan asosiasi diukur
dengan Cramer's V. Semua train-only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from eda_common import TEMP, out_dir, run1

OUT = out_dir(__file__)
cl, _, _, _ = run1()
lat = pd.read_csv(TEMP / "eda_204_background/output/latar.csv").query("split == 'train'")[["image_id", "tipe", "fg_frac"]]
ma = pd.read_csv(TEMP / "eda_102_sumber/output/meta_assign.csv").query("split == 'train'")[["image_id", "fam", "qhash", "q_ijg", "w", "h"]]
tr = pd.read_csv(TEMP / "prep_301_verifikasi/output/trim_hitam.csv").query("split == 'train'")[["image_id", "trim"]]
d = cl.merge(lat, on="image_id").merge(ma, on="image_id").merge(tr, on="image_id")
assert len(d) == 1920
top_fam = d.fam.value_counts().index[:6]
d["sumber"] = np.where(d.fam.isin(top_fam), d.fam, "lain")
d["qtab"] = np.where(d.qhash == d.qhash.value_counts().index[0], "qhash_utama", "qhash_lain")
d["pendek"] = pd.cut(d[["w", "h"]].min(1), [0, 383, 640, 1e5], labels=["<384", "384-640", ">640"]).astype(str)
d["pita_hitam"] = d.trim.map({True: "ada", False: "tidak"})
d["objek"] = pd.cut(d.fg_frac, [-1, .1, .5, 2], labels=["kecil<0.1", "sedang", "besar>0.5"]).astype(str)
ATR = ["tipe", "sumber", "qtab", "pendek", "pita_hitam", "objek"]


def cramer_v(x, y):
    t = pd.crosstab(x, y)
    chi2 = chi2_contingency(t, correction=False)[0]
    return np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))


rows, flag = [], []
for a in ATR:
    glob = d[a].value_counts(normalize=True)
    ct = pd.crosstab(d.cluster, d[a], normalize="index")
    v = cramer_v(d.cluster, d[a])
    jarang = glob[glob < .3].index
    dom = ct[jarang].max(1) if len(jarang) else pd.Series(0, ct.index)
    for k in ct.index[dom > .8]:
        nilai = ct.loc[k, jarang].idxmax()
        flag.append(dict(cluster=k, atribut=a, nilai=nilai, porsi=round(ct.loc[k, nilai], 3), porsi_global=round(glob[nilai], 3)))
    rows.append(dict(atribut=a, cramer_v=round(v, 3), nilai_global=", ".join(f"{i} {p:.2f}" for i, p in glob.items())))
R = pd.DataFrame(rows)
Fl = pd.DataFrame(flag).sort_values(["cluster", "atribut"]) if flag else pd.DataFrame()
with pd.option_context("display.width", 200, "display.max_colwidth", 120):
    print(R.to_string(index=False))
    print(f"\ncluster didominasi (> 80%) nilai atribut yang jarang secara global (< 30%): {Fl.cluster.nunique() if len(Fl) else 0}")
    print(Fl.to_string(index=False))
Fl.to_csv(OUT / "cluster_atribut_dominan.csv", index=False)
d[["image_id", "cluster"] + ATR].to_csv(OUT / "atribut_train.csv", index=False)

fig, ax = plt.subplots(1, len(ATR), figsize=(22, 9), gridspec_kw=dict(width_ratios=[d[a].nunique() for a in ATR]))
for j, a in enumerate(ATR):
    ct = pd.crosstab(d.cluster, d[a], normalize="index")
    ax[j].imshow(ct.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax[j].set_xticks(range(ct.shape[1]), ct.columns, rotation=60, fontsize=7)
    ax[j].set_yticks(range(len(ct)), ct.index, fontsize=6)
    ax[j].set_title(f"{a}\nV={R.cramer_v[j]:.2f}", fontsize=9)
fig.suptitle("komposisi atribut non-isi per cluster Run 1 (baris dinormalisasi)")
fig.tight_layout()
fig.savefig(OUT / "jalan_pintas.png", dpi=100)
print("simpan jalan_pintas.png, cluster_atribut_dominan.csv di", OUT)
