"""Bahan bersama tahap 3. Hanya CPU, maksimal 4 thread. Tanpa temuan.

Menyatukan label v4, keluarga sumber, group, embedding ViT dan CNN, prediksi model v2, dan LR cross-fit train-only.
Keluaran: output/dasar.csv, output/pl_crossfit.npy (1920 x 8, NaN untuk citra tanpa label)

    python temp/eda_113_elektronik/dasar.py
"""
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
RC = ROOT / "temp" / "research_uji_crossfit" / "output"
E10 = ROOT / "temp" / "exp_010_dua_backbone" / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
ID = {c: i for i, c in enumerate(NAMES)}
nt = 1920
pd.set_option("display.width", 230, "display.max_columns", 40, "display.max_rows", 200)

ids = np.load(E6 / "ids_train.npy").astype(str)
ids_te = np.load(E6 / "ids_test.npy").astype(str)
Ev, Ev_te = np.load(E6 / "emb_train.npy"), np.load(E6 / "emb_test.npy")
ic, ic_te = np.load(RC / "ids_cnn_train.npy").astype(str), np.load(RC / "ids_cnn_test.npy").astype(str)
Ec, Ec_te = np.load(RC / "emb_cnn_train.npy"), np.load(RC / "emb_cnn_test.npy")
print("embedding CNN", Ec.shape, Ec_te.shape, "urutan sama:", (ic == ids).all(), (ic_te == ids_te).all())
assert (ic == ids).all() and (ic_te == ids_te).all()

v4 = pd.read_csv(ROOT / "temp" / "exp_011_label_v4" / "output" / "label_folds_v4.csv", dtype={"image_id": str})
print("v4", len(v4), "group kosong", int(v4.group.isna().sum()), "fam kosong", int(v4.fam.isna().sum()), "fold", v4.fold.value_counts().to_dict())
m = pd.read_csv(ROOT / "temp" / "eda_102_sumber" / "output" / "meta_assign.csv", dtype={"image_id": str})
fam = m[m.split == "train"].set_index("image_id").loc[ids].fam.values
fam_te = m[m.split == "test"].set_index("image_id").loc[ids_te].fam.values

# group ulang persis label_final.py, agar citra di luar v4 juga punya group
S = Ev @ Ev.T
np.fill_diagonal(S, 0)
a, b = np.where(S >= .9)
ph = pd.read_csv(ROOT / "temp" / "eda_104_duplikat" / "output" / "pasangan_hash.csv")
ph = ph[(ph.sa == "train") & (ph.sb == "train") & (ph.ham <= 4)]
ii, jj = np.r_[a, ph.a.values], np.r_[b, ph.b.values]
_, grp = connected_components(csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(nt, nt)), directed=False)

d = pd.DataFrame({"image_id": ids, "fam": fam, "grp": grp}).merge(v4[["image_id", "label", "suara", "group", "fold"]], on="image_id", how="left")
c = d[d.group.notna()]
print("group v4 sama dengan hitung ulang:", bool((c.group.astype(int) == c.grp).all()), "| baris v4 tanpa group:", int((d.label.notna() & d.group.isna()).sum()))
for tag in ("vit_v2", "cnn_v2"):
    oof = np.load(E10 / tag / "oof_fold0.npy")
    print(tag, "oof", oof.shape, "sisa", np.load(E10 / tag / "sisa_fold0.npy").shape, "test", np.load(E10 / tag / "test_fold0.npy").shape)

lab = d.label.notna().values
idx = np.where(lab)[0]
y = d.label[idx].map(ID).values
Xv, Xc = normalize(Ev), normalize(Ec)
X = np.hstack([Xv, Xc])
pl = np.full((nt, 8), np.nan)
pu = np.zeros((nt, 8))
for f, (tr_, va_) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(idx, y, grp[idx])):
    mdl = LogisticRegression(C=3, max_iter=2000).fit(X[idx[tr_]], y[tr_])
    pl[idx[va_]] = mdl.predict_proba(X[idx[va_]])
    pu[~lab] += mdl.predict_proba(X[~lab]) / 5
pl[~lab] = pu[~lab]
np.save(OUT / "pl_crossfit.npy", pl)
d["pl"] = np.array(NAMES)[pl.argmax(1)]
for i, c_ in enumerate(NAMES):
    d["p_" + c_] = pl[:, i].round(4)
d["fam_"] = d.fam
d.to_csv(OUT / "dasar.csv", index=False)
print("LR cross-fit setuju v4:", round(float((d.pl == d.label)[lab].mean()), 4))
print("salah LR per pasangan terbesar:")
s = d[lab & (d.pl != d.label)]
print(s.groupby(["label", "pl"]).size().sort_values(ascending=False).head(12).to_string())

print("\n== train v4: label per keluarga ==")
t = pd.crosstab(d.fam, d.label.fillna("(kosong)"))
print(t.reindex(columns=NAMES + ["(kosong)"]).fillna(0).astype(int).assign(n=t.sum(1)).to_string())
print("total per kelas v4:", d.label.value_counts().reindex(NAMES).to_dict())

pv, pc = [np.load(E10 / t_ / "test_fold0.npy") for t_ in ("vit_v2", "cnn_v2")]
g = (pv + pc) / 2
te = pd.DataFrame({"image_id": ids_te, "fam": fam_te, "pred": np.array(NAMES)[g.argmax(1)], "conf": g.max(1).round(4), "p_el": g[:, 2].round(4), "p_me": g[:, 3].round(4)})
te.to_csv(OUT / "dasar_test.csv", index=False)
print("\n== test: prediksi gabungan per keluarga (bukan fit) ==")
t = pd.crosstab(te.fam, te.pred)
print(t.reindex(columns=NAMES).fillna(0).astype(int).assign(n=t.sum(1)).to_string())
print("\ntest berprediksi metal per keluarga, kuantil p_electronic:")
print(te[te.pred == "metal"].groupby("fam").p_el.describe().round(3).to_string())
print("\ntrain v4: label metal per keluarga, kuantil p_electronic cross-fit:")
print(d[d.label == "metal"].groupby("fam").p_electronic.describe().round(3).to_string())
