"""Buat berkas label train dan kandidat test untuk tiga aturan otomatis, plus probe linear DINOv3.

Group gabungan: cos DINOv3 >= 0,9 atau pHash dihedral Hamming <= 4, hanya pasangan train.
Ambang 10 ditolak: kesepakatan label hanya 0,63 dan merantai citra berlatar polos (lihat eda_104).
Hold-out sesuai docs/lock.md: StratifiedGroupKFold 5, fold 0, seed 42. Keyakinan: porsi suara >= 0,75.
Tanpa temuan.

Keluaran: output/label_folds_{varian}.csv, output/kandidat_test_{varian}_vote.csv, output/kandidat_test_{varian}_lr.csv

    python temp/eda_110_fusi/label_final.py
"""
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from kosakata import NAMES

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 300)
g = pd.read_csv(ROOT / "temp" / "eda_102_sumber" / "output" / "meta_assign.csv", dtype={"image_id": str})
tr = g[g.split == "train"].sort_values("image_id").reset_index(drop=True)
te = g[g.split == "test"].sort_values("image_id").reset_index(drop=True)
nt = len(tr)
E_tr, E_te = np.load(E6 / "emb_train.npy"), np.load(E6 / "emb_test.npy")

# group gabungan, hanya pasangan train
S = E_tr @ E_tr.T
np.fill_diagonal(S, 0)
a, b = np.where(S >= .9)
ph = pd.read_csv(ROOT / "temp" / "eda_104_duplikat" / "output" / "pasangan_hash.csv")
ph = ph[(ph.sa == "train") & (ph.sb == "train") & (ph.ham <= 4)]
ii = np.r_[a, ph.a.values]
jj = np.r_[b, ph.b.values]
n_g, grp = connected_components(csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(nt, nt)), directed=False)
sz = pd.Series(grp).value_counts()
print("group", n_g, "terbesar", int(sz.max()), "citra dalam group > 1:", int(sz[sz > 1].sum()))
g09 = connected_components(csr_matrix((np.ones(len(a)), (a, b)), shape=(nt, nt)), directed=False)[0]
print("group cos saja", g09, "; tambahan penggabungan oleh pHash:", g09 - n_g)

lama = pd.read_csv(E6 / "draft_submission_dinov3_cluster.csv", dtype=str).set_index("image_id").loc[te.image_id]["waste_category"].values
hasil = {}
for var in ("t1_in1k_coco_abst", "t05_abst", "t1_in1k"):
    V = np.load(OUT / f"var_{var}.npy")
    y = V.argmax(1)
    suara = V.max(1)
    pakai = suara[:nt] >= .75
    df = pd.DataFrame({"image_id": tr.image_id, "label": np.array(NAMES)[y[:nt]], "suara": suara[:nt].round(4), "group": grp, "fam": tr.fam})
    p = df[pakai].reset_index(drop=True)
    sg = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    _, va = next(sg.split(p, p.label, p.group))
    p["fold"] = 1
    p.loc[va, "fold"] = 0
    assert not (p.groupby("group").fold.nunique() > 1).any()
    p.to_csv(OUT / f"label_folds_{var}.csv", index=False)

    clf = LogisticRegression(C=3, max_iter=3000)
    idx = df.index[pakai].values
    clf.fit(E_tr[idx][p.fold.values == 1], p.label[p.fold == 1])
    acc = float((clf.predict(E_tr[idx][p.fold.values == 0]) == p.label[p.fold == 0]).mean())
    per = pd.DataFrame({"y": p.label[p.fold == 0].values, "ok": clf.predict(E_tr[idx][p.fold.values == 0]) == p.label[p.fold == 0].values}).groupby("y").ok.mean().round(3).to_dict()
    clf = LogisticRegression(C=3, max_iter=3000).fit(E_tr[idx], p.label)
    lr = clf.predict(E_te)
    vote = np.array(NAMES)[y[nt:]]
    pd.DataFrame({"image_id": te.image_id, "waste_category": vote}).to_csv(OUT / f"kandidat_test_{var}_vote.csv", index=False)
    pd.DataFrame({"image_id": te.image_id, "waste_category": lr}).to_csv(OUT / f"kandidat_test_{var}_lr.csv", index=False)
    hasil[var] = lr
    print(f"\n== {var}: label {len(p)} dari {nt}, hold-out {int((p.fold == 0).sum())}, kesepakatan LR hold-out {acc:.4f}")
    print("label:", p.label.value_counts().reindex(NAMES).to_dict())
    print("per kelas hold-out:", per)
    print("vote test:", dict(zip(NAMES, [int((vote == c).sum()) for c in NAMES])))
    print("lr   test:", dict(zip(NAMES, [int((lr == c).sum()) for c in NAMES])))
    print("setuju vote vs lr:", round(float((vote == lr).mean()), 4), "| lr vs EXP-006:", round(float((lr == lama).mean()), 4))
    print(pd.crosstab(te.fam, pd.Series(lr, name="lr")).reindex(columns=NAMES).fillna(0).astype(int).to_string())

v = list(hasil)
for i in range(3):
    for j in range(i + 1, 3):
        print(v[i], v[j], "setuju lr:", round(float((hasil[v[i]] == hasil[v[j]]).mean()), 4))
