"""Penamaan cluster otomatis yang di-bagging, train-only.

Untuk tiap (K, seed): KMeans pada train, nama cluster = argmax jumlah log-odds tiga kosakata yang dinormalkan prior.
Semua keputusan nama keluar dari rumus ini. Test diberi suara lewat pusat cluster terdekat, tidak ikut fit.
Suara per citra dirata-rata lintas run menjadi distribusi 8 kelas.

Keluaran: output/bag_votes.npy (2400 x 8), output/label_train_bag.csv, output/kandidat_test_bag.csv

    python temp/eda_110_fusi/bagging.py
"""
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from kosakata import NAMES

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 300)

g = pd.read_csv(ROOT / "temp" / "eda_102_sumber" / "output" / "meta_assign.csv", dtype={"image_id": str})
tr = g[g.split == "train"].sort_values("image_id")
te = g[g.split == "test"].sort_values("image_id")
d = pd.concat([tr, te], ignore_index=True)
nt, N = len(tr), len(d)
E_tr, E_te = np.load(E6 / "emb_train.npy"), np.load(E6 / "emb_test.npy")
pca = PCA(128, random_state=42).fit(E_tr)


def z(x):
    v = pca.transform(x)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


Ztr, Zte = z(E_tr), z(E_te)
S = {k: np.load(OUT / f"skor_{'conv' if k == 'in1k' else k}.npy") for k in ("in1k", "coco", "ade")}
PRIOR = {k: v[:nt].mean(0) + 1e-6 for k, v in S.items()}


def nama_cluster(lab, k, views):
    L = np.zeros((k, 8))
    for w in views:
        v = S[w][:nt] / PRIOR[w]
        r = np.stack([v[lab == c].mean(0) if (lab == c).any() else np.ones(8) for c in range(k)])
        r = r / r.sum(1, keepdims=True)
        L += np.log(r + 1e-3)
    return L.argmax(1)


def bag(views, Ks=(48, 64, 96), seeds=range(12)):
    V = np.zeros((N, 8))
    run = 0
    for k in Ks:
        for s in seeds:
            km = KMeans(k, n_init=1, random_state=1000 * k + s).fit(Ztr)
            nm = nama_cluster(km.labels_, k, views)
            a = np.r_[km.labels_, km.predict(Zte)]
            V[np.arange(N), nm[a]] += 1
            run += 1
    return V / run


hasil = {}
for nama, views in (("semua", ["in1k", "coco", "ade"]), ("in1k", ["in1k"]), ("in1k_coco", ["in1k", "coco"])):
    V = bag(views)
    hasil[nama] = V
    np.save(OUT / f"bag_{nama}.npy", V.astype(np.float32))
    y = V.argmax(1)
    print(f"{nama:10s} test  {dict(zip(NAMES, np.bincount(y[nt:], minlength=8).tolist()))}")
    print(f"{'':10s} train {dict(zip(NAMES, np.bincount(y[:nt], minlength=8).tolist()))}  median suara {np.median(V.max(1)[:nt]):.3f}", flush=True)

V = hasil["semua"]
y = V.argmax(1)
y1 = hasil["in1k"].argmax(1)
print("\nporsi suara teratas train, kuantil:", np.quantile(V.max(1)[:nt], [.1, .25, .5, .75]).round(3).tolist())
for th in (.6, .75, .9):
    print(f"suara >= {th}: train {(V.max(1)[:nt] >= th).sum()} | test {(V.max(1)[nt:] >= th).sum()}")
print("\ntest per keluarga (bukan bagian fit)")
print(pd.crosstab(d.fam[nt:].values, np.array(NAMES)[y[nt:]]).reindex(columns=NAMES).fillna(0).astype(int).to_string())
print("\ntrain per keluarga")
print(pd.crosstab(d.fam[:nt].values, np.array(NAMES)[y[:nt]]).reindex(columns=NAMES).fillna(0).astype(int).to_string())
print("\nsetuju dengan bag in1k saja, test:", round(float((y == y1)[nt:].mean()), 4))
print(pd.crosstab(pd.Series(np.array(NAMES)[y1[nt:]], name="in1k"), pd.Series(np.array(NAMES)[y[nt:]], name="semua")).reindex(index=NAMES, columns=NAMES).fillna(0).astype(int).to_string())

R = np.load(OUT / "rt_semua.npy")
print("\nsetuju bag vs EM train-only, test:", round(float((R.argmax(1) == y)[nt:].mean()), 4), "train:", round(float((R.argmax(1) == y)[:nt].mean()), 4))
lama = pd.read_csv(E6 / "draft_submission_dinov3_cluster.csv", dtype=str).set_index("image_id").loc[d.image_id[nt:]]["waste_category"].values
print("setuju bag vs EXP-006, test:", round(float((np.array(NAMES)[y[nt:]] == lama).mean()), 4))
print(pd.crosstab(pd.Series(lama, name="exp006"), pd.Series(np.array(NAMES)[y[nt:]], name="bag")).reindex(index=NAMES, columns=NAMES).fillna(0).astype(int).to_string())

pd.DataFrame({"image_id": d.image_id[:nt], "label": np.array(NAMES)[y[:nt]], "suara": V[:nt].max(1).round(4), "fam": d.fam[:nt]}).to_csv(OUT / "label_train_bag.csv", index=False)
pd.DataFrame({"image_id": d.image_id[nt:], "waste_category": np.array(NAMES)[y[nt:]]}).to_csv(OUT / "kandidat_test_bag.csv", index=False)
