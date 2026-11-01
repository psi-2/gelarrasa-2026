"""Reproduksi cluster K=64 EXP-006 persis, atau bila gagal beri nama ulang dengan aturan massa yang sama.

Keluaran: output/assign.csv (cluster dan nama semua citra; test dari pusat terdekat, tidak ikut fit).

    python temp/eda_102_sumber/assign.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
E5 = ROOT / "temp" / "exp_005_imagenet_map" / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]

emb = np.load(E6 / "emb_train.npy")
ids = np.load(E6 / "ids_train.npy").astype(str)
emb_te = np.load(E6 / "emb_test.npy")
ids_te = np.load(E6 / "ids_test.npy").astype(str)
lf = pd.read_csv(E6 / "label_folds.csv", dtype={"image_id": str})

pca = PCA(n_components=128, random_state=42)
z = pca.fit_transform(emb)
z = z / np.linalg.norm(z, axis=1, keepdims=True)
km = KMeans(n_clusters=64, n_init=10, random_state=42).fit(z)
d = pd.DataFrame({"image_id": ids, "cluster": km.labels_}).merge(lf[["image_id", "cluster"]], on="image_id", suffixes=("", "_l"))
print("sama persis:", float((d.cluster == d.cluster_l).mean()), "ARI:", adjusted_rand_score(d.cluster, d.cluster_l))

p = pd.read_csv(E5 / "prob_train.csv", dtype={"image_id": str}).set_index("image_id").loc[ids]
massa = p[NAMES].values * p[["cakupan"]].values
rows = []
for c in range(64):
    m = km.labels_ == c
    rata = massa[m].mean(0)
    u = rata.argsort()[::-1]
    rows.append({"cluster": c, "n": int(m.sum()), "nama": NAMES[u[0]], "porsi": round(float(rata[u[0]] / rata.sum()), 3),
                 "kedua": NAMES[u[1]], "porsi2": round(float(rata[u[1]] / rata.sum()), 3)})
ct = pd.DataFrame(rows).set_index("cluster")
ct.to_csv(OUT / "cluster_nama.csv")
ambang = np.where(ct["nama"] == "plastic", 0.6, 0.4)
ct["lolos"] = (ct["porsi"] >= ambang) & (ct["porsi"] - ct["porsi2"] >= 0.07)
print("cluster lolos", int(ct.lolos.sum()), "citra lolos", int(ct[ct.lolos].n.sum()))

zt = pca.transform(emb_te)
zt = zt / np.linalg.norm(zt, axis=1, keepdims=True)
a = pd.concat([pd.DataFrame({"split": "train", "image_id": ids, "cluster": km.labels_}),
               pd.DataFrame({"split": "test", "image_id": ids_te, "cluster": km.predict(zt)})], ignore_index=True)
for c in ("nama", "porsi", "kedua", "lolos"):
    a[c] = a["cluster"].map(ct[c])
a.to_csv(OUT / "assign.csv", index=False)
print(a.groupby("split").nama.value_counts().unstack(0).to_string())
print("test: 5 teratas cluster non-lolos")
print(a[(a.split == "test") & (~a.lolos)].groupby(["cluster", "nama"]).size().sort_values(ascending=False).head(12).to_string())
