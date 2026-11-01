"""Gabungkan ConvNeXtV2 (ImageNet-1k) dengan kosakata EoMT, lalu nama cluster dan propagasi kNN.

Tanpa temuan. Konsistensi diuji dengan jumlah per kelas pada test dan komposisi per keluarga sumber.

Keluaran: output/skor_{sumber}.npy, output/cluster_fusi.csv, output/kandidat_*.csv

    python temp/eda_110_fusi/fusi.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kosakata import ADE, COCO, NAMES

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
E5 = ROOT / "temp" / "exp_005_imagenet_map" / "output"
EM = ROOT / "temp" / "eda_106_eomt" / "output"
pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 300, "display.max_colwidth", 90)

g = pd.read_csv(ROOT / "temp" / "eda_102_sumber" / "output" / "meta_assign.csv", dtype={"image_id": str})
tr = g[g.split == "train"].sort_values("image_id")
te = g[g.split == "test"].sort_values("image_id")
d = pd.concat([tr, te], ignore_index=True)
nt = len(tr)
E = np.vstack([np.load(E6 / "emb_train.npy"), np.load(E6 / "emb_test.npy")]).astype(np.float32)


def skor_eomt(m, peta):
    kel = json.loads((EM / f"{m}_kelas.json").read_text(encoding="utf-8"))
    A = np.vstack([np.load(EM / f"{m}_train.npy"), np.load(EM / f"{m}_test.npy")])
    C = 0.5 * A[:, :, 0] + 0.5 * A[:, :, 1]
    M = np.zeros((len(NAMES), len(kel)), dtype=np.float32)
    for k, v in peta.items():
        i = kel.index(k)
        for c, w in v.items():
            M[NAMES.index(c), i] = w
    return C @ M.T


def skor_conv():
    p = pd.concat([pd.read_csv(E5 / f"prob_{s}.csv", dtype={"image_id": str}).assign(split=s) for s in ("train", "test")])
    p = d[["split", "image_id"]].merge(p, on=["split", "image_id"])
    return p[NAMES].values * p[["cakupan"]].values


S = {"conv": skor_conv(), "coco": skor_eomt("coco", COCO)}
if (EM / "ade_train.npy").exists():
    S["ade"] = skor_eomt("ade", ADE)
for k, v in S.items():
    np.save(OUT / f"skor_{k}.npy", v)
print("sumber skor:", list(S), {k: round(float(v.sum(1).mean()), 3) for k, v in S.items()})
print("prediksi per sumber (test):")
for k, v in S.items():
    print(k, dict(zip(NAMES, np.bincount(v[nt:].argmax(1), minlength=8).tolist())))


def per_kluster(lab, bobot=None):
    """Log-odds per cluster dari rata-rata skor tiap sumber. Sumber dinormalkan prior per kolom agar bias kosakata hilang."""
    uni = np.unique(lab)
    L = np.zeros((len(uni), 8))
    for k, v in S.items():
        prior = v.mean(0) + 1e-6
        rata = np.stack([v[lab == c].mean(0) for c in uni])
        rata = rata / prior
        rata = rata / rata.sum(1, keepdims=True)
        L += (bobot or {}).get(k, 1.0) * np.log(rata + 1e-3)
    return uni, L


# 1. nama cluster K=64 dari fusi
uni, L = per_kluster(d.cluster.values)
Ps = np.exp(L - L.max(1, keepdims=True))
Ps = Ps / Ps.sum(1, keepdims=True)
tb = pd.DataFrame({"cluster": uni, "lama": d.groupby("cluster").nama.first().reindex(uni).values, "fusi": np.array(NAMES)[L.argmax(1)],
                   "yakin": Ps.max(1).round(3), "kedua": np.array(NAMES)[np.argsort(L, 1)[:, -2]],
                   "n": d.groupby("cluster").size().reindex(uni).values, "te": d[d.split == "test"].groupby("cluster").size().reindex(uni).fillna(0).astype(int).values,
                   "fam": d.groupby("cluster").fam.agg(lambda s: s.value_counts().index[0]).reindex(uni).values})
print("\ncluster yang berubah nama:")
print(tb[tb.lama != tb.fusi].sort_values("te", ascending=False).to_string(index=False))
tb.to_csv(OUT / "cluster_fusi.csv", index=False)
d["fusi64"] = d.cluster.map(tb.set_index("cluster").fusi)

print("\njumlah per kelas, K=64 fusi")
print(pd.crosstab(d.fusi64, d.split).reindex(NAMES).to_string())
print("\ntest per keluarga, K=64 fusi")
print(pd.crosstab(d[d.split == "test"].fam, d[d.split == "test"].fusi64).reindex(columns=NAMES).fillna(0).astype(int).to_string())
print("\ntrain per keluarga, K=64 fusi")
print(pd.crosstab(d[d.split == "train"].fam, d[d.split == "train"].fusi64).reindex(columns=NAMES).fillna(0).astype(int).to_string())

# 2. skor gabungan per citra dengan kNN DINOv3
Sim = E @ E.T
np.fill_diagonal(Sim, -1)
K = 15
idx = np.argpartition(-Sim, K, axis=1)[:, :K]
gab = np.zeros((len(E), 8))
for k, v in S.items():
    prior = v.mean(0) + 1e-6
    r = (v / prior)
    smooth = 0.5 * r + 0.5 * r[idx].mean(1)
    smooth = smooth / (smooth.sum(1, keepdims=True) + 1e-9)
    gab += np.log(smooth + 1e-3)
pred = np.array(NAMES)[gab.argmax(1)]
d["fusi_knn"] = pred
print("\nkNN fusi per kelas test:", d[d.split == "test"].fusi_knn.value_counts().reindex(NAMES).to_dict())
print(pd.crosstab(d[d.split == "test"].fam, d[d.split == "test"].fusi_knn).reindex(columns=NAMES).fillna(0).astype(int).to_string())
print("setuju fusi64 vs knn test", round(float((d.fusi64 == d.fusi_knn)[d.split == "test"].mean()), 4))
np.save(OUT / "gab.npy", gab)
d[["split", "image_id", "fam", "cluster", "nama", "fusi64", "fusi_knn"]].to_csv(OUT / "kandidat_semua.csv", index=False)
