"""Cluster embedding DINOv3 train, beri nama cluster dari massa kategori ConvNeXtV2 (EXP-005).

Test tidak ikut clustering.

    python temp/exp_006_cluster_dinov3/cluster.py                 # bandingkan K dan tampilkan cluster
    python temp/exp_006_cluster_dinov3/cluster.py --k 48 --simpan # tulis label dan hold-out
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
CONV = ROOT / "temp" / "exp_005_imagenet_map" / "output" / "prob_train.csv"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
SEED = 42


def muat():
    emb = np.load(OUT / "emb_train.npy")
    ids = np.load(OUT / "ids_train.npy").astype(str)
    p = pd.read_csv(CONV, dtype={"image_id": str}).set_index("image_id").loc[ids]
    massa = p[NAMES].values * p[["cakupan"]].values
    return emb, ids, p, massa


def fit_kmeans(z, k):
    return KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(z)


def tabel_cluster(km, massa, p, emb):
    rows = []
    for c in range(km.n_clusters):
        m = km.labels_ == c
        rata = massa[m].mean(0)
        urut = rata.argsort()[::-1]
        pusat = emb[m].mean(0)
        rows.append({"cluster": c, "n": int(m.sum()), "nama": NAMES[urut[0]],
                     "porsi": round(float(rata[urut[0]] / max(rata.sum(), 1e-9)), 3),
                     "kedua": NAMES[urut[1]], "porsi2": round(float(rata[urut[1]] / max(rata.sum(), 1e-9)), 3),
                     "cakupan": round(float(rata.sum()), 3),
                     "rapat": round(float((emb[m] @ (pusat / np.linalg.norm(pusat))).mean()), 3),
                     "top_imagenet": ", ".join(p["top1_imagenet"][m].value_counts().head(3).index)})
    return pd.DataFrame(rows).sort_values(["nama", "porsi"], ascending=[True, False])


def grup_mirip(emb, ambang):
    sim = emb @ emb.T
    np.fill_diagonal(sim, 0)
    b, k = np.where(sim >= ambang)
    n = len(emb)
    return connected_components(csr_matrix((np.ones(len(b)), (b, k)), shape=(n, n)), directed=False)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--min-porsi", type=float, default=0.4)
    ap.add_argument("--min-selisih", type=float, default=0.07)
    ap.add_argument("--min-porsi-plastic", type=float, default=0.6, help="lebih ketat, packet dan tray ambigu")
    ap.add_argument("--sim-grup", type=float, default=0.9)
    ap.add_argument("--simpan", action="store_true")
    args = ap.parse_args()

    emb, ids, p, massa = muat()
    z = PCA(n_components=128, random_state=SEED).fit_transform(emb)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)

    if args.k is None:
        print("K, silhouette, cluster porsi>=0.6, citra di cluster itu:")
        for k in (16, 24, 32, 40, 48, 64, 80):
            km = fit_kmeans(z, k)
            t = tabel_cluster(km, massa, p, emb)
            baik = t[t["porsi"] >= 0.6]
            print(k, round(float(silhouette_score(z, km.labels_, metric="cosine")), 4), len(baik), int(baik["n"].sum()))
        sim = emb @ emb.T
        np.fill_diagonal(sim, 0)
        for a in (0.8, 0.85, 0.9, 0.95):
            g = pd.Series(grup_mirip(emb, a)).value_counts()
            print(f"grup cos>={a}: {int(g[g > 1].sum())} citra, {int((g > 1).sum())} grup, terbesar {int(g.max())}")
        return

    km = fit_kmeans(z, args.k)
    t = tabel_cluster(km, massa, p, emb)
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(t.to_string(index=False))
    t.to_csv(OUT / f"cluster_k{args.k}.csv", index=False)

    peta = t.set_index("cluster")
    df = pd.DataFrame({"image_id": ids, "cluster": km.labels_})
    df["label"] = df["cluster"].map(peta["nama"])
    df["porsi"] = df["cluster"].map(peta["porsi"])
    df["selisih"] = df["porsi"] - df["cluster"].map(peta["porsi2"])
    df["group"] = grup_mirip(emb, args.sim_grup)
    ambang = np.where(df["label"] == "plastic", args.min_porsi_plastic, args.min_porsi)
    lolos = (df["porsi"] >= ambang) & (df["selisih"] >= args.min_selisih)
    pakai = df[lolos].sort_values("image_id").reset_index(drop=True)
    print(f"\ndipakai {len(pakai)} dari {len(df)} citra, {pakai['cluster'].nunique()} dari {args.k} cluster")
    print(pakai["label"].value_counts().reindex(NAMES).fillna(0).astype(int).to_string())
    if args.simpan:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        _, va = next(sgkf.split(pakai, pakai["label"], pakai["group"]))
        pakai["fold"] = 1
        pakai.loc[va, "fold"] = 0
        assert not (pakai.groupby("group")["fold"].nunique() > 1).any()
        print("\nhold-out per kelas:")
        print(pd.crosstab(pakai["label"], pakai["fold"]).reindex(NAMES).fillna(0).astype(int).to_string())
        pakai.to_csv(OUT / "label_folds.csv", index=False)


if __name__ == "__main__":
    main()
