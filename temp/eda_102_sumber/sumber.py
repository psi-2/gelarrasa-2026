"""Hipotesis A lanjutan: bentuk sumber dari sidik jari berkas, lalu uji kaitannya dengan cluster DINOv3.

Keluaran: output/assign.csv (cluster K64 semua citra, test dari pusat terdekat, tanpa ikut fit),
output/sumber.csv, output/sumber_vs_nama.csv.

    python temp/eda_102_sumber/sumber.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
E5 = ROOT / "temp" / "exp_005_imagenet_map" / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
pd.set_option("display.width", 240, "display.max_columns", 50, "display.max_rows", 300, "display.max_colwidth", 60)

meta = pd.read_csv(ROOT / "temp" / "eda_101_metadata" / "output" / "meta.csv", dtype={"image_id": str})
emb = {s: np.load(E6 / f"emb_{s}.npy") for s in ("train", "test")}
ids = {s: np.load(E6 / f"ids_{s}.npy").astype(str) for s in ("train", "test")}
ct = pd.read_csv(E6 / "cluster_k64.csv").set_index("cluster")

pca = PCA(n_components=128, random_state=42).fit(emb["train"])
def z(x):
    v = pca.transform(x)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

km = KMeans(n_clusters=64, n_init=10, random_state=42).fit(z(emb["train"]))
as_tr = pd.DataFrame({"split": "train", "image_id": ids["train"], "cluster": km.labels_})
as_te = pd.DataFrame({"split": "test", "image_id": ids["test"], "cluster": km.predict(z(emb["test"]))})
asg = pd.concat([as_tr, as_te], ignore_index=True)
lf = pd.read_csv(E6 / "label_folds.csv", dtype={"image_id": str})
cek = as_tr.merge(lf, on="image_id", suffixes=("", "_lama"))
print("cek cluster sama dengan label_folds:", float((cek.cluster == cek.cluster_lama).mean()))
asg["nama"] = asg["cluster"].map(ct["nama"])
asg["porsi"] = asg["cluster"].map(ct["porsi"])
asg.to_csv(OUT / "assign.csv", index=False)

df = meta.merge(asg, on=["split", "image_id"])
df["wh"] = df.w.astype(str) + "x" + df.h.astype(str)


def bucket(r):
    wh = r["wh"]
    if wh in ("640x640", "416x416", "640x480", "400x533", "1599x1599", "320x320", "225x225", "512x384", "400x534",
              "275x183", "1280x1280", "800x600", "533x400", "400x711", "800x533"):
        return wh
    return "lain"


df["wh_b"] = df.apply(bucket, axis=1)
df["q_b"] = df.qhash.where(df.qhash.map(df.qhash.value_counts()) >= 6, "qlain").fillna("noq")
df["sumber"] = df.wh_b + "|" + df.q_b + "|" + df.subsamp.fillna("-") + "|" + df.n_dht.fillna(0).astype(int).astype(str)
t = df.groupby("sumber").agg(train=("split", lambda s: (s == "train").sum()), test=("split", lambda s: (s == "test").sum()),
                              bpp=("bpp", "median"), kb=("size", lambda s: s.median() / 1024)).sort_values("train", ascending=False)
print(t.head(45).round(2).to_string())
t.to_csv(OUT / "sumber.csv")

print("\n== ukuran vs nama cluster (kolom: nama, baris: ukuran) ==")
x = pd.crosstab(df.wh_b, df.nama).reindex(columns=NAMES).fillna(0).astype(int)
x["n"] = x.sum(1)
print(x.sort_values("n", ascending=False).to_string())
x.to_csv(OUT / "sumber_vs_nama.csv")

print("\n== qhash utama vs nama ==")
x = pd.crosstab(df.q_b, df.nama).reindex(columns=NAMES).fillna(0).astype(int)
x["n"] = x.sum(1)
print(x.sort_values("n", ascending=False).to_string())

print("\n== sumber vs nama ==")
x = pd.crosstab(df.sumber, df.nama).reindex(columns=NAMES).fillna(0).astype(int)
x["n"] = x.sum(1)
print(x[x.n >= 8].sort_values("n", ascending=False).to_string())

for a, b in [("wh_b", "cluster"), ("q_b", "cluster"), ("sumber", "cluster"), ("wh_b", "nama"), ("sumber", "nama")]:
    c = pd.crosstab(df[a], df[b])
    chi, pv, dof, _ = chi2_contingency(c)
    print(f"{a} vs {b}: AMI {adjusted_mutual_info_score(df[a], df[b]):.3f} NMI {normalized_mutual_info_score(df[a], df[b]):.3f} chi2 p={pv:.2e}")

print("\n== cluster dengan sumber dominan ==")
g = df.groupby("cluster")
rows = []
for c, d in g:
    v = d.wh_b.value_counts(normalize=True)
    q = d.q_b.value_counts(normalize=True)
    rows.append({"cluster": c, "nama": ct.loc[c, "nama"], "porsi": ct.loc[c, "porsi"], "n": len(d), "test": (d.split == "test").sum(),
                 "wh_top": v.index[0], "wh_p": round(v.iloc[0], 2), "wh2": v.index[1] if len(v) > 1 else "", "p2": round(v.iloc[1], 2) if len(v) > 1 else 0,
                 "q_top": q.index[0], "q_p": round(q.iloc[0], 2)})
print(pd.DataFrame(rows).sort_values("wh_p", ascending=False).to_string(index=False))

print("\n== train vs test per cluster, proporsi ==")
c = pd.crosstab(df.cluster, df.split)
c["ratio_test"] = (c.test / c.sum(1)).round(3)
c["nama"] = ct["nama"]
print(c.sort_values("ratio_test").to_string())
