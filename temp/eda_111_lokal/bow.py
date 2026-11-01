"""Hipotesis D: bandingkan CLS vs bag of visual words patch pada pasangan sulit, dengan target yang tidak berasal dari DINOv3.

Target per citra train: skor kosakata mentah per citra dinormalkan prior, tanpa kNN dan tanpa cluster.
Citra dipakai bila semua kosakata yang bersuara untuk kedua kelas (porsi >= 0,03) sepakat dan marginnya cukup.
Kosakata visual di-fit dari train saja. Tanpa nomor cluster dan tanpa temuan.

    python temp/eda_111_lokal/bow.py
"""
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
F = ROOT / "temp" / "eda_110_fusi" / "output"
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
nt = 1920
T = np.load(OUT / "tokens.npy", mmap_mode="r")
rng = np.random.default_rng(42)
ambil = np.sort(rng.choice(nt, 700, replace=False))
sampel = np.asarray(T[ambil][:, ::5]).reshape(-1, 64).astype(np.float32)
print("sampel", sampel.shape, flush=True)
km = MiniBatchKMeans(256, batch_size=4096, n_init=3, random_state=42).fit(sampel)
C = km.cluster_centers_.astype(np.float32)
hist = np.zeros((nt, 256), dtype=np.float32)
for i in range(0, nt, 64):
    x = np.asarray(T[i:i + 64]).astype(np.float32)
    n, m, _ = x.shape
    for j in range(n):
        dd = ((x[j][:, None, :] - C[None]) ** 2).sum(-1)
        hist[i + j] = np.bincount(dd.argmin(1), minlength=256)
tf = hist / hist.sum(1, keepdims=True)
idf = np.log(nt / (1 + (hist > 0).sum(0)))
B = normalize(np.sqrt(tf) * idf)
np.save(OUT / "bow_train.npy", B)
CLS = np.load(E6 / "emb_train.npy")

S = {k: np.load(F / f"skor_{k}.npy")[:nt] for k in ("conv", "coco", "ade")}
grp = pd.read_csv(F / "label_folds_t1_in1k.csv", dtype={"image_id": str})
ids = np.load(E6 / "ids_train.npy").astype(str)
gm = pd.Series(np.arange(nt), index=ids)
g_all = -np.arange(1, nt + 1)
g_all[gm.loc[grp.image_id].values] = grp.group.values

rows = []
for a, b in [("plastic", "glass"), ("plastic", "paper"), ("glass", "paper"), ("metal", "plastic"), ("metal", "glass"), ("metal", "paper"), ("paper", "cardboard")]:
    ia, ib = NAMES.index(a), NAMES.index(b)
    pil = np.ones(nt, bool)
    arah = None
    nview = 0
    for k, v in S.items():
        pri = v.mean(0) + 1e-6
        sh = pri / pri.sum()
        if sh[ia] < .03 or sh[ib] < .03:
            continue
        r = v / pri
        m = np.log((r[:, ia] + .05) / (r[:, ib] + .05))
        arah = m if arah is None else arah + m
        nview += 1
        if k == "conv":
            s0 = np.sign(m)
        else:
            pil &= np.sign(m) == s0
    pos = pil & (arah > np.log(2) * nview)
    neg = pil & (arah < -np.log(2) * nview)
    idx = np.where(pos | neg)[0]
    y = pos[idx].astype(int)
    if min(y.sum(), (1 - y).sum()) < 25:
        print(a, b, "terlalu sedikit", int(y.sum()), int((1 - y).sum()))
        continue
    sg = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    res = {"pasangan": f"{a}-{b}", "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "n_kosakata": nview}
    for nama, X in (("cls", CLS), ("bow", B), ("gab", np.hstack([normalize(CLS), B]))):
        oof = np.zeros(len(idx))
        for tr_, va_ in sg.split(idx, y, g_all[idx]):
            oof[va_] = LogisticRegression(C=1, max_iter=2000, class_weight="balanced").fit(X[idx[tr_]], y[tr_]).predict_proba(X[idx[va_]])[:, 1]
        res[nama] = round(float(roc_auc_score(y, oof)), 4)
    rows.append(res)
    print(res, flush=True)
t = pd.DataFrame(rows)
t.to_csv(OUT / "cls_vs_bow.csv", index=False)
print(t.to_string(index=False))
