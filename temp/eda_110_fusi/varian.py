"""Uji sensitivitas aturan penamaan otomatis: tempering prior dan abstain kosakata. Train-only, tanpa temuan.

tau: skor dibagi prior^tau. tau=0 setara massa mentah EXP-006, tau=1 normalisasi penuh.
abstain: kosakata dengan prior kelas di bawah ambang tidak bersuara untuk kelas itu (kolom netral).
Ambang dan tau tidak dipilih dari temuan. Keluaran untuk dinilai koordinator.

    python temp/eda_110_fusi/varian.py
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
print("prior massa per kelas per kosakata (porsi kolom, train):")
tb = pd.DataFrame({k: v[:nt].mean(0) / v[:nt].mean(0).sum() for k, v in S.items()}, index=NAMES).round(3)
print(tb.to_string())

# kumpulan run tetap agar perbandingan adil
runs = []
for k in (48, 64, 96):
    for s in range(12):
        km = KMeans(k, n_init=1, random_state=1000 * k + s).fit(Ztr)
        runs.append((k, km.labels_, km.predict(Zte)))
print("run", len(runs), flush=True)


def bag(views, tau, abstain):
    V = np.zeros((N, 8))
    for k, lab, lab_te in runs:
        L = np.zeros((k, 8))
        for w in views:
            pri = S[w][:nt].mean(0) + 1e-6
            share = pri / pri.sum()
            v = S[w][:nt] / pri ** tau
            rata = np.stack([v[lab == c].mean(0) if (lab == c).any() else np.ones(8) for c in range(k)])
            rata = rata / rata.sum(1, keepdims=True)
            lr = np.log(rata + 1e-3)
            if abstain > 0:
                lr[:, share < abstain] = np.log(1 / 8)
            L += lr
        nm = L.argmax(1)
        a = np.r_[lab, lab_te]
        V[np.arange(N), nm[a]] += 1
    return V / len(runs)


rows = []
for nama, views, tau, ab in [("mentah_in1k", ["in1k"], 0, 0), ("t05_in1k", ["in1k"], .5, 0), ("t1_in1k", ["in1k"], 1, 0),
                             ("t05_semua", ["in1k", "coco", "ade"], .5, 0), ("t1_semua", ["in1k", "coco", "ade"], 1, 0),
                             ("t05_abst", ["in1k", "coco", "ade"], .5, .03), ("t1_abst", ["in1k", "coco", "ade"], 1, .03), ("t1_in1k_coco_abst", ["in1k", "coco"], 1, .03)]:
    V = bag(views, tau, ab)
    np.save(OUT / f"var_{nama}.npy", V.astype(np.float32))
    y = V.argmax(1)
    b = np.bincount(y[nt:], minlength=8)
    f416 = np.bincount(y[nt:][(d.fam[nt:] == "F_416").values], minlength=8)
    r = {"varian": nama, **dict(zip(NAMES, b.tolist())), "L1_ke_60": int(np.abs(b - 60).sum()), "train_plastic": int((y[:nt] == 4).sum()), "yakin90_tr": int((V.max(1)[:nt] >= .9).sum())}
    rows.append(r)
    print(nama, "F_416 test:", dict(zip(NAMES, f416.tolist())), flush=True)
t = pd.DataFrame(rows)
print(t.to_string(index=False))
t.to_csv(OUT / "varian.csv", index=False)
