"""Langkah 3: skor per sumber, normalisasi prior tau=1 train-only, nama cluster otomatis lewat 36 run KMeans.

Sumber: in1k (ConvNeXtV2, t1 lama), in22k (rata-rata 3 model), dms (material). Gabungan dihitung pada log-odds.
Kolom yang tidak dipetakan sebuah sumber dianggap abstain (netral). Test tidak ikut fit.
Peta dimuat dari berkas beku dan hash diverifikasi.

Keluaran: output/skor_{sumber}.npy, output/vote_{varian}.npy (2400 x 8)

    python temp/eda_112_label_v2/sumber_v2.py
"""
import hashlib
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
E8 = ROOT / "temp" / "exp_008_in22k_map" / "output"
F = ROOT / "temp" / "eda_110_fusi" / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
nt, N = 1920, 2400

M22, Md = np.load(OUT / "peta_in22k.npy"), np.load(OUT / "peta_dms.npy")
rek = json.loads((OUT / "peta_beku.json").read_text(encoding="utf-8"))
assert hashlib.sha256(M22.tobytes()).hexdigest() == rek["sha256_peta_in22k"], "peta 22k berubah"
assert hashlib.sha256(Md.tobytes()).hexdigest() == rek["sha256_peta_dms"], "peta dms berubah"
print("hash peta cocok dengan pembekuan", rek["waktu"])

P = {m: np.vstack([np.load(E8 / f"p22k_train_{m}.npy"), np.load(E8 / f"p22k_test_{m}.npy")]) for m in ("convnext_l", "swinv2_l", "beitv2_l")}
t = {m: v.argmax(1) for m, v in P.items()}
print("selaras indeks, top-1 sama:", {f"{a}-{b}": round(float((t[a] == t[b]).mean()), 3) for a, b in [("convnext_l", "swinv2_l"), ("convnext_l", "beitv2_l"), ("swinv2_l", "beitv2_l")]})
print("rata cakupan massa terpetakan:", {m: round(float((v @ (M22.sum(0) > 0).astype(np.float32)).mean()), 3) for m, v in P.items()})
per_model = {m: v @ M22.T for m, v in P.items()}
in22k = np.mean(list(per_model.values()), axis=0)
D = .5 * np.vstack([np.load(E8 / "dms_train.npy"), np.load(E8 / "dms_test.npy")]) + .5 * np.vstack([np.load(E8 / "dms_tengah_train.npy"), np.load(E8 / "dms_tengah_test.npy")])
dms = D @ Md.T
S = {"in1k": np.load(F / "skor_conv.npy"), "in22k": in22k, "dms": dms}
for k, v in S.items():
    np.save(OUT / f"skor_{k}.npy", v.astype(np.float32))
for m, v in per_model.items():
    np.save(OUT / f"skor_22k_{m}.npy", v.astype(np.float32))
print("porsi massa train per sumber:")
print(pd.DataFrame({k: v[:nt].mean(0) / v[:nt].mean(0).sum() for k, v in S.items()}, index=NAMES).round(3).to_string())


def relatif(v):
    """Skor per citra dinormalkan prior train, kolom tanpa massa jadi nol."""
    pri = v[:nt].mean(0)
    r = np.divide(v, pri, out=np.zeros_like(v, dtype=np.float64), where=pri > 0)
    return r, pri > 0


# argmax per citra tanpa cluster
for k, v in S.items():
    r, _ = relatif(v)
    np.save(OUT / f"citra_{k}.npy", r.argmax(1))
    print(f"argmax citra {k}, train:", dict(zip(NAMES, np.bincount(r[:nt].argmax(1), minlength=8).tolist())))

E_tr, E_te = np.load(E6 / "emb_train.npy"), np.load(E6 / "emb_test.npy")
pca = PCA(128, random_state=42).fit(E_tr)


def z(x):
    v = pca.transform(x)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


Ztr, Zte = z(E_tr), z(E_te)
runs = []
for k in (48, 64, 96):
    for s in range(12):
        km = KMeans(k, n_init=1, random_state=1000 * k + s).fit(Ztr)
        runs.append((k, km.labels_, km.predict(Zte)))
print("run", len(runs), flush=True)


def bag(views):
    V = np.zeros((N, 8))
    for k, lab, lab_te in runs:
        L = np.zeros((k, 8))
        for w in views:
            r, ada = relatif(S[w])
            v = r[:nt]
            rata = np.stack([v[lab == c].mean(0) if (lab == c).any() else np.ones(8) for c in range(k)])
            rata = rata / np.maximum(rata.sum(1, keepdims=True), 1e-9)
            lr = np.log(rata + 1e-3)
            lr[:, ~ada] = np.log(1 / 8)
            L += lr
        V[np.arange(N), L.argmax(1)[np.r_[lab, lab_te]]] += 1
    return V / len(runs)


for nama, views in (("in1k", ["in1k"]), ("in22k", ["in22k"]), ("dms", ["dms"]), ("in22k_dms", ["in22k", "dms"]), ("semua", ["in1k", "in22k", "dms"])):
    V = bag(views)
    np.save(OUT / f"vote_{nama}.npy", V.astype(np.float32))
    y = V.argmax(1)
    print(f"{nama:10s} train {dict(zip(NAMES, np.bincount(y[:nt], minlength=8).tolist()))} suara>=.75 {(V.max(1)[:nt] >= .75).sum()}", flush=True)
lama = np.load(F / "var_t1_in1k.npy")
baru = np.load(OUT / "vote_in1k.npy")
print("reproduksi t1_in1k: selisih maks", float(np.abs(lama - baru).max()), "kesepakatan label", float((lama.argmax(1) == baru.argmax(1)).mean()))
