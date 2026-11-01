"""EM semi-terawasi: emisi kosakata per kelas dengan inisialisasi peta a priori.

Tiap citra memiliki K kata (top kelas ConvNeXtV2 ImageNet-1k, area EoMT COCO, area EoMT ADE) dihaluskan lewat kNN DINOv3.
E-step: posterior kelas. M-step: emisi p(kata | kelas). Prior kelas bisa global atau per (keluarga, split).
Peta a priori hanya inisialisasi dan regularisasi Dirichlet, jadi emisi `packet` atau `suitcase` dipelajari dari co-occurrence.

Tanpa temuan.

    python temp/eda_110_fusi/em.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kosakata import NAMES

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
E5 = ROOT / "temp" / "exp_005_imagenet_map" / "output"
EM_D = ROOT / "temp" / "eda_106_eomt" / "output"
pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 300, "display.max_colwidth", 90)

g = pd.read_csv(ROOT / "temp" / "eda_102_sumber" / "output" / "meta_assign.csv", dtype={"image_id": str})
tr = g[g.split == "train"].sort_values("image_id")
te = g[g.split == "test"].sort_values("image_id")
d = pd.concat([tr, te], ignore_index=True)
nt, N = len(tr), len(d)
E = np.vstack([np.load(E6 / "emb_train.npy"), np.load(E6 / "emb_test.npy")]).astype(np.float32)
Sim = E @ E.T
np.fill_diagonal(Sim, -1)
K = 10
idx = np.argpartition(-Sim, K, axis=1)[:, :K]
wk = np.clip(np.take_along_axis(Sim, idx, 1), 0, None) ** 4
wk = wk / wk.sum(1, keepdims=True)


def halus(X, alpha=.5):
    return alpha * X + (1 - alpha) * (X[idx] * wk[..., None]).sum(1)


# ImageNet-1k penuh
P1 = np.vstack([np.load(E5 / "imagenet_train.npy"), np.load(E5 / "imagenet_test.npy")])
LATAR_C = ["wall-other-merged", "floor-other-merged", "table-merged", "floor-wood", "wall-wood", "wall-tile", "wall-stone", "wall-brick", "dirt-merged", "rock-merged", "dining table",
           "counter", "mirror-stuff", "ceiling-merged", "cabinet-merged", "shelf", "person", "pavement-merged", "grass-merged", "sand", "snow", "sky-other-merged", "water-other",
           "window-other", "door-stuff", "bed", "gravel", "road", "platform"]
LATAR_A = ["wall", "floor", "table", "ceiling", "earth", "person", "desk", "cabinet", "sidewalk", "road", "grass", "sand", "windowpane", "door", "countertop", "shelf", "water", "rock", "building", "sky"]


def muat(m, latar):
    kel = json.loads((EM_D / f"{m}_kelas.json").read_text(encoding="utf-8"))
    A = np.vstack([np.load(EM_D / f"{m}_train.npy"), np.load(EM_D / f"{m}_test.npy")])
    C = .5 * A[:, :, 0] + .5 * A[:, :, 1]
    C[:, [kel.index(k) for k in latar]] = 0
    return C


X = {"in1k": P1, "coco": muat("coco", LATAR_C)}
if (EM_D / "ade_train.npy").exists():
    X["ade"] = muat("ade", LATAR_A)
for k in X:
    X[k] = halus(X[k])
    X[k] = X[k] / (X[k].sum(1, keepdims=True) + 1e-9)

gab0 = np.load(OUT / "gab.npy")
R0 = np.exp(gab0 - gab0.max(1, keepdims=True))
R0 = R0 / R0.sum(1, keepdims=True)

fam = (d.fam + "|" + d.split).values
F = sorted(set(fam))
fi = np.array([F.index(x) for x in fam])


def latih(R_init, per_sumber, iters=30, kata_bobot=None, kekang=1.0):
    R = R_init.copy()
    A_prior = {k: None for k in X}
    for it in range(iters):
        # M-step
        Em = {}
        for k, v in X.items():
            c = R.T @ v  # 8 x V
            if A_prior[k] is None:
                A_prior[k] = c / (c.sum(0, keepdims=True) + 1e-9) * kekang  # sekali, dari inisialisasi
            e = c + A_prior[k]
            Em[k] = e / e.sum(1, keepdims=True)
        if per_sumber:
            pi = np.zeros((len(F), 8))
            np.add.at(pi, fi, R)
            pi = pi + 2
            pi = pi / pi.sum(1, keepdims=True)
            lp = np.log(pi[fi])
        else:
            p = R.sum(0) + 1
            lp = np.tile(np.log(p / p.sum()), (N, 1))
        # E-step
        L = lp.copy()
        for k, v in X.items():
            w = (kata_bobot or {}).get(k, 1.0)
            L = L + w * 3.0 * (v @ np.log(Em[k].T + 1e-12))
        Rn = np.exp(L - L.max(1, keepdims=True))
        Rn = Rn / Rn.sum(1, keepdims=True)
        delta = float(np.abs(Rn - R).sum(1).mean() / 2)
        R = Rn
        if delta < 1e-3:
            break
    return R, Em, (pi if per_sumber else None), it


def lapor(R, nama):
    p = R.argmax(1)
    y = np.array(NAMES)[p]
    print(f"\n== {nama}: test {dict(zip(NAMES, np.bincount(p[nt:], minlength=8).tolist()))}")
    print(f"   train {dict(zip(NAMES, np.bincount(p[:nt], minlength=8).tolist()))}")
    print(pd.crosstab(d.fam[nt:].values, y[nt:]).reindex(columns=NAMES).fillna(0).astype(int).to_string())
    return y


vocab_in = (ROOT / "temp" / "exp_005_imagenet_map" / "output" / "label_model.txt")
hasil = {}
for nama, ps in (("global", False), ("per_sumber", True)):
    R, Em, pi, it = latih(R0, ps)
    print("iterasi", it)
    y = lapor(R, nama)
    hasil[nama] = y
    d[f"em_{nama}"] = y
    np.save(OUT / f"R_{nama}.npy", R)
    if pi is not None:
        print("prior per (keluarga, split):")
        print(pd.DataFrame(pi, index=F, columns=NAMES).round(2).to_string())
    print("setuju dengan init (test):", round(float((np.array(NAMES)[R0.argmax(1)] == y)[nt:].mean()), 4), "| yakin >=.9 test:", round(float((R.max(1)[nt:] >= .9).mean()), 3))
    kel = json.loads((EM_D / "coco_kelas.json").read_text(encoding="utf-8"))
    print("emisi COCO utama per kelas:")
    for i, c in enumerate(NAMES):
        u = Em["coco"][i].argsort()[::-1][:7]
        print(f"  {c:11s}", ", ".join(f"{kel[j]} {Em['coco'][i, j]:.2f}" for j in u))

print("\nsetuju global vs per_sumber (test)", round(float((hasil["global"] == hasil["per_sumber"])[nt:].mean()), 4))
d.to_csv(OUT / "kandidat_em.csv", index=False)
