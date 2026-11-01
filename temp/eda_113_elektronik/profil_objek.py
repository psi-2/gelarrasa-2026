"""Objek apa yang muncul pada kelompok metal dan electronic, per keluarga, train dan test.

Hanya nama kelas dan massa rata-rata. Tidak membuka citra. Test diukur untuk perbandingan, tidak ada fit.

Keluaran: output/profil_kelompok.txt

    python temp/eda_113_elektronik/profil_objek.py
"""
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
from timm.data import ImageNetInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
E8 = ROOT / "temp" / "exp_008_in22k_map" / "output"
EM = ROOT / "temp" / "eda_106_eomt" / "output"
nt = 1920
d = pd.read_csv(OUT / "dasar.csv", dtype={"image_id": str})
te = pd.read_csv(OUT / "dasar_test.csv", dtype={"image_id": str})

info = ImageNetInfo("imagenet-22k")
lem = [info.index_to_description(i).split(",")[0][:26] for i in range(info.num_classes())]
P = [(np.load(E8 / f"p22k_train_{m}.npy", mmap_mode="r"), np.load(E8 / f"p22k_test_{m}.npy", mmap_mode="r")) for m in ("convnext_l", "swinv2_l", "beitv2_l")]


def mean22k(tr_idx, te_idx):
    v = np.zeros(len(lem))
    n = 0
    for a, b in P:
        if len(tr_idx):
            v += np.asarray(a[tr_idx]).sum(0)
            n += len(tr_idx)
        if len(te_idx):
            v += np.asarray(b[te_idx]).sum(0)
            n += len(te_idx)
    return v / n


def top1_22k(tr_idx, te_idx):
    c = {}
    for a, b in P:
        if len(tr_idx):
            for i in np.asarray(a[tr_idx]).argmax(1):
                c[lem[i]] = c.get(lem[i], 0) + 1
        if len(te_idx):
            for i in np.asarray(b[te_idx]).argmax(1):
                c[lem[i]] = c.get(lem[i], 0) + 1
    return sorted(c.items(), key=lambda x: -x[1])


LATAR_C = {"wall-other-merged", "floor-other-merged", "table-merged", "floor-wood", "wall-wood", "wall-tile", "wall-stone", "wall-brick", "dirt-merged", "rock-merged", "dining table", "counter", "mirror-stuff",
           "ceiling-merged", "cabinet-merged", "shelf", "person", "pavement-merged", "grass-merged", "sand", "snow", "sky-other-merged", "water-other", "window-other", "door-stuff", "bed", "gravel", "road", "platform"}
LATAR_A = {"wall", "floor", "table", "ceiling", "earth", "person", "desk", "cabinet", "sidewalk", "road", "grass", "sand", "windowpane", "door", "countertop", "shelf", "water", "rock", "building", "sky"}


def muat(m, latar):
    kel = json.loads((EM / f"{m}_kelas.json").read_text(encoding="utf-8"))
    tr = .5 * np.load(EM / f"{m}_train.npy").sum(2)
    tes = .5 * np.load(EM / f"{m}_test.npy").sum(2)
    b = np.array([k in latar for k in kel])
    tr[:, b] = 0
    tes[:, b] = 0
    return kel, tr, tes


kc, Ct, Cq = muat("coco", LATAR_C)
ka, At, Aq = muat("ade", LATAR_A)


def atas(v, nama, n=9):
    u = np.argsort(v)[::-1][:n]
    return ", ".join(f"{nama[i]} {v[i]:.3f}" for i in u if v[i] >= .005)


def kelompok(judul, tr_idx, te_idx, f):
    tr_idx, te_idx = np.asarray(tr_idx, int), np.asarray(te_idx, int)
    n = len(tr_idx) + len(te_idx)
    if n == 0:
        return
    b = [f"\n### {judul} (n={n})"]
    b.append("  22k massa: " + atas(mean22k(tr_idx, te_idx), lem, 10))
    b.append("  22k top-1: " + ", ".join(f"{k} {v}" for k, v in top1_22k(tr_idx, te_idx)[:10]))
    vc = np.r_[Ct[tr_idx], Cq[te_idx]].mean(0)
    va = np.r_[At[tr_idx], Aq[te_idx]].mean(0)
    b.append("  COCO: " + atas(vc, kc))
    b.append("  ADE : " + atas(va, ka))
    t = "\n".join(b)
    print(t, flush=True)
    f.write(t + "\n")


with open(OUT / "profil_kelompok.txt", "w", encoding="utf-8") as f:
    kelompok("train electronic F_640", d.index[(d.label == "electronic") & (d.fam == "F_640")], [], f)
    for fam in ("F_640", "F_q74fb", "F_q75_4x3", "F_lain_web", "F_416"):
        kelompok(f"train metal {fam}", d.index[(d.label == "metal") & (d.fam == fam)], [], f)
    for fam in ("F_416", "F_lain_web"):
        kelompok(f"TEST prediksi metal {fam}", [], te.index[(te.pred == "metal") & (te.fam == fam)], f)
    kelompok("TEST prediksi electronic F_640", [], te.index[(te.pred == "electronic") & (te.fam == "F_640")], f)
    kelompok("train kosong F_416", d.index[d.label.isna() & (d.fam == "F_416")], [], f)
    kelompok("train kosong F_640", d.index[d.label.isna() & (d.fam == "F_640")], [], f)
