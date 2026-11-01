"""Pemeriksaan sebelum aturan dibekukan. Tidak mengubah label. Test hanya dihitung.

    python temp/eda_113_elektronik/pra_beku.py
"""
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from timm.data import ImageNetInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
RC = ROOT / "temp" / "research_uji_crossfit" / "output"
E8 = ROOT / "temp" / "exp_008_in22k_map" / "output"
E12 = ROOT / "temp" / "eda_112_label_v2" / "output"
pd.set_option("display.width", 230, "display.max_columns", 40, "display.max_rows", 200)
d = pd.read_csv(OUT / "dasar.csv", dtype={"image_id": str})
x = pd.read_csv(OUT / "massa_listrik.csv", dtype={"image_id": str})
tr = x[x.split == "train"].reset_index(drop=True)
te = x[x.split == "test"].reset_index(drop=True)

print("== dua baris v4 tanpa group ==")
b = d[d.label.notna() & d.group.isna()]
for r in b.itertuples():
    sefold = d[(d.grp == r.grp) & d.fold.notna() & (d.image_id != r.image_id)]
    print("label", r.label, "fold", r.fold, "fam asli", r.fam, "group hitung ulang", r.grp, "anggota lain:", len(sefold), "fold anggota:", sefold.fold.value_counts().to_dict())

print("\n== massa sel sedang (0,02 sampai 0,2 rata-rata) ==")
s = tr[(tr.sel >= .02) & (tr.sel_min < .2)]
print("train:", s.groupby(["fam", "kelas"]).size().to_dict(), "| massa", sorted(s.sel.round(3).tolist(), reverse=True)[:12])
q = te[(te.sel >= .02) & (te.sel_min < .2)]
print("test:", q.groupby(["fam", "kelas"]).size().to_dict(), "| massa", sorted(q.sel.round(3).tolist(), reverse=True))

Ev = normalize(np.load(E6 / "emb_train.npy"))
Ec = normalize(np.load(RC / "emb_cnn_train.npy"))
hit = (tr.sel_min >= .2).values
print("\n== tetangga embedding jangkar baterai (train) ==", int(hit.sum()))
sv = (Ev @ Ev[hit].T)
sc = (Ec @ Ec[hit].T)
pv = np.sort(sv, 1)[:, -3:].mean(1)   # rata 3 jangkar terdekat, jangkar sendiri ikut bagi anggota
pc = np.sort(sc, 1)[:, -3:].mean(1)
t = tr.assign(cv=pv.round(3), cc=pc.round(3))
print("cos jangkar antar sesama (tanpa diri):", np.round(np.sort(sv[hit] - 2 * np.eye(hit.sum()), 1)[:, -1], 3).tolist())
s = t[~hit].sort_values("cv", ascending=False).head(16)
print(s[["fam", "kelas", "sel", "cv", "cc"]].to_string(index=False))
print("kelas teratas pada 30 tetangga non-jangkar:", t[~hit].sort_values("cv", ascending=False).head(30).kelas.value_counts().to_dict())

print("\n== sengketa metal-electronic tahap 2: lemma 22k ==")
lem = [ImageNetInfo("imagenet-22k").index_to_description(i).split(",")[0][:24] for i in range(21841)]
P = [np.load(E8 / f"p22k_train_{m}.npy", mmap_mode="r") for m in ("convnext_l", "swinv2_l", "beitv2_l")]
r2 = pd.read_csv(E12 / "rincian_v2.csv", dtype={"image_id": str})
for a, b_ in (("electronic", "metal"), ("metal", "electronic")):
    m = ((d.label == a) & ((r2.b_in22k == b_) | (r2.c_in22k == b_))).values
    i = np.where(m)[0]
    c = {}
    for p in P:
        for t_ in np.asarray(p[i]).argmax(1):
            c[lem[t_]] = c.get(lem[t_], 0) + 1
    print(a, "dibantah in22k:", len(i), "| keluarga", d.fam[m].value_counts().to_dict(), "| baterai", int((tr.sel_min[m] >= .2).sum()))
    print("  ", sorted(c.items(), key=lambda z: -z[1])[:14])
