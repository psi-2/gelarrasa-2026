"""Ukur massa grup kosakata per citra dan material DMS. Tidak mengubah label. Test hanya dihitung.

Keluaran: output/massa.csv (train dan test)

    python temp/eda_114_paper/ukur.py
"""
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd

from kosakata_material import GRUP, bangun

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
E8 = ROOT / "temp" / "exp_008_in22k_map" / "output"
EM = ROOT / "temp" / "eda_106_eomt" / "output"
pd.set_option("display.width", 230, "display.max_columns", 40, "display.max_rows", 200)
d = pd.read_csv(OUT / "dasar.csv", dtype={"image_id": str})
te = pd.read_csv(OUT / "dasar_test.csv", dtype={"image_id": str})
M, _ = bangun()
tr = pd.DataFrame({"split": "train", "image_id": d.image_id, "fam": d.fam, "kelas": d.label.fillna("(kosong)")})
ts = pd.DataFrame({"split": "test", "image_id": te.image_id, "fam": te.fam, "kelas": te.pred})
for fname, df in (("train", tr), ("test", ts)):
    P = [np.load(E8 / f"p22k_{fname}_{m}.npy", mmap_mode="r") for m in ("convnext_l", "swinv2_l", "beitv2_l")]
    hasil = {k: [] for k in GRUP}
    for a in range(0, len(df), 256):
        blok = [np.asarray(p[a:a + 256]) for p in P]
        for k, v in M.items():
            hasil[k].append(np.median(np.stack([b[:, v].sum(1) for b in blok], 1), 1))
        del blok
    for k in GRUP:
        df["v_" + k] = np.concatenate(hasil[k]).round(4)
dm = json.loads((E8 / "dms_kelas.json").read_text(encoding="utf-8"))
Dtr = .5 * np.load(E8 / "dms_train.npy") + .5 * np.load(E8 / "dms_tengah_train.npy")
Dte = .5 * np.load(E8 / "dms_test.npy") + .5 * np.load(E8 / "dms_tengah_test.npy")
MAT = {"Paper": ["Paper"], "Cardboard": ["Cardboard"], "Metal": ["Metal"], "Glass": ["Glass"], "Plastic": ["Plastic, clear", "Plastic, non-clear"], "PlasticJernih": ["Plastic, clear"], "Fabric": ["Fabric/cloth"]}
for k, v in MAT.items():
    ix = [dm.index(x) for x in v]
    tr["d_" + k], ts["d_" + k] = Dtr[:, ix].sum(1).round(4), Dte[:, ix].sum(1).round(4)
kc = json.loads((EM / "coco_kelas.json").read_text(encoding="utf-8"))
Ctr, Cte = np.load(EM / "coco_train.npy").mean(2), np.load(EM / "coco_test.npy").mean(2)
for k, v in {"paper": ["paper-merged", "book"], "cardboard": ["cardboard"]}.items():
    ix = [kc.index(x) for x in v]
    tr["c_" + k], ts["c_" + k] = Ctr[:, ix].sum(1).round(4), Cte[:, ix].sum(1).round(4)
x = pd.concat([tr, ts], ignore_index=True)
x.to_csv(OUT / "massa.csv", index=False)

K = ["paper", "cardboard", "metal", "glass", "plastic", "textile", "electronic", "foodwaste", "(kosong)"]
for th in (.2, .4):
    print(f"\n=== jumlah citra dengan massa grup >= {th} ===")
    for s, df in (("train", tr), ("test", ts)):
        t = pd.DataFrame({g: df[df["v_" + g] >= th].kelas.value_counts() for g in GRUP}).reindex(K).fillna(0).astype(int)
        print(s)
        print(t.to_string())
print("\n=== histogram foil ===")
bins = [0, .02, .05, .1, .2, .3, .5, .7, 1.01]
print("train", np.histogram(tr.v_foil, bins)[0].tolist(), "test", np.histogram(ts.v_foil, bins)[0].tolist(), bins)
print("\n=== material DMS dominan per kelas ===")
md = ["d_Paper", "d_Cardboard", "d_Metal", "d_Glass", "d_Plastic", "d_Fabric"]
for s, df in (("train", tr), ("test", ts)):
    w = df[md].idxmax(axis=1).str[2:]
    w = w.where(df[md].max(axis=1) >= .3, "(lemah)")
    print(s)
    print(pd.crosstab(df.kelas, w).reindex(K).fillna(0).astype(int).to_string())
print("\n=== paper per keluarga: material dominan ===")
for s, df in (("train", tr[tr.kelas == "paper"]), ("test", ts[ts.kelas == "paper"])):
    w = df[md].idxmax(axis=1).str[2:]
    w = w.where(df[md].max(axis=1) >= .3, "(lemah)")
    print(s)
    print(pd.crosstab(df.fam, w).to_string())
