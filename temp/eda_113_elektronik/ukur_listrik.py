"""Ukur massa kosakata listrik per citra dan per model. Tidak mengubah label. Test hanya dihitung.

Keluaran: output/massa_listrik.csv (train dan test)

    python temp/eda_113_elektronik/ukur_listrik.py
"""
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd

from kosakata_listrik import bangun

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
E8 = ROOT / "temp" / "exp_008_in22k_map" / "output"
MODEL = ("convnext_l", "swinv2_l", "beitv2_l")
pd.set_option("display.width", 230, "display.max_columns", 40, "display.max_rows", 200)
d = pd.read_csv(OUT / "dasar.csv", dtype={"image_id": str})
te = pd.read_csv(OUT / "dasar_test.csv", dtype={"image_id": str})
M, _ = bangun()
Ptr = [np.load(E8 / f"p22k_train_{m}.npy") for m in MODEL]
Pte = [np.load(E8 / f"p22k_test_{m}.npy") for m in MODEL]
tr = pd.DataFrame({"split": "train", "image_id": d.image_id, "fam": d.fam, "kelas": d.label.fillna("(kosong)")})
tes = pd.DataFrame({"split": "test", "image_id": te.image_id, "fam": te.fam, "kelas": te.pred})
for k, v in M.items():
    a = np.stack([p[:, v].sum(1) for p in Ptr], 1)
    b = np.stack([p[:, v].sum(1) for p in Pte], 1)
    tr[k], tes[k] = a.mean(1).round(4), b.mean(1).round(4)
    tr[k + "_min"], tes[k + "_min"] = np.sort(a, 1)[:, 1].round(4), np.sort(b, 1)[:, 1].round(4)  # median 3 model
tr["kan"] = np.mean([p[:, np.zeros(p.shape[1], bool)].sum(1) for p in Ptr], 0)
del Ptr, Pte
x = pd.concat([tr, tes], ignore_index=True).drop(columns=["kan"])
x.to_csv(OUT / "massa_listrik.csv", index=False)

for k in ("sel", "lampu", "kabel", "papan"):
    print(f"\n=== {k}: kuantil massa, train ===")
    print(np.round(tr[k].quantile([.5, .9, .95, .98, .99, .995]).values, 4))
    bins = [0, .02, .05, .1, .2, .3, .5, .7, 1.01]
    print("histogram train:", np.histogram(tr[k], bins)[0].tolist(), "| test:", np.histogram(tes[k], bins)[0].tolist(), "| tepi", bins)
    for th in (.1, .2, .3, .5):
        s, q = tr[tr[k + "_min"] >= th], tes[tes[k + "_min"] >= th]
        print(f"median 3 model >= {th}: train {len(s)} {s.kelas.value_counts().to_dict()} | test {len(q)} {q.kelas.value_counts().to_dict()}")
    s = tr[tr[k + "_min"] >= .3]
    if len(s):
        print("per keluarga x label, train:")
        print(pd.crosstab(s.fam, s.kelas).to_string())
    q = tes[tes[k + "_min"] >= .3]
    if len(q):
        print("per keluarga x prediksi, test:")
        print(pd.crosstab(q.fam, q.kelas).to_string())
