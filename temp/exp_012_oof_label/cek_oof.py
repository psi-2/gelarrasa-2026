"""Ukur dampak koreksi label dari prediksi out-of-fold fine-tune. Tidak membaca temuan.

    python temp/exp_012_oof_label/cek_oof.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "temp" / "nb_final" / "output"
C = NB / "cache"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
ids = np.load(C / "ids_train.npy").astype(str)
v = pd.read_csv(NB / "label_final.csv", dtype={"image_id": str}).sort_values("image_id").reset_index(drop=True)
k_ens = np.zeros(len(v), dtype=int)
for k, (_, va) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(v, v.label, v.group)):
    k_ens[va] = k
v["k"] = k_ens
Pa, Pb = {}, {}
for k in range(3):
    sisa = sorted(x for x in ids if x not in set(v.image_id[v.k != k]))
    Pa[k] = pd.DataFrame(np.load(C / f"vit_k{k}_sisa.npy"), index=sisa, columns=NAMES)
    Pb[k] = pd.DataFrame(np.load(C / f"cnn_k{k}_sisa.npy"), index=sisa, columns=NAMES)

baris = []
for r in v.itertuples():
    pa, pb = Pa[r.k].loc[r.image_id], Pb[r.k].loc[r.image_id]
    ca, cb = pa.idxmax(), pb.idxmax()
    baris.append((r.image_id, r.label, ca, cb, float((pa[ca] + pb[ca]) / 2), float((pa[r.label] + pb[r.label]) / 2)))
t = pd.DataFrame(baris, columns=["image_id", "label", "ca", "cb", "p", "p_label"])
print("OOF setuju label:", round(float(((t.ca == t.label) & (t.cb == t.label)).mean()), 4))
sepakat = t[(t.ca == t.cb) & (t.ca != t.label)]
print("dua head sepakat beda dari label:", len(sepakat))
for th in (.6, .7, .8, .9, .95):
    s = sepakat[sepakat.p >= th]
    print(f"p>={th}: {len(s)}", s.groupby(["label", "ca"]).size().to_dict())

kos = sorted(set(ids) - set(v.image_id))
ra = sum(Pa[k].loc[kos] for k in range(3)) / 3
rb = sum(Pb[k].loc[kos] for k in range(3)) / 3
ca, cb = ra.idxmax(axis=1), rb.idxmax(axis=1)
p = pd.Series([(ra.loc[i, ca[i]] + rb.loc[i, ca[i]]) / 2 for i in kos], index=kos)
print("citra kosong:", len(kos), "| sepakat:", int((ca == cb).sum()))
for th in (.7, .8, .9):
    m = (ca == cb) & (p >= th)
    print(f"isi p>={th}: {int(m.sum())}", ca[m].value_counts().to_dict())
t.to_csv(Path(__file__).resolve().parent / "oof_label.csv", index=False)
