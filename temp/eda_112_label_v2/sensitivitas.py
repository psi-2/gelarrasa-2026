"""Batas label: seberapa bergantung hasil pada e_lr, tanpa menyentuh label_folds_v2.csv.

Skenario A: DS dengan semua annotator (aturan beku). Skenario B: DS tanpa e_lr, hanya sumber independen.
Pemeriksaan out-of-sample: EXP-009 fold 0 tidak dilatih pada label hold-out dan bukan masukan DS.
Tanpa temuan, test tidak dipakai.

Keluaran: output/batas_label_rentang.csv

    python temp/eda_112_label_v2/sensitivitas.py
"""
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
ID = {c: i for i, c in enumerate(NAMES)}
EL = ID["electronic"]
nt = 1920
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)
d = pd.read_csv(OUT / "rincian_v2.csv", dtype={"image_id": str})
v = pd.read_csv(OUT / "label_folds_v2.csv", dtype={"image_id": str})


def ds(bobot, iters=5):
    ann = list(bobot)
    W = np.array([bobot[k] for k in ann])
    R = np.stack([d[k].map(ID).values for k in ann], 1)
    dm = [i for i, k in enumerate(ann) if k.endswith("_dms")]
    q = np.zeros((nt, 8))
    for j, w in enumerate(W):
        q[np.arange(nt), R[:, j]] += w
    q = q / q.sum(1, keepdims=True)
    for _ in range(iters):
        pi = (q.sum(0) + 1) / (q.sum() + 8)
        C = np.zeros((len(ann), 8, 8))
        for j in range(len(ann)):
            for r in range(8):
                C[j, :, r] = q[R[:, j] == r].sum(0)
        C = (C + 1) / (C + 1).sum(2, keepdims=True)
        non = [c for c in range(8) if c != EL]
        for j in dm:
            C[j, EL, :] = (C[j, non, :] * pi[non, None]).sum(0) / pi[non].sum()
        L = np.tile(np.log(pi), (nt, 1))
        for j, w in enumerate(W):
            L += w * np.log(C[j, :, R[:, j]])
        q = np.exp(L - L.max(1, keepdims=True))
        q = q / q.sum(1, keepdims=True)
    return q


A = {"c_in1k": .5, "c_in22k": .5, "c_dms": .5, "b_in22k": .5, "b_dms": .5, "e_lr": 1.0}
B = {k: x for k, x in A.items() if k != "e_lr"}
qa, qb = ds(A), ds(B)
ix = {i: n for n, i in enumerate(d.image_id)}
pos = v.image_id.map(ix).values
yv = v.label.map(ID).values
sa = qa[pos, yv]
sb = qb[pos, yv]
print("setuju argmax DS A vs B, train:", round(float((qa.argmax(1) == qb.argmax(1)).mean()), 4))
print("median q maks B:", round(float(np.median(qb.max(1))), 3), "| q B >= .9:", int((qb.max(1) >= .9).sum()))

rows = []
for c in NAMES:
    m = (v.label == c).values
    rows.append({"kelas": c, "n": int(m.sum()), "A_harapan": round(float((1 - sa[m]).sum()), 1), "A_q<.5": int((sa[m] < .5).sum()),
                 "B_harapan": round(float((1 - sb[m]).sum()), 1), "B_q<.5": int((sb[m] < .5).sum()), "A_dan_B_q<.5": int(((sa[m] < .5) & (sb[m] < .5)).sum())})
t = pd.DataFrame(rows)
t.loc[len(t)] = ["TOTAL"] + [round(t[c].sum(), 1) for c in t.columns[1:]]
print(t.to_string(index=False))
t.to_csv(OUT / "batas_label_rentang.csv", index=False)

print("\nkeluarga, B q<.5:", pd.Series(v.fam.values[sb < .5]).value_counts().to_dict())
print("per keputusan, B q<.5:")
x = v.merge(d[["image_id", "keputusan", "e9", "e9_p"]], on="image_id")
x["sb"], x["sa"] = sb, sa
print(x.groupby("keputusan").apply(lambda g: pd.Series({"n": len(g), "B_q<.5": int((g.sb < .5).sum()), "B_harapan": round(float((1 - g.sb).sum()), 1)}), include_groups=False).to_string())
t_ = x[x.keputusan == "tambah"]
print("tambah: setuju EXP-009", int((t_.e9 == t_.label).sum()), "dari", len(t_), "| tidak setuju per kelas label:", t_[t_.e9 != t_.label].label.value_counts().to_dict())
print("tambah tidak setuju, rata keyakinan EXP-009:", round(float(t_[t_.e9 != t_.label].e9_p.mean()), 3))
print("pasangan tambah vs EXP-009:")
print(pd.crosstab(t_[t_.e9 != t_.label].label, t_[t_.e9 != t_.label].e9).to_string())

print("\npemeriksaan out-of-sample hold-out 388 (EXP-009 bukan masukan DS):")
h = x[x.fold == 0]
for nama, kol, th in (("B q<.5", "sb", .5), ("B q<.9", "sb", .9), ("A q<.9", "sa", .9)):
    f = h[kol] < th
    print(f"{nama}: ditandai {int(f.sum())}, EXP-009 beda dari label {int((h.e9 != h.label)[f].sum())} | tidak ditandai {int((~f).sum())}, beda {int((h.e9 != h.label)[~f].sum())}")

print("\nkoreksi:")
print(d[d.keputusan == "koreksi"][["image_id", "fam", "label", "akhir", "q_max", "e_lr", "c_in1k", "c_in22k", "c_dms", "b_in22k", "b_dms", "e9"]].to_string(index=False))

print("\npasangan metal-electronic pada label v2:")
for a, b in (("electronic", "metal"), ("metal", "electronic")):
    s = d[d.akhir == a]
    k = (s.b_in22k == b) | (s.c_in22k == b)
    print(f"{a}: in22k bilang {b} {int(k.sum())}; dari itu in1k citra setuju {a} {int((s.c_in1k == a)[k].sum())}; e_lr setuju {a} {int((s.e_lr == a)[k].sum())}")
print("citra kosong 83, kelas DS terbanyak:", d[d.keputusan == "kosong"].ds.value_counts().to_dict(), "| q median", round(float(d[d.keputusan == "kosong"].q_max.median()), 3))
