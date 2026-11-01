"""Evaluasi v5 tanpa temuan, tidak mengubah label. Test hanya diprediksi, fit hanya pada train.

1. Recall baterai saat group-nya ditahan (leave-group-out) pada LR embedding, dengan bobot sampel 1, 3, 10.
2. Perubahan prediksi test LR v4 ke v5.
3. Kelompok test berprediksi metal: massa perangkat luas dari peta beku 22k dan EoMT.

Keluaran: output/test_listrik.csv (citra test dengan massa listrik tinggi, untuk diperiksa koordinator, bukan untuk fit)

    python temp/eda_113_elektronik/uji_v5.py
"""
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
RC = ROOT / "temp" / "research_uji_crossfit" / "output"
E8 = ROOT / "temp" / "exp_008_in22k_map" / "output"
E12 = ROOT / "temp" / "eda_112_label_v2" / "output"
EM = ROOT / "temp" / "eda_106_eomt" / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
ID = {c: i for i, c in enumerate(NAMES)}
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)
d = pd.read_csv(OUT / "dasar.csv", dtype={"image_id": str})
te = pd.read_csv(OUT / "dasar_test.csv", dtype={"image_id": str})
x = pd.read_csv(OUT / "massa_listrik.csv", dtype={"image_id": str})
xtr, xte = x[x.split == "train"].reset_index(drop=True), x[x.split == "test"].reset_index(drop=True)
v5 = pd.read_csv(OUT / "label_folds_v5.csv", dtype={"image_id": str})
v4 = pd.read_csv(ROOT / "temp" / "exp_011_label_v4" / "output" / "label_folds_v4.csv", dtype={"image_id": str})
ch = pd.read_csv(OUT / "perubahan_v5.csv", dtype={"image_id": str})
X = np.hstack([normalize(np.load(E6 / "emb_train.npy")), normalize(np.load(RC / "emb_cnn_train.npy"))])
Xt = np.hstack([normalize(np.load(E6 / "emb_test.npy")), normalize(np.load(RC / "emb_cnn_test.npy"))])
pos = dict(zip(d.image_id, range(len(d))))
grp = d.grp.values


def data(v):
    i = v.image_id.map(pos).values
    return i, v.label.map(ID).values


i5, y5 = data(v5)
bat = set(ch.image_id[ch.jenis == "ubah"])
is_bat = v5.image_id.isin(bat).values
print("baterai", int(is_bat.sum()), "group unik", len(set(grp[i5[is_bat]])))

print("\n== 1. recall leave-group-out: baterai ditahan per group ==")
for w in (1, 3, 10):
    hit, pel, pm = 0, [], []
    for g in sorted(set(grp[i5[is_bat]])):
        va = np.isin(grp[i5], [g]) & is_bat
        tr_ = ~np.isin(grp[i5], [g])
        sw = np.where(is_bat[tr_], w, 1.0)
        m = LogisticRegression(C=3, max_iter=1500).fit(X[i5[tr_]], y5[tr_], sample_weight=sw)
        p = m.predict_proba(X[i5[va]])
        hit += int((p.argmax(1) == ID["electronic"]).sum())
        pel += p[:, ID["electronic"]].tolist()
        pm += p[:, ID["metal"]].tolist()
    print(f"bobot {w}: electronic {hit} dari {int(is_bat.sum())}, median p_electronic {np.median(pel):.3f}, median p_metal {np.median(pm):.3f}", flush=True)

print("\n== 2. prediksi test LR ==")
i4, y4 = data(v4)
m4 = LogisticRegression(C=3, max_iter=2000).fit(X[i4], y4)
p4 = m4.predict_proba(Xt)
hasil = {}
for w in (1, 3, 10):
    sw = np.where(is_bat, w, 1.0)
    m5 = LogisticRegression(C=3, max_iter=2000).fit(X[i5], y5, sample_weight=sw)
    p5 = m5.predict_proba(Xt)
    hasil[w] = p5
    a, b = p4.argmax(1), p5.argmax(1)
    pindah = pd.crosstab(pd.Series(np.array(NAMES)[a[a != b]], name="v4"), pd.Series(np.array(NAMES)[b[a != b]], name=f"v5_w{w}"))
    print(f"bobot {w}: berubah {int((a != b).sum())}, test per kelas {dict(zip(NAMES, np.bincount(b, minlength=8).tolist()))}")
    print(pindah.to_string() if (a != b).any() else "  tidak ada")
sel = (xte.sel_min >= .2).values
print("\ncitra test sel >= 0,2:", int(sel.sum()), "| prediksi v4:", pd.Series(np.array(NAMES)[p4.argmax(1)[sel]]).value_counts().to_dict())
for w in (1, 3, 10):
    print(f"  v5 bobot {w}:", pd.Series(np.array(NAMES)[hasil[w].argmax(1)[sel]]).value_counts().to_dict(), "median p_electronic", round(float(np.median(hasil[w][sel, ID['electronic']])), 3))

print("\n== 3. test berprediksi metal: bukti perangkat luas ==")
M22 = np.load(E12 / "peta_in22k.npy")
el = M22[2] > 0
p = np.mean([np.load(E8 / f"p22k_test_{m}.npy")[:, el].sum(1) for m in ("convnext_l", "swinv2_l", "beitv2_l")], 0)
kc = json.loads((EM / "coco_kelas.json").read_text(encoding="utf-8"))
ka = json.loads((EM / "ade_kelas.json").read_text(encoding="utf-8"))
Ec = np.load(EM / "coco_test.npy").mean(2)[:, [kc.index(k) for k in ("tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "toaster", "hair drier", "oven", "refrigerator")]].sum(1)
Ea = np.load(EM / "ade_test.npy").mean(2)[:, [ka.index(k) for k in ("computer", "monitor", "crt screen", "television receiver", "screen", "microwave", "washer", "dishwasher", "stove", "fan")]].sum(1)
t = te.assign(p22=p.round(3), coco=Ec.round(3), ade=Ea.round(3), sel=xte.sel, sel_m=xte.sel_min, papan=xte.papan, lampu=xte.lampu, kabel=xte.kabel)
s = t[t.pred == "metal"]
print(s.groupby("fam")[["p22", "coco", "ade", "sel"]].agg(["median", "max"]).round(3).to_string())
u = s[~(s.sel_m >= .2)].sort_values("p22", ascending=False).head(8)
print("metal non-baterai, massa perangkat 22k tertinggi:")
print(u[["fam", "p_el", "p22", "coco", "ade", "sel", "papan", "kabel"]].to_string(index=False))
print("\nkelas lain dengan bukti perangkat kuat (p22 >= .5) di test:", t[(t.p22 >= .5) & (t.pred != "electronic")].pred.value_counts().to_dict())
print("electronic prediksi vs keluarga:", t[t.pred == "electronic"].fam.value_counts().to_dict())
t[(t.sel_m >= .2) | (t.papan >= .2) | (t.lampu >= .2)][["image_id", "fam", "pred", "sel", "sel_m", "papan", "lampu"]].to_csv(OUT / "test_listrik.csv", index=False)
print("baterai test per keluarga:", t[t.sel_m >= .2].fam.value_counts().to_dict(), "| pred", t[t.sel_m >= .2].pred.value_counts().to_dict())
