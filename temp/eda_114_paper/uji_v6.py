"""Evaluasi v6 tanpa temuan, tidak mengubah label. Test hanya diprediksi, fit hanya pada train.

1. Recall foil leave-group-out pada LR, bobot sampel 1, 3, 10.
2. Perubahan prediksi test LR v5 ke v6 dan nasib citra test foil.
3. Audit glass lawan plastic: dua sumber bersepakat melawan label. Tidak mengubah apa pun.

Keluaran: output/test_foil.csv (untuk diperiksa koordinator, bukan untuk fit)

    python temp/eda_114_paper/uji_v6.py
"""
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
E13 = ROOT / "temp" / "eda_113_elektronik" / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
ID = {c: i for i, c in enumerate(NAMES)}
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)
d = pd.read_csv(OUT / "dasar.csv", dtype={"image_id": str})
te = pd.read_csv(OUT / "dasar_test.csv", dtype={"image_id": str})
x = pd.read_csv(OUT / "massa.csv", dtype={"image_id": str})
xtr, xte = x[x.split == "train"].reset_index(drop=True), x[x.split == "test"].reset_index(drop=True)
v5 = pd.read_csv(E13 / "label_folds_v5.csv", dtype={"image_id": str})
v6 = pd.read_csv(OUT / "label_folds_v6.csv", dtype={"image_id": str})
X = np.hstack([normalize(np.load(E6 / "emb_train.npy")), normalize(np.load(RC / "emb_cnn_train.npy"))])
Xt = np.hstack([normalize(np.load(E6 / "emb_test.npy")), normalize(np.load(RC / "emb_cnn_test.npy"))])
pos = dict(zip(d.image_id, range(len(d))))
grp = d.grp.values


def data(v):
    return v.image_id.map(pos).values, v.label.map(ID).values


i6, y6 = data(v6)
foil_id = set(xtr.image_id[xtr.v_foil >= .2])
is_f = v6.image_id.isin(foil_id).values
print("foil train di v6:", int(is_f.sum()), "group", len(set(grp[i6[is_f]])))
print("\n== 1. recall leave-group-out ==")
for w in (1, 3, 10):
    hit, pm, pp = 0, [], []
    for g in sorted(set(grp[i6[is_f]])):
        va = (grp[i6] == g) & is_f
        tr_ = grp[i6] != g
        m = LogisticRegression(C=3, max_iter=1500).fit(X[i6[tr_]], y6[tr_], sample_weight=np.where(is_f[tr_], w, 1.0))
        p = m.predict_proba(X[i6[va]])
        hit += int((p.argmax(1) == ID["metal"]).sum())
        pm += p[:, ID["metal"]].tolist()
        pp += p[:, ID["paper"]].tolist()
    print(f"bobot {w}: metal {hit} dari {int(is_f.sum())}, median p_metal {np.median(pm):.3f}, median p_paper {np.median(pp):.3f}", flush=True)

print("\n== 2. prediksi test LR ==")
i5, y5 = data(v5)
p5 = LogisticRegression(C=3, max_iter=2000).fit(X[i5], y5).predict_proba(Xt)
f_te = (xte.v_foil >= .2).values
print("LR v5 test:", dict(zip(NAMES, np.bincount(p5.argmax(1), minlength=8).tolist())))
for w in (1, 3, 10):
    p6 = LogisticRegression(C=3, max_iter=2000).fit(X[i6], y6, sample_weight=np.where(is_f, w, 1.0)).predict_proba(Xt)
    a, b = p5.argmax(1), p6.argmax(1)
    print(f"bobot {w}: berubah {int((a != b).sum())}, test {dict(zip(NAMES, np.bincount(b, minlength=8).tolist()))}")
    if (a != b).any():
        print(pd.crosstab(pd.Series(np.array(NAMES)[a[a != b]], name="v5"), pd.Series(np.array(NAMES)[b[a != b]], name="v6")).to_string())
    print("   test foil:", [NAMES[i] for i in b[f_te]], "p_metal", np.round(p6[f_te, ID['metal']], 3).tolist())

print("\n== 3. audit glass lawan plastic, dua sumber melawan label ==")
t = xtr.assign(pl=d.pl, p_plastic=d.p_plastic, p_glass=d.p_glass)
a_ = t[(t.kelas == "glass") & (t.v_botol_plastik >= .2) & (t.d_Plastic >= .3)]
b_ = t[(t.kelas == "plastic") & (t.v_kaca >= .2) & (t.d_Glass >= .3)]
print("glass berlabel, 22k botol plastik dan DMS plastik:", len(a_), "| LR setuju plastic:", int((a_.pl == "plastic").sum()))
print("plastic berlabel, 22k kaca dan DMS kaca:", len(b_), "| LR setuju glass:", int((b_.pl == "glass").sum()))
c_ = t[(t.kelas == "paper") & (t.v_kardus >= .2) & (t.d_Cardboard >= .3)]
e_ = t[(t.kelas == "cardboard") & (t.v_kertas >= .2) & (t.d_Paper >= .3)]
print("paper berlabel, kardus dua sumber:", len(c_), "| cardboard berlabel, kertas dua sumber:", len(e_))
s = te.assign(v_foil=xte.v_foil)[f_te][["image_id", "fam", "pred", "conf", "p_metal", "v_foil"]]
s.to_csv(OUT / "test_foil.csv", index=False)
print("\ntest_foil.csv:", len(s), "baris, pred", s.pred.value_counts().to_dict())
print("perkiraan test jika foil berpindah: paper", int((te.pred == "paper").sum()) - int(f_te.sum()), "metal", int((te.pred == "metal").sum()) + int(f_te.sum()))
