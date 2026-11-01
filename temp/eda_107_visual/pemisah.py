"""Hipotesis E dan F: fitur non-deep pemisah pasangan sulit.

Tanpa nomor cluster dan tanpa nama tertulis tangan. Kelompok acuan keluar otomatis dari label bagging eda_110,
hanya citra train dengan porsi suara >= 0,9. Fitur non-deep tidak dipakai saat membuat label itu, jadi AUC ini juga
pemeriksaan independen. Tetap bukan akurasi kelas asli.

    python temp/eda_107_visual/pemisah.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 300)
d = pd.read_csv(OUT / "gabung.csv", dtype={"image_id": str})
d = d[d.split == "train"]
lb = pd.read_csv(ROOT / "temp" / "eda_110_fusi" / "output" / "label_train_bag.csv", dtype={"image_id": str})
d = d.merge(lb[["image_id", "label", "suara"]], on="image_id")
d = d[d.suara >= .9]
fit = ["r_mean", "g_mean", "b_mean", "s_mean", "s_std", "v_mean", "v_std", "warna", "hue_entropi", "putih", "hitam", "abu", "coklat", "specular",
       "tepi_rapat", "tepi_mean2", "orient_entropi", "lap_var", "lok_std_mean", "lok_std_q10", "lok_halus", "lbp_entropi", "lbp_datar", "kontras",
       "pusat_pinggir", "pusat_rendah_sat", "pusat_spec", "sudut_mean", "sudut_var", "noise", "hf_025"] + [f"hue{i:02d}" for i in range(12)]
print(d.label.value_counts().to_string())
print(d.groupby("label")[["s_mean", "v_mean", "putih", "abu", "coklat", "specular", "tepi_rapat", "orient_entropi", "lbp_entropi", "lok_halus", "kontras", "pusat_pinggir"]].median().round(3).to_string())
cv = StratifiedKFold(5, shuffle=True, random_state=42)
for a, b in [("plastic", "glass"), ("plastic", "paper"), ("glass", "paper"), ("paper", "cardboard"), ("metal", "plastic"), ("metal", "paper"), ("metal", "glass")]:
    s = d[d.label.isin([a, b])]
    y = (s.label == a).astype(int).values
    t = pd.Series({f: roc_auc_score(y, s[f].fillna(0)) for f in fit})
    t = (t - .5).abs().sort_values(ascending=False).head(5) + .5
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=.08, random_state=42)
    p = cross_val_predict(m, s[fit].fillna(0).values, y, cv=cv, method="predict_proba")[:, 1]
    keluarga = s.fam.value_counts().index[0]
    k = (s.fam == keluarga).values
    teks = ""
    if k.sum() > 60 and y[k].min() != y[k].max() and min(y[k].sum(), (1 - y[k]).sum()) >= 10:
        pk = cross_val_predict(m, s[k][fit].fillna(0).values, y[k], cv=cv, method="predict_proba")[:, 1]
        teks = f" | hanya {keluarga} n={int(k.sum())} AUC {roc_auc_score(y[k], pk):.3f}"
    print(f"{a} vs {b}: n={len(s)}, AUC GBM {roc_auc_score(y, p):.3f}{teks} | terbaik: " + ", ".join(f"{i} {v:.2f}" for i, v in t.items()))
