"""Validasi akhir label_folds_v5.csv terhadap format v4. Tidak membaca temuan.

    python temp/eda_113_elektronik/cek_v5.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
v5 = pd.read_csv(OUT / "label_folds_v5.csv", dtype={"image_id": str})
v4 = pd.read_csv(ROOT / "temp" / "exp_011_label_v4" / "output" / "label_folds_v4.csv", dtype={"image_id": str})
print("kolom sama:", list(v5.columns) == list(v4.columns), list(v5.columns), "| baris", len(v5), "| id unik", v5.image_id.is_unique)
print("kosong per kolom:", v5.isna().sum().to_dict())
m = v5.merge(v4, on="image_id", suffixes=("", "_l"))
print("fold citra lama sama:", bool((m.fold == m.fold_l).all()), "| label berubah", int((m.label != m.label_l).sum()), m[m.label != m.label_l].groupby(["label_l", "label"]).size().to_dict())
print("citra baru:", len(v5) - len(m), v5[~v5.image_id.isin(v4.image_id)].label.value_counts().to_dict(), "fold", v5[~v5.image_id.isin(v4.image_id)].fold.value_counts().to_dict())
print("group lintas fold:", int((v5.groupby("group").fold.nunique() > 1).sum()), "| fold", v5.fold.value_counts().to_dict())
t = pd.read_csv(OUT / "test_listrik.csv", dtype={"image_id": str})
print("test_listrik:", len(t), "| sel_m >= .2:", int((t.sel_m >= .2).sum()), t[t.sel_m >= .2].pred.value_counts().to_dict())
