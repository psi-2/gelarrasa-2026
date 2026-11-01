"""Cari kolom kosong di label v4. Tidak mengubah apa pun.

    python temp/eda_113_elektronik/cek_nan.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
v4 = pd.read_csv(ROOT / "temp" / "exp_011_label_v4" / "output" / "label_folds_v4.csv", dtype={"image_id": str})
print(v4.isna().sum().to_dict())
print(v4[v4.isna().any(axis=1)].assign(ada_id=True).drop(columns=["image_id"]).to_string(index=False))
