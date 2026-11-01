"""Diagnosis agregat: berapa kesalahan ada di citra yang disepakati semua model. Hanya angka per kelas.

    python temp/exp_007_eval_temuan/analisis_konsensus.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "temp"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
ss = pd.read_csv(ROOT / "data" / "SampleSubmission_DSC.csv", dtype=str)
temuan_df = pd.read_csv(ROOT / "data" / "temuan" / "anotate.csv", dtype=str)
temuan_df["image_id"] = temuan_df["image_id"].str.rsplit(".", n=1).str[0].str.zfill(3)
y = temuan_df.set_index("image_id").loc[ss["image_id"], "waste_category"].str.strip().values

PRED = {"exp009": T / "exp_009_t1_dinov3/output/draft_submission_t1_dinov3.csv",
        "vit_v2": T / "exp_010_dua_backbone/output/draft_vit_v2.csv",
        "cnn_v2": T / "exp_010_dua_backbone/output/draft_cnn_v2.csv",
        "lr_t1": T / "eda_110_fusi/output/kandidat_test_t1_in1k_lr.csv"}
P = pd.DataFrame({k: pd.read_csv(v, dtype=str)["waste_category"].values for k, v in PRED.items()})
sepakat = P.nunique(axis=1).values == 1
salah = P["vit_v2"].values != y
print("citra sepakat semua:", int(sepakat.sum()), "| salah di antaranya:", int((sepakat & (P["vit_v2"].values != y)).sum()))
print("citra tidak sepakat:", int((~sepakat).sum()), "| salah vit_v2 di antaranya:", int((~sepakat & salah).sum()))
s = sepakat & salah
print("salah konsensus per kelas temuan:", pd.Series(y[s]).value_counts().reindex(NAMES).fillna(0).astype(int).to_dict())
print("salah konsensus per prediksi:", pd.Series(P["vit_v2"].values[s]).value_counts().reindex(NAMES).fillna(0).astype(int).to_dict())
print("batas atas bila semua citra tidak sepakat benar:", int((~s).sum()), "dari", len(y), round(float((~s).mean()), 4))
