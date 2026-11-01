"""Kaitan sumber berkas dengan nama cluster yang sudah benar, dan proporsi train vs test per sumber.

    python temp/eda_102_sumber/sumber2.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
pd.set_option("display.width", 240, "display.max_columns", 50, "display.max_rows", 300)
meta = pd.read_csv(ROOT / "temp" / "eda_101_metadata" / "output" / "meta.csv", dtype={"image_id": str})
a = pd.read_csv(OUT / "assign.csv", dtype={"image_id": str})
df = meta.merge(a, on=["split", "image_id"])
df["wh"] = df.w.astype(str) + "x" + df.h.astype(str)
top = df.wh.value_counts()
df["wh_b"] = df.wh.where(df.wh.map(top) >= 10, "lain")
q = df.qhash.value_counts()
df["q_b"] = df.qhash.where(df.qhash.map(q) >= 6, "qlain").fillna("noq")
df.to_csv(OUT / "meta_assign.csv", index=False)

# keluarga sumber berbasis kuantisasi dan ukuran
fam = np.select(
    [df.q_b == "4885dc9d7c", df.q_b == "2527d1e5ea", df.q_b == "74fbe5a861",
     (df.q_b == "e09fb31b26") & (df.wh == "640x640"), (df.q_b == "e09fb31b26") & (df.wh == "416x416"),
     (df.q_b == "e09fb31b26") & df.wh.isin(["640x480", "512x384"]), (df.q_b == "e09fb31b26")],
    ["F_q4885_besar", "F_q2527_400x", "F_q74fb", "F_640", "F_416", "F_q75_4x3", "F_q75_lain"], "F_lain_web")
df["fam"] = fam
df.to_csv(OUT / "meta_assign.csv", index=False)
print(pd.crosstab(df.fam, df.split).assign(rasio_test=lambda t: (t.test / t.sum(1)).round(3)).to_string())
print()
x = pd.crosstab(df.fam, df.nama).reindex(columns=NAMES).fillna(0).astype(int)
print("nama cluster per keluarga, semua split")
print(x.to_string())
print("\nhanya test")
print(pd.crosstab(df[df.split == "test"].fam, df[df.split == "test"].nama).reindex(columns=NAMES).fillna(0).astype(int).to_string())
print("\nhanya train")
print(pd.crosstab(df[df.split == "train"].fam, df[df.split == "train"].nama).reindex(columns=NAMES).fillna(0).astype(int).to_string())
print("\nchi2 fam vs nama p =", chi2_contingency(pd.crosstab(df.fam, df.nama))[1])

print("\nketinggian keluarga q2527 per subset ukuran")
s = df[df.fam == "F_q2527_400x"]
print(pd.crosstab(s.wh_b, s.nama).to_string())
s = df[df.fam == "F_q4885_besar"]
print(pd.crosstab(s.wh_b, s.nama).to_string())
print(s.groupby("cluster").size().sort_values(ascending=False).head(8).to_string())
s = df[df.fam == "F_q74fb"]
print(pd.crosstab(s.wh_b, s.nama).to_string())
