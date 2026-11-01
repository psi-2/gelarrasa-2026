"""Satukan label EDA v2 dan riset v3 lewat aturan otomatis yang dibekukan. Tidak membaca temuan.

Aturan: 14 citra diragukan dibuang dan 3 koreksi riset diterapkan. Isian yang bertentangan antara v2 dan v3 dibuang.

    python temp/exp_011_label_v4/gabung_label.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
V2 = ROOT / "temp" / "eda_112_label_v2" / "output" / "label_folds_v2.csv"
V3 = ROOT / "temp" / "research_uji_crossfit" / "output" / "label_folds_v3.csv"
RAGU = ROOT / "temp" / "research_uji_crossfit" / "output" / "daftar_label_diragukan.csv"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]

v2 = pd.read_csv(V2, dtype={"image_id": str}).set_index("image_id")
v3 = pd.read_csv(V3, dtype={"image_id": str}).set_index("image_id")
ragu = pd.read_csv(RAGU, dtype=str)
print("kolom diragukan:", list(ragu.columns))
print(ragu.head(3).to_string())
aksi = "aksi" if "aksi" in ragu else ragu.columns[-1]
print(ragu[aksi].value_counts().to_string())

gab = v2.copy()
tambah = v3.index.difference(v2.index)
gab = pd.concat([gab, v3.loc[tambah].reindex(columns=gab.columns)])
if gab["fam"].isna().any() or gab["group"].isna().any():
    print("baris tanpa group atau fam:", int(gab["group"].isna().sum()))

konflik = v2.index.intersection(v3.index)
beda = konflik[v2.loc[konflik, "label"].values != v3.loc[konflik, "label"].values]
print(f"v2 {len(v2)} | v3 {len(v3)} | hanya di v3 {len(tambah)} | berbeda {len(beda)}")
print("rincian beda v2 lalu v3:", pd.crosstab(v2.loc[beda, "label"], v3.loc[beda, "label"]).stack().loc[lambda s: s > 0].to_dict())
gab = gab.drop(index=beda)

id_buang = ragu.loc[ragu[aksi].str.contains("buang", case=False), "image_id"].str.zfill(4)
koreksi = ragu[ragu[aksi].str.contains("koreksi", case=False)]
gab = gab.drop(index=id_buang[id_buang.isin(gab.index)])
ke = [c for c in ragu.columns if c in ("label_baru", "ke", "baru", "label_koreksi")]
print("kolom koreksi:", ke)
for _, r in koreksi.iterrows():
    gab.loc[str(r["image_id"]).zfill(4), "label"] = r[ke[0]]

gab = gab.sort_index().reset_index()
gab["fold"] = gab["fold"].fillna(1).astype(int)
print("hasil:", len(gab), gab["label"].value_counts().reindex(NAMES).fillna(0).astype(int).to_dict())
print("fold:", gab["fold"].value_counts().to_dict())
gab.to_csv(OUT / "label_folds_v4.csv", index=False)
