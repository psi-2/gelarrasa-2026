"""Ringkas meta.csv: tabel frekuensi sidik jari dan keterkaitan dengan cluster.

    python temp/eda_101_metadata/rangkum_meta.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
df = pd.read_csv(OUT / "meta.csv", dtype={"image_id": str})
pd.set_option("display.width", 220, "display.max_columns", 50, "display.max_rows", 200, "display.max_colwidth", 70)


def vc(col, n=15):
    t = pd.crosstab(df[col].fillna("<NA>"), df["split"])
    t["tot"] = t.sum(1)
    print(f"\n== {col} ({df[col].nunique()} unik) ==")
    print(t.sort_values("tot", ascending=False).head(n).to_string())


df["wh"] = df.w.astype(str) + "x" + df.h.astype(str)
for c in ["fmt", "mode", "ext", "progressive", "sof", "subsamp", "n_dqt", "n_dht", "dri", "adobe", "icc", "jfif", "dpi",
          "comment", "app_list", "markers", "info_keys", "wh", "q_ijg", "qhash", "exif_Make", "exif_Model", "exif_Software",
          "exif_Orientation", "exif_ImageDescription", "exif_Artist", "exif_Copyright", "trail"]:
    if c in df.columns:
        vc(c)

print("\nsize stat per split")
print(df.groupby("split")[["size", "bpp", "mp", "ar", "w", "h", "q_ijg"]].describe().T.round(3).to_string())
print("\nSHA1 duplikat total:", int(df.sha1.duplicated(keep=False).sum()))
print("persegi", df.assign(sq=df.w == df.h).groupby("split")["sq"].mean().round(3).to_dict())
print("kelipatan 8", df.assign(m=(df.w % 8 == 0) & (df.h % 8 == 0)).groupby("split")["m"].mean().round(3).to_dict())
dt = df["exif_DateTime"].dropna() if "exif_DateTime" in df else []
print("DateTime EXIF ada:", len(dt))
if len(dt):
    print(df[df.exif_DateTime.notna()][["split", "image_id", "exif_DateTime", "exif_Make", "exif_Model", "wh"]].head(40).to_string())
