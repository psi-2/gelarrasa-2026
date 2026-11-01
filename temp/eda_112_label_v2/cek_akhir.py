"""Validasi akhir label_folds_v2.csv, metadata tiga citra koreksi, dan ketersediaan bobot backbone rencana.

Tidak membuka citra, tidak membaca temuan.

    python temp/eda_112_label_v2/cek_akhir.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
v = pd.read_csv(OUT / "label_folds_v2.csv", dtype={"image_id": str})
l = pd.read_csv(ROOT / "temp" / "exp_009_t1_dinov3" / "output" / "label_folds.csv", dtype={"image_id": str})
print("kolom", list(v.columns), "baris", len(v), "NaN", int(v.isna().sum().sum()), "id unik", v.image_id.is_unique)
m = v.merge(l[["image_id", "label", "fold"]], on="image_id", suffixes=("", "_l"))
print("citra lama", len(m), "fold sama", int((m.fold == m.fold_l).sum()), "label berubah", int((m.label != m.label_l).sum()), "citra baru", len(v) - len(m))
print("fold 0 citra sama persis:", set(v.image_id[v.fold == 0]) == set(l.image_id[l.fold == 0]))
print("citra baru di fold 0:", int((~v.image_id[v.fold == 0].isin(l.image_id)).sum()))
print("group lintas fold:", int((v.groupby("group").fold.nunique() > 1).sum()))
ex = pd.read_csv(ROOT / "temp" / "eda_101_metadata" / "output" / "meta.csv", dtype={"image_id": str})
ex = ex[(ex.split == "train") & ex.image_id.isin(["0550", "0680", "1920"])]
print(ex[["image_id", "w", "h", "fmt", "qhash", "exif_ImageDescription", "exif_Copyright", "exif_Artist"]].to_string(index=False))
hub = Path(r"D:\cache_moved\huggingface\hub")
for r in ("models--timm--convnext_large.dinov3_lvd1689m", "models--timm--vit_large_patch16_dinov3.lvd1689m"):
    b = [p for p in (hub / r).rglob("*") if p.is_file() and p.stat().st_size > 5e6]
    print(r, [(p.name, round(p.stat().st_size / 1e6, 1)) for p in b])
