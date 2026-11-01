"""Skor semua prediksi test terhadap temuan. Hanya evaluasi dan diagnosis.

Satu-satunya script yang membaca data/temuan/. Hasilnya tidak boleh masuk ke label, fitur, atau tuning.

    python temp/exp_007_eval_temuan/eval_temuan.py [nama=path/ke/prediksi.csv ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
T = ROOT / "temp"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
SEED = 42
N_BOOT = 2000


def stem(s):
    return s.astype(str).str.strip().str.rsplit(".", n=1).str[0].str.zfill(3)


temuan_df = pd.read_csv(ROOT / "data" / "temuan" / "anotate.csv", dtype=str)
temuan_df["image_id"] = stem(temuan_df["image_id"])
temuan_df["waste_category"] = temuan_df["waste_category"].str.strip()
temuan_df = temuan_df.set_index("image_id")
ss = pd.read_csv(ROOT / "data" / "SampleSubmission_DSC.csv", dtype=str)
print("baris temuan:", len(temuan_df), "| ID unik:", temuan_df.index.is_unique,
      "| ID cocok test:", set(temuan_df.index) == set(ss["image_id"]))
print("kelas di luar 8 kategori:", sorted(set(temuan_df["waste_category"]) - set(NAMES)))
print("kosong:", int(temuan_df["waste_category"].isna().sum()))
print("distribusi temuan:", temuan_df["waste_category"].value_counts().reindex(NAMES).fillna(0).astype(int).to_dict())
y = temuan_df.loc[ss["image_id"], "waste_category"].values

PRED = {
    "EXP-005 ConvNeXtV2 pemetaan": pd.read_csv(T / "exp_005_imagenet_map" / "output" / "draft_submission_convnextv2.csv", dtype=str)["waste_category"].values,
    "EXP-006 DINOv3 label cluster": pd.read_csv(T / "exp_006_cluster_dinov3" / "output" / "draft_submission_dinov3_cluster.csv", dtype=str)["waste_category"].values,
}
for arg in sys.argv[1:]:
    nama, path = arg.split("=", 1)
    d = pd.read_csv(ROOT / path, dtype=str)
    kolom = "waste_category" if "waste_category" in d else "pred"
    d["image_id"] = stem(d["image_id"])
    PRED[nama] = d.set_index("image_id").loc[ss["image_id"], kolom].values

rng = np.random.default_rng(SEED)
boot_idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
hasil, per_kelas = {}, {}
for nama, p in PRED.items():
    mf1 = f1_score(y, p, labels=NAMES, average="macro", zero_division=0)
    acc = accuracy_score(y, p)
    b = [f1_score(y[i], p[i], labels=NAMES, average="macro", zero_division=0) for i in boot_idx]
    hasil[nama] = {"macro_f1": round(float(mf1), 4), "accuracy": round(float(acc), 4),
                   "benar": int((y == p).sum()),
                   "macro_f1_ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]}
    per_kelas[nama] = dict(zip(NAMES, f1_score(y, p, labels=NAMES, average=None, zero_division=0).round(4).tolist()))

print("\nSkor temuan:")
print(pd.DataFrame(hasil).T.to_string())
print("\nF1 per kelas:")
print(pd.DataFrame(per_kelas).T.to_string())
for nama in list(PRED)[1:]:
    print(f"\n{nama}, baris temuan, kolom prediksi:")
    print(pd.crosstab(y, PRED[nama]).reindex(index=NAMES, columns=NAMES).fillna(0).astype(int).to_string())

(OUT / "skor_temuan.json").write_text(json.dumps({"skor": hasil, "per_kelas": per_kelas}, indent=2, ensure_ascii=False), encoding="utf-8")
