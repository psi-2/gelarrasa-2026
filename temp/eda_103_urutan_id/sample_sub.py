"""Uji struktur SampleSubmission: periodik atau mengandung informasi.

Bila berkorelasi dengan prediksi model, ini kebocoran dan hanya dilaporkan.

    python temp/eda_103_urutan_id/sample_sub.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
s = pd.read_csv(ROOT / "data" / "SampleSubmission_DSC.csv", dtype=str)
print(s.columns.tolist(), len(s))
y = s["waste_category"].values
print(s["waste_category"].value_counts().to_string())
print("20 pertama:", list(y[:20]))
for p in range(1, 41):
    m = float((y[p:] == y[:-p]).mean())
    if m > 0.2 or p == 8:
        print(f"lag {p}: sama {m:.3f}")
print("rata-rata lag 1..20", np.mean([(y[p:] == y[:-p]).mean() for p in range(1, 21)]).round(4), "harapan acak 0.1234")

for nama, f in [("exp005", ROOT / "temp" / "exp_005_imagenet_map" / "output" / "draft_submission_convnextv2.csv"),
                ("exp006", ROOT / "temp" / "exp_006_cluster_dinov3" / "output" / "draft_submission_dinov3_cluster.csv")]:
    d = pd.read_csv(f, dtype=str)
    m = s.merge(d, on="image_id", suffixes=("", "_p"))
    a = float((m["waste_category"] == m["waste_category_p"]).mean())
    print(nama, "kesepakatan dengan sample submission", round(a, 4), "n", len(m))
    print(pd.crosstab(m["waste_category"], m["waste_category_p"]).to_string())
rng = np.random.default_rng(42)
print("baseline shuffle", np.mean([(rng.permutation(y) == d["waste_category"].values).mean() for _ in range(2000)]).round(4))
