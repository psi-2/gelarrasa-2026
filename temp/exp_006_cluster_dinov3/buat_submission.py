"""Buat draft submission dari model hold-out dan ringkas hasilnya.

    python temp/exp_006_cluster_dinov3/buat_submission.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
RUN = OUT / "holdout"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]

p = np.load(RUN / "test_fold0.npy")
ids = np.load(OUT / "ids_test.npy").astype(str)
pred = np.array(NAMES)[p.argmax(1)]
sub = pd.DataFrame({"image_id": ids, "waste_category": pred})
sub.to_csv(OUT / "draft_submission_dinov3_cluster.csv", index=False)

ss = pd.read_csv(ROOT / "data" / "SampleSubmission_DSC.csv", dtype=str)
print("format draft:", list(sub.columns) == list(ss.columns), (sub["image_id"].values == ss["image_id"].values).all(),
      sub["waste_category"].isin(NAMES).all(), len(sub))

conv = pd.read_csv(ROOT / "temp" / "exp_005_imagenet_map" / "output" / "draft_submission_convnextv2.csv", dtype=str)
hasil = {
    "distribusi_test": pd.Series(pred).value_counts().reindex(NAMES).fillna(0).astype(int).to_dict(),
    "prob_maks_rata": round(float(p.max(1).mean()), 4),
    "porsi_prob_maks_lt_0.5": round(float((p.max(1) < 0.5).mean()), 4),
    "setuju_convnextv2": round(float((pred == conv["waste_category"].values).mean()), 4),
}
ps = np.load(RUN / "sisa_fold0.npy")
hasil["distribusi_sisa_train"] = pd.Series(np.array(NAMES)[ps.argmax(1)]).value_counts().reindex(NAMES).fillna(0).astype(int).to_dict()
print(json.dumps(hasil, indent=2, ensure_ascii=False))
print(json.dumps(json.loads((RUN / "fold0.json").read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
(OUT / "ringkasan.json").write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")
