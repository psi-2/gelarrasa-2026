"""EDA 209: distribution shift train vs test (DIAGNOSTIK SAJA, tidak untuk tuning/seleksi/pembobotan).

KS per fitur skalar + adversarial validation (AUC 5 fold) pada tiga set fitur: metadata berkas, fitur visual
(warna/tekstur/latar), dan embedding ViT-L. Fitur sumber berkas (ukuran, qhash) sengaja dipisah karena itu
artefak akuisisi, bukan isi citra.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from eda_common import TEMP, VISUAL, emb, meta, out_dir

OUT = out_dir(__file__)
m = meta().sort_values(["split", "image_id"])
v = pd.read_csv(VISUAL)
w = pd.read_csv(TEMP / "eda_205_warna/output/warna.csv")
lt = pd.read_csv(TEMP / "eda_204_background/output/latar.csv").drop(columns=["path", "tipe"])
d = m[["split", "image_id", "w", "h", "size", "q_ijg", "bpp", "progressive"]].merge(v, on=["split", "image_id"]).merge(w, on=["split", "image_id"], suffixes=("", "_w")).merge(lt, on=["split", "image_id"])
d = d.sort_values(["split", "image_id"], key=lambda s: s.map({"train": 0, "test": 1}) if s.name == "split" else s).reset_index(drop=True)
y = (d.split == "test").astype(int).values
SET = {"metadata_berkas": ["w", "h", "size", "q_ijg", "bpp", "progressive"],
       "visual_isi": [c for c in v.columns if c not in ("split", "image_id", "per_x", "per_y", "per_x_kuat", "per_y_kuat", "blk8")] + list(w.columns[2:]) + list(lt.columns[2:])}
ks = pd.DataFrame([(c, ks_2samp(d.loc[y == 0, c].dropna(), d.loc[y == 1, c].dropna()).statistic) for c in SET["metadata_berkas"] + SET["visual_isi"]], columns=["fitur", "D"]).sort_values("D", ascending=False)
print("KS terbesar:\n", ks.head(12).round(3).to_string(index=False))
skf = StratifiedKFold(5, shuffle=True, random_state=42)
auc = {}
for nama, cols in SET.items():
    X = d[cols].values.astype(float)
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, random_state=42), X, y, cv=skf, method="predict_proba")[:, 1]
    auc[nama] = roc_auc_score(y, p)
tr, te = emb("vit")
E = np.vstack([tr, te])
p = cross_val_predict(LogisticRegression(C=.5, max_iter=2000), E, y, cv=skf, method="predict_proba")[:, 1]
auc["embedding_vitl"] = roc_auc_score(y, p)
print("\nAUC adversarial (0.5 = tak terbedakan):", {k: round(a, 3) for k, a in auc.items()})
pt = p[y == 0]
print(f"porsi train yang 'mirip test' (p>0.5): {np.mean(pt > .5):.3f}; ESS bobot p/(1-p) train: {pt.sum() / (1 - pt).sum() and (np.sum(pt / (1 - pt)) ** 2 / np.sum((pt / (1 - pt)) ** 2)):.0f} dari 1920")
g = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, random_state=42).fit(d[SET["visual_isi"]].values, y)
imp = permutation_importance(g, d[SET["visual_isi"]].values, y, n_repeats=5, random_state=42, scoring="roc_auc")
top = pd.Series(imp.importances_mean, SET["visual_isi"]).sort_values(ascending=False).head(10)
print("\nfitur visual paling membedakan (perm. importance, in-sample):\n", top.round(4).to_string())

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].barh(list(auc), list(auc.values())); ax[0].axvline(.5, c="k", ls="--"); ax[0].set(xlim=(.4, 1), title="AUC adversarial train vs test")
ax[1].barh(ks.head(12).fitur[::-1], ks.head(12).D[::-1]); ax[1].set(title="statistik KS terbesar")
fig.tight_layout(); fig.savefig(OUT / "shift.png", dpi=120)
print("simpan shift.png di", OUT)
