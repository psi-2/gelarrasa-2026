"""Hipotesis D: k optimal, stabilitas antar seed, shift train-test (adversarial), dan kemurnian.

Keluaran: output/k_stabilitas.csv, output/adv.json

    python temp/eda_108_embedding/struktur.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
E_tr, E_te = np.load(E6 / "emb_train.npy"), np.load(E6 / "emb_test.npy")
E = np.vstack([E_tr, E_te])
nt = len(E_tr)
y = np.r_[np.zeros(nt), np.ones(len(E_te))]
pca = PCA(128, random_state=42)
Z = pca.fit_transform(E)
Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
print("variansi 16/32/64/128:", [round(float(pca.explained_variance_ratio_[:k].sum()), 3) for k in (16, 32, 64, 128)])

rows = []
for k in (8, 12, 16, 24, 32, 48, 64, 96, 128, 192):
    labs = [KMeans(k, n_init=3, random_state=s).fit_predict(Z) for s in range(5)]
    ari = [adjusted_rand_score(labs[i], labs[j]) for i in range(5) for j in range(i + 1, 5)]
    sil = silhouette_score(Z, labs[0], metric="cosine")
    rows.append({"k": k, "silhouette": round(float(sil), 4), "ari_mean": round(float(np.mean(ari)), 3), "ari_min": round(float(np.min(ari)), 3)})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "k_stabilitas.csv", index=False)

# adversarial train vs test
cv = StratifiedKFold(5, shuffle=True, random_state=42)
out = {}
clf = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
p = cross_val_predict(clf, Z, y, cv=cv, method="predict_proba")[:, 1]
out["auc_dinov3_pca128"] = round(float(roc_auc_score(y, p)), 4)

g = pd.read_csv(ROOT / "temp" / "eda_107_visual" / "output" / "gabung.csv", dtype={"image_id": str})
tr = g[g.split == "train"].sort_values("image_id")
te = g[g.split == "test"].sort_values("image_id")
d = pd.concat([tr, te], ignore_index=True)
num = ["size", "mp", "ar", "bpp", "q_mean", "aniso", "hf_aniso", "hf_025", "noise", "blk8", "tepi_std", "putih_baris", "putih_kolom", "hitam_baris", "hitam_kolom"]
X = StandardScaler().fit_transform(d[num].fillna(d[num].median()))
p2 = cross_val_predict(LogisticRegression(C=1, max_iter=2000, class_weight="balanced"), X, y, cv=cv, method="predict_proba")[:, 1]
out["auc_forensik_lr"] = round(float(roc_auc_score(y, p2)), 4)
from sklearn.ensemble import HistGradientBoostingClassifier
p3 = cross_val_predict(HistGradientBoostingClassifier(max_depth=4, learning_rate=.08, max_iter=150, random_state=42, class_weight="balanced"),
                       d[num].values, y, cv=cv, method="predict_proba")[:, 1]
out["auc_forensik_gbm"] = round(float(roc_auc_score(y, p3)), 4)
# bobot importance per citra train pada F_640/F_416: berapa train yang mirip test
w = p[:nt] / (1 - p[:nt])
out["ess_train_bobot_iw"] = round(float(w.sum() ** 2 / (w ** 2).sum()), 1)
vis = [c for c in g.columns if c in ("s_mean", "v_mean", "warna", "putih", "abu", "coklat", "lap_var", "lok_std_mean", "tepi_rapat", "lbp_entropi", "orient_entropi", "kontras")]
print(pd.concat([d.iloc[:nt][vis + num].median().rename("train"), d.iloc[nt:][vis + num].median().rename("test")], axis=1).round(3).to_string())
print(json.dumps(out, indent=2))
(OUT / "adv.json").write_text(json.dumps(out, indent=2))

for fam in ("F_416", "F_640"):
    m = (d.fam == fam).values
    pp = cross_val_predict(LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"), Z[m], y[m], cv=cv, method="predict_proba")[:, 1]
    print(fam, "AUC adversarial dalam keluarga", round(float(roc_auc_score(y[m], pp)), 4), "n", int(m.sum()), "test", int(y[m].sum()))
