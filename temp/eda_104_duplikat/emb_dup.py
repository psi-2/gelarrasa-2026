"""Hipotesis C bagian embedding: tetangga DINOv3 train-test, kurva jumlah citra unik per ambang,
dan kemurnian kategori dalam keluarga cos tinggi.

    python temp/eda_104_duplikat/emb_dup.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)
a = pd.read_csv(ROOT / "temp" / "eda_102_sumber" / "output" / "meta_assign.csv", dtype={"image_id": str})
tr = a[a.split == "train"].sort_values("image_id")
te = a[a.split == "test"].sort_values("image_id")
d = pd.concat([tr, te], ignore_index=True)
E = np.vstack([np.load(E6 / "emb_train.npy"), np.load(E6 / "emb_test.npy")]).astype(np.float32)
assert list(np.load(E6 / "ids_train.npy").astype(str)) == list(tr.image_id)
nt = len(tr)
S = E @ E.T
np.fill_diagonal(S, -1)

print("kuantil cos NN:")
nn_tt = S[nt:, :nt].max(1)  # test ke train
nn_tete = S[nt:, nt:].max(1)
nn_trtr = S[:nt, :nt].max(1)
nn_trte = S[:nt, nt:].max(1)
q = [.05, .25, .5, .75, .95]
print("test->train", np.quantile(nn_tt, q).round(3))
print("test->test ", np.quantile(nn_tete, q).round(3))
print("train->train", np.quantile(nn_trtr, q).round(3))
print("train->test", np.quantile(nn_trte, q).round(3))
for th in (0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.75):
    print(f"cos>={th}: test berteman di train {(nn_tt >= th).mean():.3f}, train berteman di test {(nn_trte >= th).mean():.3f}, test di test {(nn_tete >= th).mean():.3f}, train di train {(nn_trtr >= th).mean():.3f}")

print("\nkomponen per ambang: citra unik")
for th in (0.99, 0.97, 0.95, 0.93, 0.9, 0.85, 0.8):
    b, k = np.where(S >= th)
    n, lab = connected_components(csr_matrix((np.ones(len(b)), (b, k)), shape=(len(E), len(E))), directed=False)
    sz = pd.Series(lab).value_counts()
    cross = pd.DataFrame({"l": lab, "s": d.split}).groupby("l").s.nunique()
    print(th, "komponen", n, "terbesar", int(sz.max()), "komponen silang train-test", int((cross == 2).sum()),
          "citra di dalamnya", int(pd.Series(lab).isin(cross[cross == 2].index).sum()))

# per keluarga sumber: test -> NN train
d["nn_tr"] = np.r_[nn_trtr, nn_tt]
print("\ncos NN ke train per keluarga, test:")
print(d[d.split == "test"].groupby("fam").nn_tr.describe().round(3).to_string())
print("\nkeluarga sumber sama pada NN test->train:")
nn = S[nt:, :nt].argmax(1)
print("sama fam:", float((tr.fam.values[nn] == te.fam.values).mean()), " sama nama:", float((tr.nama.values[nn] == te.nama.values).mean()))
print(pd.crosstab(te.fam.values, tr.fam.values[nn]).to_string())

# kemurnian: per komponen cos>=0.9, sebaran nama cluster ConvNeXt
p = pd.read_csv(ROOT / "temp" / "exp_005_imagenet_map" / "output" / "prob_train.csv", dtype={"image_id": str})
p2 = pd.read_csv(ROOT / "temp" / "exp_005_imagenet_map" / "output" / "prob_test.csv", dtype={"image_id": str})
pp = pd.concat([p, p2], ignore_index=True)
pp = pd.concat([pp.iloc[:nt].set_index("image_id").loc[tr.image_id].reset_index(), pp.iloc[nt:].set_index("image_id").loc[te.image_id].reset_index()], ignore_index=True)
d["pred"] = pp["pred"].values
for th in (0.95, 0.9, 0.85):
    b, k = np.where(S >= th)
    n, lab = connected_components(csr_matrix((np.ones(len(b)), (b, k)), shape=(len(E), len(E))), directed=False)
    x = pd.DataFrame({"l": lab, "pred": d.pred, "nama": d.nama})
    g = x.groupby("l")
    s = g.size()
    big = s[s >= 3].index
    pur = g.pred.agg(lambda v: v.value_counts(normalize=True).iloc[0])[big]
    print(th, "komponen>=3:", len(big), "kemurnian pred ConvNeXt rerata", round(float(pur.mean()), 3),
          "antar komponen pred dominan beda dengan nama", round(float((g.pred.agg(lambda v: v.mode()[0])[big] != g.nama.agg(lambda v: v.mode()[0])[big]).mean()), 3))
