"""Tahap 4: profil kosakata paper, cardboard, glass, plastic per keluarga, train lawan test. CPU, 4 thread.

Tidak membuka citra, tidak membaca temuan, test hanya diukur. Tidak mengubah label.
Keluaran: output/dasar.csv, output/dasar_test.csv, output/pl_crossfit.npy, output/profil.txt

    python temp/eda_114_paper/profil.py
"""
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
from timm.data import ImageNetInfo

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
RC = ROOT / "temp" / "research_uji_crossfit" / "output"
E8 = ROOT / "temp" / "exp_008_in22k_map" / "output"
E10 = ROOT / "temp" / "exp_010_dua_backbone" / "output"
E13 = ROOT / "temp" / "eda_113_elektronik" / "output"
EM = ROOT / "temp" / "eda_106_eomt" / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
ID = {c: i for i, c in enumerate(NAMES)}
nt = 1920
pd.set_option("display.width", 230, "display.max_columns", 40, "display.max_rows", 200)

ids = np.load(E6 / "ids_train.npy").astype(str)
ids_te = np.load(E6 / "ids_test.npy").astype(str)
Xtr = np.hstack([normalize(np.load(E6 / "emb_train.npy")), normalize(np.load(RC / "emb_cnn_train.npy"))])
Xte = np.hstack([normalize(np.load(E6 / "emb_test.npy")), normalize(np.load(RC / "emb_cnn_test.npy"))])
b = pd.read_csv(E13 / "dasar.csv", dtype={"image_id": str})
assert (b.image_id.values == ids).all()
v5 = pd.read_csv(E13 / "label_folds_v5.csv", dtype={"image_id": str})
d = b[["image_id", "fam", "grp"]].merge(v5[["image_id", "label", "suara", "fold"]], on="image_id", how="left")
print("v5", len(v5), "kosong", int(d.label.isna().sum()))
ids_sisa = sorted(set(ids) - set(v5.image_id))
for t in ("vit_v5", "cnn_v5"):
    print(t, "test", np.load(E10 / t / "test_fold0.npy").shape, "sisa", np.load(E10 / t / "sisa_fold0.npy").shape, "oof", np.load(E10 / t / "oof_fold0.npy").shape)
ps = (np.load(E10 / "vit_v5" / "sisa_fold0.npy") + np.load(E10 / "cnn_v5" / "sisa_fold0.npy")) / 2
assert len(ps) == len(ids_sisa) == 68
s = pd.DataFrame({"image_id": ids_sisa, "m5": np.array(NAMES)[ps.argmax(1)], "m5_p": ps.max(1).round(4)})
d = d.merge(s, on="image_id", how="left")
oof = (np.load(E10 / "vit_v5" / "oof_fold0.npy") + np.load(E10 / "cnn_v5" / "oof_fold0.npy")) / 2
ia = v5[v5.fold == 0].image_id.values
assert len(ia) == len(oof)
o = pd.DataFrame({"image_id": ia, "m5o": np.array(NAMES)[oof.argmax(1)], "m5o_p": oof.max(1).round(4)})
d = d.merge(o, on="image_id", how="left")
d["m5"] = d.m5.fillna(d.m5o)
d["m5_p"] = d.m5_p.fillna(d.m5o_p)

lab = d.label.notna().values
idx = np.where(lab)[0]
y = d.label[idx].map(ID).values
pl = np.zeros((nt, 8))
for tr_, va_ in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(idx, y, d.grp.values[idx]):
    m = LogisticRegression(C=3, max_iter=2000).fit(Xtr[idx[tr_]], y[tr_])
    pl[idx[va_]] = m.predict_proba(Xtr[idx[va_]])
    pl[~lab] += m.predict_proba(Xtr[~lab]) / 5
np.save(OUT / "pl_crossfit.npy", pl)
d["pl"] = np.array(NAMES)[pl.argmax(1)]
d["pl_p"] = pl.max(1).round(4)
for c in NAMES:
    d["p_" + c] = pl[:, ID[c]].round(4)
d.to_csv(OUT / "dasar.csv", index=False)
print("LR cross-fit setuju v5:", round(float((d.pl == d.label)[lab].mean()), 4))
print(d[lab & (d.pl != d.label)].groupby(["label", "pl"]).size().sort_values(ascending=False).head(14).to_string())

g = (np.load(E10 / "vit_v5" / "test_fold0.npy") + np.load(E10 / "cnn_v5" / "test_fold0.npy")) / 2
m_te = pd.read_csv(ROOT / "temp" / "eda_102_sumber" / "output" / "meta_assign.csv", dtype={"image_id": str})
fam_te = m_te[m_te.split == "test"].set_index("image_id").loc[ids_te].fam.values
te = pd.DataFrame({"image_id": ids_te, "fam": fam_te, "pred": np.array(NAMES)[g.argmax(1)], "conf": g.max(1).round(4)})
for c in NAMES:
    te["p_" + c] = g[:, ID[c]].round(4)
te.to_csv(OUT / "dasar_test.csv", index=False)

print("\n== train v5 per keluarga ==")
t = pd.crosstab(d.fam, d.label.fillna("(kosong)"))
print(t.reindex(columns=NAMES + ["(kosong)"]).fillna(0).astype(int).to_string())
print("\n== test prediksi v5 per keluarga (bukan fit) ==")
t = pd.crosstab(te.fam, te.pred)
print(t.reindex(columns=NAMES).fillna(0).astype(int).assign(n=t.sum(1)).to_string())
print("total prediksi:", te.pred.value_counts().reindex(NAMES).to_dict())
print("keyakinan median per kelas prediksi:", te.groupby("pred").conf.median().round(3).to_dict())
print("citra test conf<.7 per prediksi:", te[te.conf < .7].pred.value_counts().to_dict())

info = ImageNetInfo("imagenet-22k")
lem = [info.index_to_description(i).split(",")[0][:24] for i in range(info.num_classes())]
Ptr = [np.load(E8 / f"p22k_train_{m}.npy", mmap_mode="r") for m in ("convnext_l", "swinv2_l", "beitv2_l")]
Pte = [np.load(E8 / f"p22k_test_{m}.npy", mmap_mode="r") for m in ("convnext_l", "swinv2_l", "beitv2_l")]
dm = json.loads((E8 / "dms_kelas.json").read_text(encoding="utf-8"))
Dtr = .5 * np.load(E8 / "dms_train.npy") + .5 * np.load(E8 / "dms_tengah_train.npy")
Dte = .5 * np.load(E8 / "dms_test.npy") + .5 * np.load(E8 / "dms_tengah_test.npy")
kc = json.loads((EM / "coco_kelas.json").read_text(encoding="utf-8"))
Ctr, Cte = np.load(EM / "coco_train.npy").mean(2), np.load(EM / "coco_test.npy").mean(2)
LATAR = {"wall-other-merged", "floor-other-merged", "table-merged", "floor-wood", "wall-wood", "wall-tile", "wall-stone", "wall-brick", "dirt-merged", "rock-merged", "dining table", "counter", "mirror-stuff",
         "ceiling-merged", "cabinet-merged", "shelf", "person", "pavement-merged", "grass-merged", "sand", "snow", "sky-other-merged", "water-other", "window-other", "door-stuff", "bed", "gravel", "road", "platform"}
bk = np.array([k in LATAR for k in kc])
Ctr[:, bk] = 0
Cte[:, bk] = 0
BM = ("Paint/plaster/enamel", "Clutter", "Wood", "Ceramic", "Stone, polished", "Tile", "Concrete", "Skin/lips")


def atas(v, nama, n, buang=()):
    u = [i for i in np.argsort(v)[::-1] if nama[i] not in buang][:n]
    return ", ".join(f"{nama[i]} {v[i]:.3f}" for i in u if v[i] >= .005)


def kel(judul, itr, ite, f):
    itr, ite = np.asarray(itr, int), np.asarray(ite, int)
    n = len(itr) + len(ite)
    if n < 3:
        return
    c, m = {}, np.zeros(len(lem))
    for a, b_ in zip(Ptr, Pte):
        for p, i in ((a, itr), (b_, ite)):
            if len(i):
                z = np.asarray(p[i])
                m += z.sum(0)
                for t_ in z.argmax(1):
                    c[lem[t_]] = c.get(lem[t_], 0) + 1
    m /= 3 * n
    tx = [f"\n### {judul} (n={n})", "  22k massa: " + atas(m, lem, 9), "  22k top-1: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda z: -z[1])[:9]),
          "  DMS: " + atas(np.r_[Dtr[itr], Dte[ite]].mean(0), dm, 7, BM), "  COCO: " + atas(np.r_[Ctr[itr], Cte[ite]].mean(0), kc, 7)]
    t_ = "\n".join(tx)
    print(t_, flush=True)
    f.write(t_ + "\n")


with open(OUT / "profil.txt", "w", encoding="utf-8") as f:
    for c in ("paper", "cardboard", "glass", "plastic"):
        for fm in ("F_416", "F_640", "F_lain_web", "F_q75_4x3", "F_q74fb", "F_q4885_besar"):
            kel(f"train {c} {fm}", d.index[(d.label == c) & (d.fam == fm)], [], f)
            kel(f"TEST pred {c} {fm}", [], te.index[(te.pred == c) & (te.fam == fm)], f)
    for fm in sorted(d[d.label.isna()].fam.unique()):
        kel(f"train kosong {fm}", d.index[d.label.isna() & (d.fam == fm)], [], f)
