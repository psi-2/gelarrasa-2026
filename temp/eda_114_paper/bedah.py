"""Bedah kandidat: lemma, DMS, dan model pada tiap kelompok. Tidak mengubah label. Test hanya dihitung, tanpa ID di layar.

    python temp/eda_114_paper/bedah.py
"""
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
from timm.data import ImageNetInfo

from kosakata_material import bangun

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
E8 = ROOT / "temp" / "exp_008_in22k_map" / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
pd.set_option("display.width", 230, "display.max_columns", 40, "display.max_rows", 200)
d = pd.read_csv(OUT / "dasar.csv", dtype={"image_id": str})
te = pd.read_csv(OUT / "dasar_test.csv", dtype={"image_id": str})
x = pd.read_csv(OUT / "massa.csv", dtype={"image_id": str})
tr, ts = x[x.split == "train"].reset_index(drop=True), x[x.split == "test"].reset_index(drop=True)
lem = [ImageNetInfo("imagenet-22k").index_to_description(i).split(",")[0][:22] for i in range(21841)]
P = {"train": [np.load(E8 / f"p22k_train_{m}.npy", mmap_mode="r") for m in ("convnext_l", "swinv2_l", "beitv2_l")],
     "test": [np.load(E8 / f"p22k_test_{m}.npy", mmap_mode="r") for m in ("convnext_l", "swinv2_l", "beitv2_l")]}


def top(s, ix, mask=None, n=7):
    if len(ix) == 0:
        return ""
    ix = np.sort(np.asarray(ix))
    m = np.mean([np.asarray(p[ix]).mean(0) for p in P[s]], 0)
    if mask is not None:
        m = m * mask
    u = np.argsort(m)[::-1][:n]
    return ", ".join(f"{lem[i]} {m[i]:.3f}" for i in u if m[i] >= .003)


def baris(judul, df, s, cols=("d_Paper", "d_Cardboard", "d_Metal", "d_Glass", "d_Plastic", "d_PlasticJernih", "d_Fabric")):
    print(f"\n-- {judul}: n={len(df)}, keluarga {df.fam.value_counts().to_dict()}, kelas {df.kelas.value_counts().to_dict()}")
    if len(df):
        print("   lemma:", top(s, df.index.values))
        print("   DMS median:", df[list(cols)].median().round(3).to_dict())
    return df


M, _ = bangun()
print("=== foil ===")
a = baris("train foil>=.2", tr[tr.v_foil >= .2], "train")
print("   model v5 pada train foil (out-of-sample hanya bila fold 0 atau kosong):", d.loc[a.index, "m5"].value_counts(dropna=False).to_dict(), "| LR cross-fit:", d.loc[a.index, "pl"].value_counts().to_dict(), "| group unik:", d.loc[a.index, "grp"].nunique())
print("   p_metal LR:", d.loc[a.index, "p_metal"].round(3).tolist())
b = baris("test foil>=.2", ts[ts.v_foil >= .2], "test")
print("   conf model:", te.loc[b.index, "conf"].round(3).tolist(), "p_metal:", te.loc[b.index, "p_metal"].round(3).tolist())

print("\n=== kain pada paper ===")
for s, df in (("train", tr), ("test", ts)):
    c = baris(f"{s} paper kain>=.2", df[(df.kelas == "paper") & (df.v_kain >= .2)], s)
print("   kain textile test:", top("test", ts.index[(ts.kelas == "textile") & (ts.v_kain >= .2)].values))

print("\n=== cardboard lawan paper ===")
baris("train cardboard kertas>=.2", tr[(tr.kelas == "cardboard") & (tr.v_kertas >= .2)], "train")
baris("test cardboard kertas>=.2", ts[(ts.kelas == "cardboard") & (ts.v_kertas >= .2)], "test")
baris("train paper d_Cardboard>=.3", tr[(tr.kelas == "paper") & (tr.d_Cardboard >= .3)], "train")
baris("test paper d_Cardboard>=.3", ts[(ts.kelas == "paper") & (ts.d_Cardboard >= .3)], "test")
print("LR: paper berlabel p_cardboard>=.3:", int(((d.label == "paper") & (d.p_cardboard >= .3)).sum()), "| cardboard berlabel p_paper>=.3:", int(((d.label == "cardboard") & (d.p_paper >= .3)).sum()))

print("\n=== glass lawan plastic ===")
g = baris("train glass botol_plastik>=.2", tr[(tr.kelas == "glass") & (tr.v_botol_plastik >= .2)], "train")
print("   LR:", d.loc[g.index, "pl"].value_counts().to_dict(), "p_plastic:", d.loc[g.index, "p_plastic"].round(3).tolist())
g = baris("test glass botol_plastik>=.2", ts[(ts.kelas == "glass") & (ts.v_botol_plastik >= .2)], "test")
print("   conf:", te.loc[g.index, "conf"].round(3).tolist(), "p_plastic:", te.loc[g.index, "p_plastic"].round(3).tolist())
print("bandingan DMS, median: glass berlabel", tr[tr.kelas == "glass"][["d_Glass", "d_PlasticJernih"]].median().round(3).to_dict(), "| plastic berlabel", tr[tr.kelas == "plastic"][["d_Glass", "d_PlasticJernih"]].median().round(3).to_dict())
for th in (.1, .2, .3):
    a_ = tr[(tr.kelas == "glass") & (tr.d_PlasticJernih >= th) & (tr.d_Glass < .05)]
    b_ = tr[(tr.kelas == "plastic") & (tr.d_Glass >= th) & (tr.d_PlasticJernih < .05)]
    print(f"   DMS bertolak, ambang {th}: glass berlabel plastik-jernih {len(a_)}, plastic berlabel kaca {len(b_)}")

print("\n=== test paper keyakinan rendah (conf<.7) ===")
r = te[(te.pred == "paper") & (te.conf < .7)]
k2 = r[[f"p_{c}" for c in NAMES if c != "paper"]].idxmax(axis=1).str[2:]
print("n", len(r), "kelas kedua:", k2.value_counts().to_dict(), "| keluarga", r.fam.value_counts().to_dict())
print("   massa grup median:", ts.loc[r.index, ["v_foil", "v_kain", "v_kertas", "v_kardus", "v_kemasan", "v_kaca"]].median().round(3).to_dict())

print("\n=== 68 kosong ===")
k = d[d.label.isna()].copy()
print("model v5 prediksi:", k.m5.value_counts().to_dict(), "| median p", round(float(k.m5_p.median()), 3), "| p>=.9:", int((k.m5_p >= .9).sum()), "| p>=.7:", int((k.m5_p >= .7).sum()))
print("LR cross-fit setuju model:", int((k.m5 == k.pl).sum()))
