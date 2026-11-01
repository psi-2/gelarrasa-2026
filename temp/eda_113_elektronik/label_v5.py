"""Label v5: koreksi perangkat listrik lewat rumus kosakata ImageNet-22k. Tanpa ID citra, tanpa nama per cluster.

Rumus, train-only:
  Untuk tiap model (ConvNeXt-L, SwinV2-L, BEiTv2-L) dan tiap grup kosakata G di {sel, lampu, papan},
  m_G(citra) = jumlah softmax kelas di G. massa_G = median m_G pada 3 model.
  D = maks(massa_sel, massa_lampu, massa_papan).
  Bila D >= AMBANG: label menjadi electronic, baik citra berlabel maupun kosong.
Grup `kabel` tidak dipakai karena regex-nya berisik.
AMBANG dari celah sebaran massa train. Test, temuan, keluarga, dan ID tidak masuk rumus.

suara: nilai lama untuk label tetap, D untuk label yang berubah atau baru.
fold: citra lama tetap. Citra baru mengikuti fold group-nya, atau fold 1 bila group belum ada.
Dua baris v4 tanpa group dan fam diisi dari hitungan ulang (group 6, seluruhnya fold 1).

    python temp/eda_113_elektronik/label_v5.py --beku    # catat hash, belum menerapkan
    python temp/eda_113_elektronik/label_v5.py           # terapkan, gagal bila hash berubah
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd

# KONSTANTA BEKU
AMBANG = 0.20
GRUP_PAKAI = ("sel", "lampu", "papan")
KELAS_BARU = "electronic"

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)


def hash_aturan():
    return {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in ("label_v5.py", "kosakata_listrik.py")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beku", action="store_true")
    a = ap.parse_args()
    rek_p = OUT / "aturan_beku.json"
    v5_p = OUT / "label_folds_v5.csv"
    if a.beku:
        assert not v5_p.exists(), "v5 sudah ada, pembekuan harus lebih dulu"
        rek = {"waktu": time.strftime("%Y-%m-%d %H:%M:%S"), "hash": hash_aturan(), "AMBANG": AMBANG, "GRUP_PAKAI": GRUP_PAKAI, "catatan": "dibekukan sebelum label_folds_v5.csv ada"}
        rek_p.write_text(json.dumps(rek, indent=2), encoding="utf-8")
        print("BEKU", json.dumps(rek))
        return
    rek = json.loads(rek_p.read_text(encoding="utf-8"))
    assert rek["hash"] == hash_aturan(), "kode aturan berubah setelah dibekukan"
    print("hash cocok, beku", rek["waktu"])

    x = pd.read_csv(OUT / "massa_listrik.csv", dtype={"image_id": str})
    tr = x[x.split == "train"].reset_index(drop=True)
    D = tr[[g + "_min" for g in GRUP_PAKAI]].max(axis=1).values
    v4 = pd.read_csv(ROOT / "temp" / "exp_011_label_v4" / "output" / "label_folds_v4.csv", dtype={"image_id": str})
    d = pd.read_csv(OUT / "dasar.csv", dtype={"image_id": str})
    assert (d.image_id.values == tr.image_id.values).all()
    grp = dict(zip(d.image_id, d.grp))
    fam = dict(zip(d.image_id, d.fam))

    v = v4.copy()
    kosong = v.group.isna() | v.fam.isna()
    v.loc[kosong, "group"] = v.image_id[kosong].map(grp).astype(float)
    v.loc[kosong, "fam"] = v.image_id[kosong].map(fam)
    kena = pd.DataFrame({"image_id": tr.image_id[D >= AMBANG], "D": D[D >= AMBANG].round(4)})
    ada = kena.image_id.isin(v.image_id)
    lama = v.set_index("image_id").label
    ubah = kena[ada & (kena.image_id.map(lama) != KELAS_BARU)]
    tetap = kena[ada & (kena.image_id.map(lama) == KELAS_BARU)]
    baru = kena[~ada]
    print("kena rumus:", len(kena), "| sudah electronic", len(tetap), "| diubah", len(ubah), "| baru", len(baru))
    print("yang diubah, label lama:", ubah.image_id.map(lama).value_counts().to_dict())

    q = dict(zip(ubah.image_id, ubah.D))
    m = v.image_id.isin(q)
    v.loc[m, "label"] = KELAS_BARU
    v.loc[m, "suara"] = v.image_id[m].map(q)
    gf = v.groupby("group").fold.first()
    tb = pd.DataFrame({"image_id": baru.image_id, "label": KELAS_BARU, "suara": baru.D, "group": baru.image_id.map(grp).astype(float), "fam": baru.image_id.map(fam)})
    tb["fold"] = [int(gf.get(g, 1)) for g in tb.group]
    v = pd.concat([v, tb], ignore_index=True).sort_values("image_id").reset_index(drop=True)
    v = v[list(v4.columns)]
    # dua baris v4 tidak punya suara dan tidak ada sumber sah untuk mengisinya, jadi dibiarkan kosong
    assert v.image_id.is_unique and not v.drop(columns=["suara"]).isna().any().any()
    assert int(v.suara.isna().sum()) == int(v4.suara.isna().sum())
    assert not (v.groupby("group").fold.nunique() > 1).any()
    c = v.merge(v4[["image_id", "fold"]], on="image_id", suffixes=("", "_l"))
    assert (c.fold == c.fold_l).all()
    assert set(v4.image_id[v4.fold == 0]) <= set(v.image_id[v.fold == 0])
    v.to_csv(v5_p, index=False)

    t = pd.DataFrame({"v4": v4.label.value_counts(), "v5": v.label.value_counts()}).reindex(NAMES).fillna(0).astype(int)
    t["selisih"] = t.v5 - t.v4
    print(t.to_string())
    print("total", len(v4), "->", len(v), "| fold 0:", int((v4.fold == 0).sum()), "->", int((v.fold == 0).sum()))
    print("perubahan per keluarga:", ubah.image_id.map(fam).value_counts().to_dict(), "| baru per keluarga:", baru.image_id.map(fam).value_counts().to_dict())
    print("berubah di fold 0:", int(v[v.image_id.isin(q)].fold.eq(0).sum()), "| baru di fold 0:", int(v[v.image_id.isin(baru.image_id)].fold.eq(0).sum()))
    pd.concat([ubah.assign(jenis="ubah", label_lama=ubah.image_id.map(lama)), baru.assign(jenis="baru", label_lama="")]).assign(fam=lambda z: z.image_id.map(fam)).to_csv(OUT / "perubahan_v5.csv", index=False)


if __name__ == "__main__":
    main()
