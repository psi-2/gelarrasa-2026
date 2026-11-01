"""Label v6: dua rumus otomatis. Tanpa ID citra, tanpa nama per cluster, test dan temuan tidak masuk rumus.

R1 foil. m = median 3 model ImageNet-22k dari massa grup `foil`. Bila m >= AMBANG_FOIL, label menjadi metal, baik citra berlabel maupun kosong.

R2 isi kosong, hanya citra tanpa label yang tidak kena R1. Semua syarat wajib:
  a. ViT v5 dan CNN v5 (keduanya out-of-sample untuk citra kosong) sepakat pada kelas c.
  b. Rata-rata p[c] kedua model >= P_MIN.
  c. LR cross-fit embedding pada label v5 juga memilih c.
  d. Bukti di luar embedding untuk c: massa kosakata 22k >= AMBANG_VOC, atau porsi material DMS >= AMBANG_DMS.
     DMS tidak dipakai untuk electronic.

Tidak ada rumus untuk cardboard lawan paper, glass lawan plastic, dan kain lawan paper, karena sinyalnya tidak bimodal.

suara: nilai lama untuk label tetap, m untuk R1, rata-rata p untuk R2.
fold: citra lama tetap. Citra baru mengikuti fold group-nya, atau fold 1 bila group belum ada.

    python temp/eda_114_paper/label_v6.py --beku
    python temp/eda_114_paper/label_v6.py
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
AMBANG_FOIL = 0.20
P_MIN = 0.80
AMBANG_VOC = 0.20
AMBANG_DMS = 0.30

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
E10 = ROOT / "temp" / "exp_010_dua_backbone" / "output"
E12 = ROOT / "temp" / "eda_112_label_v2" / "output"
E13 = ROOT / "temp" / "eda_113_elektronik" / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
ID = {c: i for i, c in enumerate(NAMES)}
VOC = {"paper": ["v_kertas"], "cardboard": ["v_kardus"], "metal": ["v_kaleng", "v_foil"], "glass": ["v_kaca"], "plastic": ["v_botol_plastik", "v_kemasan"], "textile": ["v_kain"]}
DMS = {"paper": "d_Paper", "cardboard": "d_Cardboard", "metal": "d_Metal", "glass": "d_Glass", "plastic": "d_Plastic", "textile": "d_Fabric"}
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)


def hash_aturan():
    return {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in ("label_v6.py", "kosakata_material.py")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beku", action="store_true")
    a = ap.parse_args()
    rek_p, v6_p = OUT / "aturan_beku.json", OUT / "label_folds_v6.csv"
    if a.beku:
        assert not v6_p.exists(), "v6 sudah ada, pembekuan harus lebih dulu"
        rek = {"waktu": time.strftime("%Y-%m-%d %H:%M:%S"), "hash": hash_aturan(), "AMBANG_FOIL": AMBANG_FOIL, "P_MIN": P_MIN, "AMBANG_VOC": AMBANG_VOC, "AMBANG_DMS": AMBANG_DMS,
               "catatan": "dibekukan sebelum label_folds_v6.csv ada"}
        rek_p.write_text(json.dumps(rek, indent=2), encoding="utf-8")
        print("BEKU", json.dumps(rek))
        return
    rek = json.loads(rek_p.read_text(encoding="utf-8"))
    assert rek["hash"] == hash_aturan(), "kode aturan berubah setelah dibekukan"
    print("hash cocok, beku", rek["waktu"])

    d = pd.read_csv(OUT / "dasar.csv", dtype={"image_id": str})
    x = pd.read_csv(OUT / "massa.csv", dtype={"image_id": str})
    x = x[x.split == "train"].reset_index(drop=True)
    assert (x.image_id.values == d.image_id.values).all()
    v5 = pd.read_csv(E13 / "label_folds_v5.csv", dtype={"image_id": str})
    grp, fam = dict(zip(d.image_id, d.grp)), dict(zip(d.image_id, d.fam))

    # R1
    kena = x.v_foil.values >= AMBANG_FOIL
    r1 = pd.DataFrame({"image_id": d.image_id[kena], "lama": d.label[kena].fillna("(kosong)"), "baru": "metal", "nilai": x.v_foil[kena].values})

    # R2
    kos = d.label.isna().values & ~kena
    ids_sisa = sorted(set(d.image_id) - set(v5.image_id))
    pv = pd.DataFrame(np.load(E10 / "vit_v5" / "sisa_fold0.npy"), index=ids_sisa, columns=NAMES)
    pc = pd.DataFrame(np.load(E10 / "cnn_v5" / "sisa_fold0.npy"), index=ids_sisa, columns=NAMES)
    Sk = {m: np.load(E12 / f"skor_22k_{m}.npy")[:1920] for m in ("convnext_l", "swinv2_l", "beitv2_l")}
    Sm = np.median(np.stack(list(Sk.values()), 2), 2)
    dfood = np.zeros(1920)
    E8 = ROOT / "temp" / "exp_008_in22k_map" / "output"
    dm = json.loads((E8 / "dms_kelas.json").read_text(encoding="utf-8"))
    dfood = (.5 * np.load(E8 / "dms_train.npy") + .5 * np.load(E8 / "dms_tengah_train.npy"))[:, dm.index("Food")]
    baris = []
    for i in np.where(kos)[0]:
        im = d.image_id[i]
        cv, cc = pv.loc[im].idxmax(), pc.loc[im].idxmax()
        p = (pv.loc[im, cv] + pc.loc[im, cv]) / 2
        lr = d.pl[i]
        if c_ok := (cv == cc):
            c = cv
            if c == "electronic":
                voc, mat = Sm[i, ID[c]], 0.0
            elif c == "foodwaste":
                voc, mat = Sm[i, ID[c]], dfood[i]
            else:
                voc, mat = max(x.loc[i, k] for k in VOC[c]), x.loc[i, DMS[c]]
            luar = voc >= AMBANG_VOC or mat >= AMBANG_DMS
        else:
            c, voc, mat, luar = cv, 0.0, 0.0, False
        baris.append({"image_id": im, "fam": d.fam[i], "kelas": c, "sepakat": c_ok, "p": round(float(p), 4), "lr": lr, "voc": round(float(voc), 4), "dms": round(float(mat), 4), "luar": bool(luar),
                      "terima": bool(c_ok and p >= P_MIN and lr == c and luar)})
    r2 = pd.DataFrame(baris)
    r2.to_csv(OUT / "kosong_rincian.csv", index=False)
    print("R1 kena:", len(r1), r1.lama.value_counts().to_dict(), "| keluarga", r1.image_id.map(fam).value_counts().to_dict())
    print("R2 kandidat:", len(r2), "| sepakat", int(r2.sepakat.sum()), "| p>=", P_MIN, int((r2.p >= P_MIN).sum()), "| LR setuju", int((r2.lr == r2.kelas).sum()), "| bukti luar", int(r2.luar.sum()), "| TERIMA", int(r2.terima.sum()))
    t = r2[r2.terima]
    print("R2 diterima per kelas:", t.kelas.value_counts().to_dict(), "| keluarga", t.fam.value_counts().to_dict())
    print("R2 ditolak, kelas model:", r2[~r2.terima].kelas.value_counts().to_dict(), "| keluarga", r2[~r2.terima].fam.value_counts().to_dict())

    v = v5.copy()
    m = v.image_id.isin(r1.image_id)
    nil = dict(zip(r1.image_id, r1.nilai))
    v.loc[m, "label"] = "metal"
    v.loc[m, "suara"] = v.image_id[m].map(nil)
    gf = v.groupby("group").fold.first()
    br = r1[r1.lama == "(kosong)"][["image_id"]].assign(label="metal", suara=lambda z: z.image_id.map(nil))
    br = pd.concat([br, t[["image_id"]].assign(label=t.kelas.values, suara=t.p.values)], ignore_index=True)
    br["group"] = br.image_id.map(grp).astype(float)
    br["fam"] = br.image_id.map(fam)
    br["fold"] = [int(gf.get(g, 1)) for g in br.group]
    v = pd.concat([v, br], ignore_index=True).sort_values("image_id").reset_index(drop=True)[list(v5.columns)]
    assert v.image_id.is_unique and not v.drop(columns=["suara"]).isna().any().any()
    assert not (v.groupby("group").fold.nunique() > 1).any()
    c5 = v.merge(v5[["image_id", "fold"]], on="image_id", suffixes=("", "_l"))
    assert (c5.fold == c5.fold_l).all()
    v.to_csv(v6_p, index=False)

    tb = pd.DataFrame({"v5": v5.label.value_counts(), "v6": v.label.value_counts()}).reindex(NAMES).fillna(0).astype(int)
    tb["selisih"] = tb.v6 - tb.v5
    print(tb.to_string())
    print("total", len(v5), "->", len(v), "| fold 0:", int((v5.fold == 0).sum()), "->", int((v.fold == 0).sum()), "| kosong tersisa", 1920 - len(v))
    print("baru di fold 0:", int(v[v.image_id.isin(br.image_id)].fold.eq(0).sum()), "| R1 di fold 0:", int(v[v.image_id.isin(r1.image_id)].fold.eq(0).sum()))
    r1.assign(fam=r1.image_id.map(fam)).to_csv(OUT / "perubahan_r1.csv", index=False)


if __name__ == "__main__":
    main()
