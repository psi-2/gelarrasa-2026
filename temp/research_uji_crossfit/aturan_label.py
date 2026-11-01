"""Terapkan aturan beku di deteksi.py pada label EXP-009 dan hasilkan daftar image_id untuk dibuang atau dikoreksi.

Masukan aturan hanya embedding DINOv3 ViT-L dan ConvNeXt-L DINOv3, label, dan group. Tidak ada temuan, test, fam, EoMT, 22k, atau DMS.
Bagian audit di akhir tidak mengubah daftar.

Keluaran: output/daftar_label_diragukan.csv, output/aturan_semua.csv, output/label_folds_bersih.csv, output/aturan_audit.json
Label EXP-009 asli tidak ditimpa.

    python temp/research_uji_crossfit/aturan_label.py
"""
from common import *  # noqa: F401,F403
from common import E9, N_KELAS, NAMES, OUT, Waktu, bulat, fitur_pandangan, lipat, lr_fit_predict, muat, muat_cnn, pandangan_eomt, simpan_json
import deteksi
from deteksi import oof_coteach, putuskan

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

log = Waktu()
d = muat()
il = np.where(d["y"] >= 0)[0]
Ev, yl, gl, ids, suara, fam = d["E"][il], d["y"][il], d["grp"][il], d["ids"][il], d["suara"][il], d["fam"][il]
Ec = muat_cnn(d["ids"], d["ids_te"])[0][il]
n = len(yl)
a, b = oof_coteach(Ev, Ec, yl, gl)
np.savez_compressed(OUT / "aturan_oof.npz", a=a, b=b)
log("OOF selesai")
h = putuskan(a, b, yl)
kor, bua, alt = h["koreksi"], h["buang"], h["alt"]
i = np.arange(n)
aksi = np.where(kor, "koreksi", np.where(bua, "buang", "tetap"))
t = pd.DataFrame({"image_id": ids, "label": np.array(NAMES)[yl], "aksi": aksi, "label_baru": np.array(NAMES)[alt], "tolak_vit": h["tolak_v"].round(3),
                  "tolak_cnn": h["tolak_c"].round(3), "p_label_vit": h["pa"][i, yl].round(4), "p_label_cnn": h["pb"][i, yl].round(4), "p_alt_vit": h["pa"][i, alt].round(4),
                  "p_alt_cnn": h["pb"][i, alt].round(4), "suara": suara.round(4), "group": gl, "fam": fam})
t.loc[t.aksi != "koreksi", "label_baru"] = ""
t.to_csv(OUT / "aturan_semua.csv", index=False)
daftar = t[t.aksi != "tetap"].sort_values(["aksi", "image_id"])
daftar.to_csv(OUT / "daftar_label_diragukan.csv", index=False)
print(t.aksi.value_counts().to_dict())
print(pd.crosstab(t.label, t.aksi).reindex(NAMES).fillna(0).astype(int).to_string())
print(daftar.groupby(["aksi", "label", "label_baru"]).size().to_string())

# Label usulan. fold dan group tidak berubah, berkas asli EXP-009 tidak ditimpa.
lab = pd.read_csv(E9 / "label_folds.csv", dtype={"image_id": str})
p = t.set_index("image_id")
ak = p.loc[lab.image_id, "aksi"].values
baru = lab.copy()
baru.loc[ak == "koreksi", "label"] = p.loc[lab.image_id[ak == "koreksi"], "label_baru"].values
baru = baru[ak != "buang"].reset_index(drop=True)
assert not (baru.groupby("group").fold.nunique() > 1).any()
baru.to_csv(OUT / "label_folds_bersih.csv", index=False)
print("label bersih:", len(baru), baru.label.value_counts().reindex(NAMES).to_dict())

# --- AUDIT, tidak mengubah daftar ---
aud = {}
for nm, m in (("koreksi", kor), ("buang", bua)):
    aud[f"{nm}_n"] = int(m.sum())

# Kestabilan: dua paruh model OOF memberi keputusan yang sama?
R_S = len(a)
h1 = putuskan(a[: R_S // 2], b[: R_S // 2], yl)
h2 = putuskan(a[R_S // 2:], b[R_S // 2:], yl)
s1, s2 = h1["koreksi"] | h1["buang"], h2["koreksi"] | h2["buang"]
aud["paruh_1_n"], aud["paruh_2_n"], aud["paruh_iris"] = int(s1.sum()), int(s2.sum()), int((s1 & s2).sum())
aud["paruh_jaccard"] = bulat((s1 & s2).sum() / max(1, (s1 | s2).sum()))

# EoMT: pandangan yang tidak ikut membentuk label t1 dan tidak ikut aturan.
Xe = pandangan_eomt()[0][il]
pe = np.zeros((n, N_KELAS))
for tr, va in lipat(yl, gl, 5, seed=7):
    Fe, [Fq] = fitur_pandangan(Xe[tr], [Xe[va]])
    pe[va] = lr_fit_predict(Fe, yl[tr], [Fq], C=.03)[0]
ue = pe.argmax(1)
aud["eomt_setuju_label_semua"] = bulat((ue == yl).mean())
for nm, m in (("koreksi", kor), ("buang", bua)):
    if m.any():
        aud[f"{nm}_eomt_setuju_lama"] = bulat((ue[m] == yl[m]).mean())
        aud[f"{nm}_eomt_setuju_alt"] = bulat((ue[m] == alt[m]).mean())

# Sel keluarga berkas. Hanya audit.
sel = pd.crosstab(fam, yl).reindex(columns=range(N_KELAS), fill_value=0)
porsi = sel.div(sel.sum(1), axis=0)
pl = porsi.values[pd.Categorical(fam, categories=porsi.index).codes, yl]
pb_ = porsi.values[pd.Categorical(fam, categories=porsi.index).codes, alt]
aud["sel_langka_semua"] = bulat((pl < .05).mean())
for nm, m in (("koreksi", kor), ("buang", bua)):
    if m.any():
        aud[f"{nm}_sel_langka"] = bulat((pl[m] < .05).mean())
        aud[f"{nm}_porsi_fam_lama_median"] = bulat(np.median(pl[m]), 3)
if kor.any():
    aud["koreksi_porsi_fam_alt_median"] = bulat(np.median(pb_[kor]), 3)
    aud["koreksi_alt_lebih_umum_di_fam"] = bulat((pb_[kor] > pl[kor]).mean())

# Konsistensi citra mirip.
tg = pd.DataFrame({"g": gl, "y": yl})
maj = tg.groupby("g").y.agg(lambda s: s.value_counts().index[0])
uk = tg.groupby("g").size()
aud["citra_beda_mayoritas_group"] = int((yl != maj.reindex(gl).values).sum())
besar = uk.reindex(gl).values >= 2
if kor.any():
    aud["koreksi_dalam_group"] = int((kor & besar).sum())
    aud["koreksi_ke_mayoritas_group"] = int((alt[kor & besar] == maj.reindex(gl[kor & besar]).values).sum())
print(aud)
aud["sha1_deteksi"] = hashlib.sha1(Path(deteksi.__file__).read_bytes()).hexdigest()
aud["konstanta"] = {k: getattr(deteksi, k) for k in ("LUPA", "R", "S", "N_FOLD", "TOLAK", "YAKIN", "SEED_DASAR")}
simpan_json("aturan_audit.json", aud)
log("selesai")
