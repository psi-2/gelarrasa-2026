"""Uji 1: cross-fit berkelompok untuk metode pembentuk prediksi pada 1.626 label EXP-009.

Metode: LR keras (baseline), kNN, label spreading, LR soft label, LR soft + 294 sisa, partial label PRODEN.
Pandangan pembanding: EoMT COCO+ADE, tidak ikut membentuk label t1_in1k.

Karena label dibuat dari embedding yang sama, kesepakatan pada DINOv3 sifatnya melingkar. Yang dibandingkan:
- ok_semua: kesepakatan OOF dengan label, batas atas melingkar.
- ok_pinggir: kesepakatan pada suara < 0,95, wilayah yang benar-benar sulit.
- ok_lintas: kecocokan prediksi OOF dengan prediksi pandangan EoMT pada citra yang sama.

    python temp/research_uji_crossfit/uji1_crossfit.py
"""
from common import *  # noqa: F401,F403
from common import (N_KELAS, NAMES, OUT, Waktu, bulat, fitur_pandangan, graf_knn, knn_prediksi, label_spreading, lipat,
                    lr_fit_predict, lr_lunak, muat, pandangan_eomt, proden, satu_panas, simpan_json)

import numpy as np
import pandas as pd

R = 5
log = Waktu()
d = muat()
E, y, V, grp, suara = d["E"], d["y"], d["V"], d["grp"], d["suara"]
Xe, _ = pandangan_eomt()
idx_lab = np.where(y >= 0)[0]
idx_sisa = np.where(y < 0)[0]
yl = y[idx_lab]
Sn = graf_knn(E)
log(f"label {len(idx_lab)}, sisa {len(idx_sisa)}, pinggir suara<0,95: {(suara[idx_lab] < .95).sum()}, dim EoMT {Xe.shape}")

METODE = ["lr_keras", "knn", "spreading", "lr_soft", "lr_soft_sisa", "proden_sisa"]
oof = {m: np.zeros((R, len(idx_lab), N_KELAS)) for m in METODE + ["eomt"]}

for r in range(R):
    for lf, (tr, va) in enumerate(lipat(yl, grp[idx_lab], 5, seed=r)):
        itr, iva = idx_lab[tr], idx_lab[va]
        Q = E[iva]
        oof["lr_keras"][r, va] = lr_fit_predict(E[itr], y[itr], [Q])[0]
        oof["knn"][r, va] = knn_prediksi(E[itr], satu_panas(y[itr]), Q)
        Y = np.zeros((1920, N_KELAS))
        Y[itr] = satu_panas(y[itr])
        oof["spreading"][r, va] = label_spreading(Sn, Y)[iva]
        oof["lr_soft"][r, va] = lr_lunak(E[itr], V[itr], [Q])[0]
        ii = np.r_[itr, idx_sisa]
        oof["lr_soft_sisa"][r, va] = lr_lunak(E[ii], V[ii], [Q])[0]
        oof["proden_sisa"][r, va] = proden(E[ii], V[ii], [Q])[0]
        Fe, [Fq] = fitur_pandangan(Xe[itr], [Xe[iva]])
        oof["eomt"][r, va] = lr_fit_predict(Fe, y[itr], [Fq], C=.03)[0]
    log(f"ulangan {r + 1}/{R}")
    np.savez_compressed(OUT / "uji1_oof.npz", **oof)

pinggir = suara[idx_lab] < .95
baris, kelas_baris = [], []
for m in METODE + ["eomt"]:
    per = [(oof[m][r].argmax(1) == yl) for r in range(R)]
    seko = [p.mean() for p in per]
    pg = [p[pinggir].mean() for p in per]
    lt = [(oof[m][r].argmax(1) == oof["eomt"][r].argmax(1)).mean() for r in range(R)]
    u = oof[m].argmax(2)
    antar = np.mean([(u[a] == u[b]).mean() for a in range(R) for b in range(a + 1, R)])
    baris.append({"metode": m, "ok_semua": bulat(np.mean(seko)), "sd": bulat(np.std(seko), 5), "ok_pinggir": bulat(np.mean(pg)),
                  "sd_pinggir": bulat(np.std(pg), 5), "ok_lintas": bulat(np.mean(lt)), "stabil_antar_split": bulat(antar)})
    k = {n: bulat(np.mean([(oof[m][r].argmax(1) == yl)[yl == c].mean() for r in range(R)]), 3) for c, n in enumerate(NAMES)}
    kelas_baris.append({"metode": m, **k})
t = pd.DataFrame(baris)
tk = pd.DataFrame(kelas_baris)
print(t.to_string(index=False))
print(tk.to_string(index=False))
t.to_csv(OUT / "uji1_ringkas.csv", index=False)
tk.to_csv(OUT / "uji1_per_kelas.csv", index=False)

# Uji pasangan terhadap baseline per citra, rata-rata atas ulangan, pada pinggir.
base = (oof["lr_keras"].argmax(2) == yl).mean(0)
rng = np.random.default_rng(0)
gl = grp[idx_lab]
ug = np.unique(gl)
pos = {g: np.where(gl == g)[0] for g in ug}
uji = {}
for m in METODE[1:]:
    s = (oof[m].argmax(2) == yl).mean(0) - base
    bs_all, bs_pg = [], []
    for _ in range(2000):
        pilih = np.concatenate([pos[g] for g in rng.choice(ug, len(ug))])
        bs_all.append(s[pilih].mean())
        ip = pilih[pinggir[pilih]]
        bs_pg.append(s[ip].mean())
    uji[m] = {"selisih_semua": bulat(s.mean(), 5), "ci_semua": [bulat(np.quantile(bs_all, .025), 5), bulat(np.quantile(bs_all, .975), 5)],
              "selisih_pinggir": bulat(s[pinggir].mean(), 5), "ci_pinggir": [bulat(np.quantile(bs_pg, .025), 5), bulat(np.quantile(bs_pg, .975), 5)]}
    print(m, uji[m])
simpan_json("uji1_selisih.json", uji)
log("selesai")
