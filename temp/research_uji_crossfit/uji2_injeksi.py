"""Uji 2: ketahanan terhadap noise label yang disuntikkan, pada dua pandangan ViT dan CNN.

Inilah ukuran tidak melingkar: citra yang dirusak diketahui, jadi pemulihan bisa dihitung tanpa temuan.

Untuk tiap (jenis, laju, ulangan, fold):
- Hold-out 20% group tidak disentuh, dinilai terhadap label asli EXP-009.
- Label bagian latih dirusak. Metode hanya melihat label rusak.
- acc_holdout: kesepakatan prediksi dengan label asli pada hold-out.
- pulih, rusak: dari citra yang label-nya diubah metode, berapa kembali ke asli, dan berapa label bersih yang dirusak.

Jenis noise: bersih (laju 0), acak, bingung (kelas kedua suara), cluster (satu cluster KMeans di ruang ViT berpindah kelas).
Noise cluster sengaja dibuat di ruang ViT karena label asli juga lahir dari KMeans di ruang ViT.

    python temp/research_uji_crossfit/uji2_injeksi.py
"""
import sys

from common import *  # noqa: F401,F403
from common import (N_KELAS, OUT, Waktu, bulat, lipat, lr_fit_predict, lr_lunak, muat, muat_cnn, satu_panas, simpan_json)
from sim_label import baseline_sgd, co_teaching, injeksi, oof_dua

import numpy as np
import pandas as pd

ULANG = 2
SEL = [("bersih", 0.0), ("acak", .05), ("acak", .15), ("bingung", .05), ("bingung", .15), ("cluster", .05), ("cluster", .15)]
GRID = (.6, .7, .8, .9)
log = Waktu()
d = muat()
Ev, y0, V, grp = d["E"], d["y"], d["V"], d["grp"]
Ec, _ = muat_cnn(d["ids"], d["ids_te"])
il = np.where(y0 >= 0)[0]
Ev, Ec, V, y0, g0 = Ev[il], Ec[il], V[il], y0[il], grp[il]


def perbarui(pa, pb, y, ambang):
    a, b = pa.argmax(1), pb.argmax(1)
    return (a == b) & (a != y) & (pa.max(1) >= ambang) & (pb.max(1) >= ambang)


def akurasi(p, q):
    return float((p.argmax(1) == q).mean())


baris, grid = [], []
mulai = 0
if (OUT / "uji2_antara.csv").exists() and "--lanjut" in sys.argv:
    baris = pd.read_csv(OUT / "uji2_antara.csv").to_dict("records")

for s_i, (jenis, laju) in enumerate(SEL):
    for ul in range(ULANG):
        for lf, (tr, va) in enumerate(lipat(y0, g0, 5, seed=100 + ul)):
            seed = 1000 * s_i + 10 * ul + lf
            if baris and any(b["seed"] == seed for b in baris):
                continue
            Yb = y0[tr]
            Yk = Yb.copy() if laju == 0 else injeksi(Yb, V[tr], Ev[tr], jenis, laju, seed)
            kotor = Yk != Yb
            Ea, Eb, Ga = Ev[tr], Ec[tr], g0[tr]
            Qa, Qb = Ev[va], Ec[va]
            yv = y0[va]
            hasil = {}

            pa, pb = lr_fit_predict(Ea, Yk, [Qa])[0], lr_fit_predict(Eb, Yk, [Qb])[0]
            hasil["vit_keras"] = pa
            hasil["cnn_keras"] = pb
            hasil["gabung_keras"] = (pa + pb) / 2

            oa, ob = oof_dua(Ea, Eb, Yk, Ga, seed=seed % 97)
            i = np.arange(len(Yk))
            # Bootstrap lunak: target campuran label dan OOF rata-rata dua pandangan.
            T = .5 * satu_panas(Yk) + .5 * (oa + ob) / 2
            ta, tb = lr_lunak(Ea, T, [Qa])[0], lr_lunak(Eb, T, [Qb])[0]
            hasil["soft_oof"] = (ta + tb) / 2

            # Saring: buang yang ditolak dua pandangan.
            jaga = ~((oa[i, Yk] < .1) & (ob[i, Yk] < .1))
            hasil["saring"] = (lr_fit_predict(Ea[jaga], Yk[jaga], [Qa])[0] + lr_fit_predict(Eb[jaga], Yk[jaga], [Qb])[0]) / 2

            # Kesepakatan silang, maksimal 3 ronde.
            cur = Yk.copy()
            for r in range(3):
                ca, cb = (oa, ob) if r == 0 else oof_dua(Ea, Eb, cur, Ga, seed=seed % 97 + r)
                g = perbarui(ca, cb, cur, .8)
                if not g.any():
                    break
                cur = cur.copy()
                cur[g] = ca.argmax(1)[g]
            hasil["silang"] = (lr_fit_predict(Ea, cur, [Qa])[0] + lr_fit_predict(Eb, cur, [Qb])[0]) / 2
            ubah = cur != Yk

            # Satu arah: CNN menolak label yang ditolak atau tidak diputuskan ViT, ViT tidak wajib setuju.
            hanya_cnn = (ob.argmax(1) != Yk) & (ob.max(1) >= .8)
            hasil["cnn_saja_buang"] = (lr_fit_predict(Ea[~hanya_cnn], Yk[~hanya_cnn], [Qa])[0] +
                                       lr_fit_predict(Eb[~hanya_cnn], Yk[~hanya_cnn], [Qb])[0]) / 2

            sa, sb = baseline_sgd(Ea, Eb, Yk, Qa, Qb, seed=seed)
            hasil["sgd_dasar"] = (sa + sb) / 2
            for lupa in (.1, .2):
                ca2, cb2 = co_teaching(Ea, Eb, Yk, Qa, Qb, lupa=lupa, seed=seed)
                hasil[f"coteach_{lupa}"] = (ca2 + cb2) / 2

            for m, p in hasil.items():
                b = {"jenis": jenis, "laju": laju, "ulang": ul, "fold": lf, "seed": seed, "metode": m,
                     "acc_holdout": akurasi(p, yv), "n_kotor": int(kotor.sum())}
                if m == "silang":
                    b.update(diubah=int(ubah.sum()), pulih=int((cur[kotor] == Yb[kotor]).sum()),
                             rusak=int((ubah & ~kotor).sum()))
                baris.append(b)
            for am in GRID:
                g = perbarui(oa, ob, Yk, am)
                grid.append({"jenis": jenis, "laju": laju, "seed": seed, "ambang": am, "diubah": int(g.sum()),
                             "pulih": int((oa.argmax(1)[g] == Yb[g]).sum()), "kena_kotor": int((g & kotor).sum()),
                             "n_kotor": int(kotor.sum())})
        pd.DataFrame(baris).to_csv(OUT / "uji2_antara.csv", index=False)
        log(f"{jenis} {laju} ulang {ul}")

t = pd.DataFrame(baris)
t.to_csv(OUT / "uji2_mentah.csv", index=False)
tg = pd.DataFrame(grid)
tg.to_csv(OUT / "uji2_grid.csv", index=False)
r = t.groupby(["jenis", "laju", "metode"], sort=False).agg(acc=("acc_holdout", "mean"), sd=("acc_holdout", "std")).round(4).reset_index()
print(r.pivot(index=["jenis", "laju"], columns="metode", values="acc").round(4).to_string())
r.to_csv(OUT / "uji2_ringkas.csv", index=False)
s = t[t.metode == "silang"].groupby(["jenis", "laju"], sort=False)[["n_kotor", "diubah", "pulih", "rusak"]].mean().round(1)
s["recall"] = (s.pulih / s.n_kotor).round(3)
s["presisi"] = (s.pulih / s.diubah).round(3)
print(s.to_string())
s.to_csv(OUT / "uji2_pemulihan.csv")
g = tg.groupby(["jenis", "laju", "ambang"], sort=False)[["n_kotor", "diubah", "pulih", "kena_kotor"]].mean()
g["presisi"] = (g.pulih / g.diubah).round(3)
g["recall"] = (g.pulih / g.n_kotor).round(3)
print(g.round(1).to_string())
g.to_csv(OUT / "uji2_grid_ringkas.csv")
simpan_json("uji2_info.json", {"sel": SEL, "ulang": ULANG, "grid": GRID, "n_lab": len(y0), "kelas": N_KELAS, "acc_bulat": bulat(0.5)})
log("selesai")
