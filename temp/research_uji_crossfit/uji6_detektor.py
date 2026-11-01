"""Uji 6: co-teaching sebagai detektor label salah, dibanding cross-fit LR.

Pertanyaan: uji 2 menunjukkan kesepakatan silang LR hanya memulihkan sekitar 12% noise cluster, sedangkan co-teaching lupa 0,2 tahan
terhadapnya. Apakah probabilitas OOF co-teaching bisa menemukan citra berlabel salah dengan presisi tinggi?

Skema: seluruh 1.626 citra, noise disuntikkan ke semua label, OOF berkelompok 5 fold. Citra dinilai oleh head yang tidak melatihnya.
Koreksi: dua head sepakat pada kelas lain dengan p >= tau pada keduanya.

Detektor: lr (cross-fit LR, silang), ct10 dan ct20 (co-teaching lupa 0,1 dan 0,2), ct_ens (rata-rata OOF ct20 3 seed).

    python temp/research_uji_crossfit/uji6_detektor.py
"""
from common import *  # noqa: F401,F403
from common import N_KELAS, OUT, Waktu, lipat, lr_fit_predict, muat, muat_cnn
from sim_label import co_teaching, injeksi

import numpy as np
import pandas as pd

SEL = [("bersih", 0.0), ("acak", .05), ("bingung", .05), ("cluster", .05), ("cluster", .15)]
ULANG = 3
TAU = (.5, .7, .8, .9)
log = Waktu()
d = muat()
il = np.where(d["y"] >= 0)[0]
Ev, V, y0, g0 = d["E"][il], d["V"][il], d["y"][il], d["grp"][il]
Ec = muat_cnn(d["ids"], d["ids_te"])[0][il]
n = len(y0)


def oof_ct(Yk, lupa, seed, split):
    pa, pb = np.zeros((n, N_KELAS)), np.zeros((n, N_KELAS))
    for f, (tr, va) in enumerate(split):
        pa[va], pb[va] = co_teaching(Ev[tr], Ec[tr], Yk[tr], Ev[va], Ec[va], lupa=lupa, seed=seed + f)
    return pa, pb


def oof_lr(Yk, split):
    pa, pb = np.zeros((n, N_KELAS)), np.zeros((n, N_KELAS))
    for tr, va in split:
        pa[va] = lr_fit_predict(Ev[tr], Yk[tr], [Ev[va]])[0]
        pb[va] = lr_fit_predict(Ec[tr], Yk[tr], [Ec[va]])[0]
    return pa, pb


baris = []
simpan = {}
for s_i, (jenis, laju) in enumerate(SEL):
    for ul in range(ULANG):
        seed = 5000 + 100 * s_i + ul
        Yk = y0.copy() if laju == 0 else injeksi(y0, V, Ev, jenis, laju, seed)
        kotor = Yk != y0
        split = lipat(Yk, g0, 5, seed=seed)
        det = {"lr": oof_lr(Yk, split), "ct10": oof_ct(Yk, .1, seed, split), "ct20": oof_ct(Yk, .2, seed, split)}
        e = [oof_ct(Yk, .2, seed + 1000 * k, split) for k in (1, 2)]
        det["ct_ens"] = ((det["ct20"][0] + e[0][0] + e[1][0]) / 3, (det["ct20"][1] + e[0][1] + e[1][1]) / 3)
        for m, (pa, pb) in det.items():
            for tau in TAU:
                a, b = pa.argmax(1), pb.argmax(1)
                g = (a == b) & (a != Yk) & (pa.max(1) >= tau) & (pb.max(1) >= tau)
                pulih = int((a[g] == y0[g]).sum())
                baris.append({"jenis": jenis, "laju": laju, "ul": ul, "det": m, "tau": tau, "n_kotor": int(kotor.sum()), "diubah": int(g.sum()), "pulih": pulih,
                              "kena_bersih": int((g & ~kotor).sum())})
        if laju == 0 and ul == 0:
            simpan = {f"{m}_{w}": p for m, ps in det.items() for w, p in zip("ab", ps)}
        log(f"{jenis} {laju} ulang {ul}")
        pd.DataFrame(baris).to_csv(OUT / "uji6_mentah.csv", index=False)

t = pd.DataFrame(baris)
r = t.groupby(["jenis", "laju", "det", "tau"], sort=False)[["n_kotor", "diubah", "pulih", "kena_bersih"]].mean()
r["presisi"] = (r.pulih / r.diubah).round(3)
r["recall"] = (r.pulih / r.n_kotor).round(3)
print(r.round(1).to_string())
r.to_csv(OUT / "uji6_ringkas.csv")
np.savez_compressed(OUT / "uji6_bersih_oof.npz", **simpan)
log("selesai")
