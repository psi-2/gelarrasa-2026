"""Uji 4b: OT tak seimbang yang benar. Menggantikan bagian C pada uji4_isi_ot.py.

Bug versi lama: pembaruan b += f * log(pi / q) adalah iterasi teredam yang selalu konvergen ke keseimbangan penuh, sehingga semua
rho memberi hasil sama. Hasil bagian C lama dibuang.

Pembaruan benar, Sinkhorn tak seimbang dengan penalti KL pada marginal kelas:
    b_c <- f * [ log pi_c - log mean_i( p_ic / sum_k p_ik exp(b_k) ) ],   f = rho / (1 + rho)
f = 1 memberi keseimbangan penuh, f = 0 tanpa koreksi. Marginal train akhir menjadi antara marginal asli dan prior.
Offset b dihitung pada train saja, lalu dibekukan dan diterapkan ke test.

    python temp/research_uji_crossfit/uji4b_ot.py
"""
from common import *  # noqa: F401,F403
from common import N_KELAS, NAMES, OUT, Waktu, bulat, lr_fit_predict, muat, muat_cnn

import numpy as np
import pandas as pd

log = Waktu()
d = muat()
Ev, Ev_te, y = d["E"], d["E_te"], d["y"]
Ec, Ec_te = muat_cnn(d["ids"], d["ids_te"])
il, isisa = np.where(y >= 0)[0], np.where(y < 0)[0]
ft_te = d["te_ft"].argmax(1)
oof = np.load(OUT / "uji3_oof.npz")
pv, pc = np.zeros((1920, N_KELAS)), np.zeros((1920, N_KELAS))
pv[il], pc[il] = oof["vit"].mean(0), oof["cnn"].mean(0)
a = lr_fit_predict(Ev[il], y[il], [Ev[isisa], Ev_te])
c = lr_fit_predict(Ec[il], y[il], [Ec[isisa], Ec_te])
pv[isisa], pc[isisa] = a[0], c[0]
P = (pv + pc) / 2
Pte = (a[1] + c[1]) / 2
pi = np.ones(N_KELAS) / N_KELAS  # prior lunak dari SampleSubmission, bukan dari temuan


def offset(P, rho, it=2000):
    f = 1.0 if np.isinf(rho) else rho / (1 + rho)
    b = np.zeros(N_KELAS)
    for _ in range(it):
        s = (P * np.exp(b)).sum(1, keepdims=True)
        b2 = f * (np.log(pi) - np.log((P / s).mean(0)))
        if np.abs(b2 - b).max() < 1e-10:
            b = b2
            break
        b = b2
    return b


def kena(P, b):
    q = P * np.exp(b)
    return q / q.sum(1, keepdims=True)


X0, X0t = P.argmax(1), Pte.argmax(1)
rows, simpan = [], {}
for rho in (0, .1, .5, 1, 5, np.inf):
    b = offset(P, rho) if rho > 0 else np.zeros(N_KELAS)
    q, qt = kena(P, b), kena(Pte, b)
    u, ut = q.argmax(1), qt.argmax(1)
    gt = ut != X0t
    rows.append({"rho": rho, "b_maks": bulat(np.abs(b - b.mean()).max(), 3), "tr_ganti": int((u != X0).sum()), "ok_label": bulat((u[il] == y[il]).mean()),
                 "te_ganti": int(gt.sum()), "ganti_ke_finetune": int((ut[gt] == ft_te[gt]).sum()), "ganti_dari_finetune": int((X0t[gt] == ft_te[gt]).sum()),
                 "te_setuju_ft": bulat((ut == ft_te).mean()), **{f"tr_{n}": int((u == k).sum()) for k, n in enumerate(NAMES)}, **{f"te_{n}": int((ut == k).sum()) for k, n in enumerate(NAMES)}})
    simpan[str(rho)] = ut
t = pd.DataFrame(rows)
print(t.T.to_string())
t.to_csv(OUT / "uji4b_ot.csv", index=False)
np.savez_compressed(OUT / "uji4b_test.npz", **simpan)
log("selesai")
