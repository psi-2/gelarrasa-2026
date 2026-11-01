"""Uji 4: mengisi 294 citra tanpa label, dan optimal transport tak seimbang.

Bagian A, risk-coverage. Sembunyikan 381 citra pinggir (suara < 0,95), latih pada label yakin saja, lalu prediksi pinggir dan
294 sisa. Pinggir memberi perkiraan presisi di wilayah sulit, karena 294 tidak punya label. Label pinggir sendiri melingkar,
jadi ini perkiraan, bukan akurasi.

Bagian B, stabilitas. 10 bootstrap group pada citra berlabel, ukur kecocokan prediksi 294 antar bootstrap.

Bagian C, OT tak seimbang. Offset kelas b dicari pada train: q_i proporsional p_i exp(b), b ditarik agar marginal train mendekati
prior seragam dengan kekuatan rho/(1+rho). Offset beku itu baru diterapkan ke test, jadi test tidak dipakai fit.

    python temp/research_uji_crossfit/uji4_isi_ot.py
"""
from common import *  # noqa: F401,F403
from common import (N_KELAS, NAMES, OUT, Waktu, bulat, fitur_pandangan, graf_knn, label_spreading, lr_fit_predict, lr_lunak, muat, muat_cnn,
                    pandangan_eomt, proden, satu_panas, simpan_json)

import numpy as np
import pandas as pd

log = Waktu()
d = muat()
Ev, Ev_te, y, V, grp, suara = d["E"], d["E_te"], d["y"], d["V"], d["grp"], d["suara"]
Ec, Ec_te = muat_cnn(d["ids"], d["ids_te"])
Xe, _ = pandangan_eomt()
il, isisa = np.where(y >= 0)[0], np.where(y < 0)[0]
ft_sisa = d["sisa"].argmax(1)
ft_te = d["te_ft"].argmax(1)

# Pembanding EoMT pada 294: dilatih pada semua label.
Fe, [Fs] = fitur_pandangan(Xe[il], [Xe[isisa]])
eomt_sisa = lr_fit_predict(Fe, y[il], [Fs], C=.03)[0].argmax(1)

# --- A: risk-coverage ---
yakin = il[suara[il] >= .95]
pinggir = il[suara[il] < .95]
tq = np.r_[pinggir, isisa]
pv, pc = lr_fit_predict(Ev[yakin], y[yakin], [Ev[tq]])[0], lr_fit_predict(Ec[yakin], y[yakin], [Ec[tq]])[0]
Y = np.zeros((1920, N_KELAS))
Y[yakin] = satu_panas(y[yakin])
ls = label_spreading(graf_knn(Ev), Y)[tq]
ii = np.r_[yakin, isisa]
pr = proden(Ev[ii], V[ii], [Ev[tq]])[0]
gb = (pv + pc) / 2
nP = len(pinggir)
yp = y[pinggir]
rows = []
for nama, p in (("vit", pv), ("cnn", pc), ("gabung", gb), ("spreading", ls), ("proden", pr)):
    u, c = p.argmax(1), p.max(1)
    for th in (0, .5, .7, .8, .9, .95):
        ma, ms = c[:nP] >= th, c[nP:] >= th
        rows.append({"metode": nama, "ambang": th, "cakup_pinggir": bulat(ma.mean(), 3), "ok_pinggir": bulat((u[:nP][ma] == yp[ma]).mean() if ma.any() else 0),
                     "isi_294": int(ms.sum()), "setuju_finetune": bulat((u[nP:][ms] == ft_sisa[ms]).mean() if ms.any() else 0),
                     "setuju_eomt": bulat((u[nP:][ms] == eomt_sisa[ms]).mean() if ms.any() else 0)})
sepakat = pv.argmax(1) == pc.argmax(1)
for th in (0, .5, .7, .8, .9):
    m = sepakat & (pv.max(1) >= th) & (pc.max(1) >= th)
    ma, ms = m[:nP], m[nP:]
    u = pv.argmax(1)
    rows.append({"metode": "dua_sepakat", "ambang": th, "cakup_pinggir": bulat(ma.mean(), 3), "ok_pinggir": bulat((u[:nP][ma] == yp[ma]).mean() if ma.any() else 0),
                 "isi_294": int(ms.sum()), "setuju_finetune": bulat((u[nP:][ms] == ft_sisa[ms]).mean() if ms.any() else 0),
                 "setuju_eomt": bulat((u[nP:][ms] == eomt_sisa[ms]).mean() if ms.any() else 0)})
t = pd.DataFrame(rows)
print(t.to_string(index=False))
t.to_csv(OUT / "uji4_coverage.csv", index=False)
log("A selesai")

# --- B: stabilitas bootstrap ---
rng = np.random.default_rng(0)
ug = np.unique(grp[il])
hasil = {k: [] for k in ("vit", "cnn", "gabung", "lr_soft", "proden", "spreading")}
hasil_te = {k: [] for k in ("vit", "cnn", "gabung")}
for b in range(10):
    pilih = rng.choice(ug, int(.8 * len(ug)), replace=False)
    it = il[np.isin(grp[il], pilih)]
    a = lr_fit_predict(Ev[it], y[it], [Ev[isisa], Ev_te])
    c = lr_fit_predict(Ec[it], y[it], [Ec[isisa], Ec_te])
    hasil["vit"].append(a[0].argmax(1))
    hasil["cnn"].append(c[0].argmax(1))
    hasil["gabung"].append(((a[0] + c[0]) / 2).argmax(1))
    hasil_te["vit"].append(a[1].argmax(1))
    hasil_te["cnn"].append(c[1].argmax(1))
    hasil_te["gabung"].append(((a[1] + c[1]) / 2).argmax(1))
    hasil["lr_soft"].append(lr_lunak(Ev[it], V[it], [Ev[isisa]])[0].argmax(1))
    i2 = np.r_[it, isisa]
    hasil["proden"].append(proden(Ev[i2], V[i2], [Ev[isisa]])[0].argmax(1))
    Y = np.zeros((1920, N_KELAS))
    Y[it] = satu_panas(y[it])
    hasil["spreading"].append(label_spreading(graf_knn(Ev), Y)[isisa].argmax(1))
    log(f"bootstrap {b + 1}/10")


def antar(L):
    L = np.array(L)
    return bulat(np.mean([(L[i] == L[j]).mean() for i in range(len(L)) for j in range(i + 1, len(L))]))


stb = {"sisa_294": {k: antar(v) for k, v in hasil.items()}, "test_480": {k: antar(v) for k, v in hasil_te.items()}}
print(stb)
simpan_json("uji4_stabil.json", stb)

# --- C: OT tak seimbang ---
pv_all = np.zeros((1920, N_KELAS))
pc_all = np.zeros((1920, N_KELAS))
oof = np.load(OUT / "uji3_oof.npz")
pv_all[il], pc_all[il] = oof["vit"].mean(0), oof["cnn"].mean(0)
a = lr_fit_predict(Ev[il], y[il], [Ev[isisa], Ev_te])
c = lr_fit_predict(Ec[il], y[il], [Ec[isisa], Ec_te])
pv_all[isisa], pc_all[isisa] = a[0], c[0]
P = (pv_all + pc_all) / 2
Pte = (a[1] + c[1]) / 2
pi = np.ones(N_KELAS) / N_KELAS  # prior lunak dari SampleSubmission 60 per kelas, bukan dari temuan


def offset(P, rho, it=200):
    b = np.zeros(N_KELAS)
    f = 1.0 if np.isinf(rho) else rho / (1 + rho)
    for _ in range(it):
        q = P * np.exp(b)
        q = q / q.sum(1, keepdims=True)
        b = b + f * np.log(pi / q.mean(0))
        b = b - b.mean()
    return b


def kena(P, b):
    q = P * np.exp(b)
    return q / q.sum(1, keepdims=True)


Xall = P.argmax(1)
rows = []
for rho in (0, .1, .5, 1, 5, np.inf):
    b = offset(P, rho) if rho > 0 else np.zeros(N_KELAS)
    q, qt = kena(P, b), kena(Pte, b)
    u, ut = q.argmax(1), qt.argmax(1)
    ganti = u != Xall
    gt = ut != Pte.argmax(1)
    ls_ok = (u[il] == y[il]).mean()
    rows.append({"rho": rho, "b_maks": bulat(np.abs(b).max(), 3), "train_ganti": int(ganti.sum()), "ok_label": bulat(ls_ok),
                 "ganti_setuju_finetune": bulat((ut[gt] == ft_te[gt]).mean() if gt.any() else 0), "test_ganti": int(gt.sum()),
                 "test_setuju_finetune": bulat((ut == ft_te).mean()), **{f"tr_{n}": int((u == c2).sum()) for c2, n in enumerate(NAMES)},
                 **{f"te_{n}": int((ut == c2).sum()) for c2, n in enumerate(NAMES)}})
t = pd.DataFrame(rows)
print(t.T.to_string())
t.to_csv(OUT / "uji4_ot.csv", index=False)
log("selesai")
