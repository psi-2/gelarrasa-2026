"""Uji 8: mengisi 294 sisa dengan head co-teaching, pembanding LR dua pandangan, plus wasit pinggir.

Bagian A: latih pada label yakin (suara >= 0,95) saja, prediksi 381 pinggir sebagai perkiraan presisi di wilayah sulit.
Bagian B: latih pada semua label, prediksi 294 sisa, ukur stabilitas antar seed dan jumlah terisi pada ambang.
Test tidak dipakai fit, hanya diprediksi untuk mengukur stabilitas.

    python temp/research_uji_crossfit/uji8_isi_coteach.py
"""
from common import *  # noqa: F401,F403
from common import NAMES, OUT, Waktu, bulat, lr_fit_predict, muat, muat_cnn, simpan_json
from deteksi import LUPA
from sim_label import co_teaching

import numpy as np
import pandas as pd

log = Waktu()
d = muat()
Ev, Evt, y, suara = d["E"], d["E_te"], d["y"], d["suara"]
Ec, Ect = muat_cnn(d["ids"], d["ids_te"])
il, isisa = np.where(y >= 0)[0], np.where(y < 0)[0]
yakin, pinggir = il[suara[il] >= .95], il[suara[il] < .95]
SEED = (11, 22, 33, 44, 55)


def ct(tr, Qa, Qb, lupa):
    pa, pb = [], []
    for s in SEED:
        a, b = co_teaching(Ev[tr], Ec[tr], y[tr], Qa, Qb, lupa=lupa, seed=s)
        pa.append(a)
        pb.append(b)
    return np.array(pa), np.array(pb)


rows = []
for lupa in (.05, LUPA):
    A, B = ct(yakin, Ev[pinggir], Ec[pinggir], lupa)
    pa, pb = A.mean(0), B.mean(0)
    sama = pa.argmax(1) == pb.argmax(1)
    u = pa.argmax(1)
    for th in (0, .5, .7, .8, .9, .95):
        m = sama & (pa.max(1) >= th) & (pb.max(1) >= th)
        rows.append({"lupa": lupa, "ambang": th, "cakup": bulat(m.mean(), 3), "ok_pinggir": bulat((u[m] == y[pinggir][m]).mean() if m.any() else 0)})
    log(f"pinggir lupa {lupa}")
tp = pd.DataFrame(rows)
print(tp.to_string(index=False))
tp.to_csv(OUT / "uji8_pinggir.csv", index=False)

A, B = ct(il, Ev[isisa], Ec[isisa], LUPA)
At, Bt = ct(il, Evt, Ect, LUPA)
pa, pb = A.mean(0), B.mean(0)
u = pa.argmax(1)
ft = d["sisa"].argmax(1)
sama = u == pb.argmax(1)
rows = []
for th in (0, .5, .7, .8, .9, .95):
    m = sama & (pa.max(1) >= th) & (pb.max(1) >= th)
    rows.append({"ambang": th, "isi": int(m.sum()), "setuju_finetune": bulat((u[m] == ft[m]).mean() if m.any() else 0), **{n: int((u[m] == c).sum()) for c, n in enumerate(NAMES)}})
ts = pd.DataFrame(rows)
print(ts.to_string(index=False))
ts.to_csv(OUT / "uji8_sisa.csv", index=False)


def antar(L):
    L = np.array(L)
    return bulat(np.mean([(L[i] == L[j]).mean() for i in range(len(L)) for j in range(i + 1, len(L))]))


stb = {"sisa_antar_seed": antar(((A + B) / 2).argmax(2)), "test_antar_seed": antar(((At + Bt) / 2).argmax(2))}
gab_te = ((At + Bt) / 2).mean(0).argmax(1)
lr = (lr_fit_predict(Ev[il], y[il], [Evt])[0] + lr_fit_predict(Ec[il], y[il], [Ect])[0]) / 2
stb["test_setuju_lr_gabung"] = bulat((gab_te == lr.argmax(1)).mean())
stb["test_setuju_finetune"] = bulat((gab_te == d["te_ft"].argmax(1)).mean())
print(stb)
simpan_json("uji8_stabil.json", stb)
np.savez_compressed(OUT / "uji8_sisa.npz", pa=pa.astype(np.float32), pb=pb.astype(np.float32))
log("selesai")
