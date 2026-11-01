"""Uji 3: cara menggabungkan satu ViT dan satu CNN pada embedding, tanpa temuan.

Cara: rata-rata probabilitas, rata-rata logit, bobot terkalibrasi suhu OOF, concat embedding, stacking meta-LR.
Karena fusi dinilai terhadap label yang lahir di ruang ViT, kesepakatan dengan label condong ke ViT. Ukuran utama:
- ok_label: kesepakatan OOF dengan label.
- ok_pinggir: pada suara < 0,95.
- nll: log-loss OOF, lebih peka dari akurasi saat akurasi jenuh.
- wasit: saat ViT dan CNN berbeda, pihak mana yang didukung pandangan EoMT, yang tidak ikut membentuk label.

Test hanya diprediksi. Stacking dan suhu dihitung dari OOF train.

    python temp/research_uji_crossfit/uji3_fusi.py
"""
from common import *  # noqa: F401,F403
from common import (N_KELAS, NAMES, OUT, Waktu, bulat, fitur_pandangan, lipat, lr_fit_predict, muat, muat_cnn, pandangan_eomt, simpan_json)

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

R = 5
log = Waktu()
d = muat()
Ev, y, grp, suara = d["E"], d["y"], d["grp"], d["suara"]
Ec, Ec_te = muat_cnn(d["ids"], d["ids_te"])
Ev_te = d["E_te"]
Xe, Xe_te = pandangan_eomt()
il = np.where(y >= 0)[0]
yl, gl = y[il], grp[il]
pg = suara[il] < .95
n = len(il)
oof = {k: np.zeros((R, n, N_KELAS)) for k in ("vit", "cnn", "concat", "eomt")}


def nll(p, q):
    return float(-np.log(np.clip(p[np.arange(len(q)), q], 1e-9, None)).mean())


def suhu(p, q):
    lg = np.log(np.clip(p, 1e-9, None))

    def f(t):
        s = lg / t
        s = s - s.max(1, keepdims=True)
        pr = np.exp(s) / np.exp(s).sum(1, keepdims=True)
        return nll(pr, q)
    return float(minimize_scalar(f, bounds=(.1, 10), method="bounded").x)


def skala(p, t):
    s = np.log(np.clip(p, 1e-9, None)) / t
    s = s - s.max(1, keepdims=True)
    return np.exp(s) / np.exp(s).sum(1, keepdims=True)


def geo(a, b):
    g = np.sqrt(np.clip(a, 1e-9, None) * np.clip(b, 1e-9, None))
    return g / g.sum(1, keepdims=True)


Ej = np.c_[Ev, Ec] / np.sqrt(2)
for r in range(R):
    for tr, va in lipat(yl, gl, 5, seed=r):
        itr, iva = il[tr], il[va]
        oof["vit"][r, va] = lr_fit_predict(Ev[itr], y[itr], [Ev[iva]])[0]
        oof["cnn"][r, va] = lr_fit_predict(Ec[itr], y[itr], [Ec[iva]])[0]
        oof["concat"][r, va] = lr_fit_predict(Ej[itr], y[itr], [Ej[iva]])[0]
        Fe, [Fq] = fitur_pandangan(Xe[itr], [Xe[iva]])
        oof["eomt"][r, va] = lr_fit_predict(Fe, y[itr], [Fq], C=.03)[0]
    log(f"ulangan {r + 1}/{R}")
np.savez_compressed(OUT / "uji3_oof.npz", **oof)

baris = []
ta = [suhu(oof["vit"][r], yl) for r in range(R)]
tb = [suhu(oof["cnn"][r], yl) for r in range(R)]
print("suhu vit", np.round(ta, 3), "cnn", np.round(tb, 3))
for r in range(R):
    a, b = oof["vit"][r], oof["cnn"][r]
    kumpulan = {"vit": a, "cnn": b, "concat": oof["concat"][r], "avg_prob": (a + b) / 2, "avg_logit": geo(a, b),
                "avg_suhu": (skala(a, ta[r]) + skala(b, tb[r])) / 2, "geo_suhu": geo(skala(a, ta[r]), skala(b, tb[r]))}
    # Stacking bersarang: meta dilatih pada empat fold OOF, dinilai pada fold kelima.
    Z = np.c_[np.log(np.clip(a, 1e-9, None)), np.log(np.clip(b, 1e-9, None))]
    st = np.zeros_like(a)
    for tr, va in lipat(yl, gl, 5, seed=50 + r):
        st[va] = LogisticRegression(C=.1, max_iter=2000).fit(Z[tr], yl[tr]).predict_proba(Z[va])
    kumpulan["stacking"] = st
    for m, p in kumpulan.items():
        ok = p.argmax(1) == yl
        baris.append({"metode": m, "r": r, "ok_label": ok.mean(), "ok_pinggir": ok[pg].mean(), "nll": nll(p, yl),
                      "ok_eomt": (p.argmax(1) == oof["eomt"][r].argmax(1)).mean()})
t = pd.DataFrame(baris)
ring = t.groupby("metode", sort=False).agg(ok_label=("ok_label", "mean"), sd=("ok_label", "std"), ok_pinggir=("ok_pinggir", "mean"),
                                           nll=("nll", "mean"), ok_eomt=("ok_eomt", "mean")).round(4)
print(ring.to_string())
ring.to_csv(OUT / "uji3_fusi.csv")

# Wasit: citra di mana ViT dan CNN berbeda pada OOF.
w = []
for r in range(R):
    pa, pb, pe = oof["vit"][r].argmax(1), oof["cnn"][r].argmax(1), oof["eomt"][r].argmax(1)
    b = pa != pb
    w.append({"beda": int(b.sum()), "vit_benar_label": int((pa[b] == yl[b]).sum()), "cnn_benar_label": int((pb[b] == yl[b]).sum()),
              "eomt_dukung_vit": int((pe[b] == pa[b]).sum()), "eomt_dukung_cnn": int((pe[b] == pb[b]).sum())})
tw = pd.DataFrame(w)
print(tw.mean().round(1).to_dict())
tw.to_csv(OUT / "uji3_wasit.csv", index=False)
konf = pd.crosstab(pd.Series(np.array(NAMES)[oof["vit"][0].argmax(1)], name="vit"), pd.Series(np.array(NAMES)[oof["cnn"][0].argmax(1)], name="cnn")).reindex(index=NAMES, columns=NAMES).fillna(0).astype(int)
print(konf.to_string())

# Test hanya diprediksi. Model dilatih pada seluruh 1.626 label.
pv, pc = lr_fit_predict(Ev[il], y[il], [Ev_te])[0], lr_fit_predict(Ec[il], y[il], [Ec_te])[0]
gab = (pv + pc) / 2
ft = d["te_ft"]
info = {"test_setuju_vit_cnn": bulat((pv.argmax(1) == pc.argmax(1)).mean()), "test_beda_vit_cnn": int((pv.argmax(1) != pc.argmax(1)).sum()),
        "test_setuju_gabung_finetune": bulat((gab.argmax(1) == ft.argmax(1)).mean()),
        "test_setuju_vit_finetune": bulat((pv.argmax(1) == ft.argmax(1)).mean()),
        "test_setuju_cnn_finetune": bulat((pc.argmax(1) == ft.argmax(1)).mean()),
        "distribusi_test_gabung": dict(zip(NAMES, np.bincount(gab.argmax(1), minlength=8).tolist())),
        "suhu_vit": bulat(np.mean(ta), 3), "suhu_cnn": bulat(np.mean(tb), 3)}
print(info)
simpan_json("uji3_info.json", info)
np.savez_compressed(OUT / "uji3_test.npz", vit=pv.astype(np.float32), cnn=pc.astype(np.float32))
log("selesai")
