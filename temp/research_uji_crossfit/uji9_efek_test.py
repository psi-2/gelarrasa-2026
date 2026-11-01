"""Uji 9: seberapa jauh tiap varian label menggeser prediksi test, tanpa temuan.

Target 471 dari 480 berarti butuh 13 perbaikan di atas EXP-009. Sebuah varian label hanya bisa menambah bila mengubah prediksi test.
Di sini dihitung batas atas dampaknya: berapa prediksi test berubah, dan arah perubahan menurut wasit 22k + DMS.
Model: LR ViT dan CNN dengan fusi geometrik terkalibrasi suhu dari OOF train. Test hanya diprediksi. Ini proksi head linear, bukan fine-tune.

    python temp/research_uji_crossfit/uji9_efek_test.py
"""
from common import *  # noqa: F401,F403
from common import NAMES, OUT, ROOT, Waktu, bulat, lr_fit_predict, muat, muat_cnn, simpan_json

import hashlib
import json

import numpy as np
import pandas as pd

MARGIN = .5
log = Waktu()
d = muat()
Ev, Evt, y, grp, ids = d["E"], d["E_te"], d["y"], d["grp"], d["ids"]
Ec, Ect = muat_cnn(d["ids"], d["ids_te"])
il, isisa = np.where(y >= 0)[0], np.where(y < 0)[0]
ft = d["te_ft"].argmax(1)

E8 = ROOT / "temp" / "exp_008_in22k_map" / "output"
P112 = ROOT / "temp" / "eda_112_label_v2" / "output"
M22, Md = np.load(P112 / "peta_in22k.npy"), np.load(P112 / "peta_dms.npy")
rek = json.loads((P112 / "peta_beku.json").read_text(encoding="utf-8"))
assert hashlib.sha256(M22.tobytes()).hexdigest() == rek["sha256_peta_in22k"]


def skor(split):
    s = np.mean([np.load(E8 / f"p22k_{split}_{m}.npy") @ M22.T for m in ("convnext_l", "swinv2_l", "beitv2_l")], axis=0)
    dm = .5 * np.load(E8 / f"dms_{split}.npy") + .5 * np.load(E8 / f"dms_tengah_{split}.npy")
    return s.astype(np.float64), (dm @ Md.T).astype(np.float64)


s22, sd = skor("train")
s22t, sdt = skor("test")
p22, pd_ = s22.mean(0) + 1e-9, sd.mean(0)


def lo(a, b):
    r = a / p22
    r = r / r.sum(1, keepdims=True)
    L = np.log(r + 1e-3)
    ada = pd_ > 1e-6
    q = np.divide(b, pd_, out=np.zeros_like(b), where=ada)
    q = q / np.clip(q[:, ada].sum(1, keepdims=True), 1e-9, None)
    L2 = np.zeros(q.shape)
    L2[:, ada] = np.log(q[:, ada] + 1e-3)
    L2[:, ~ada] = L2[:, ada].mean(1, keepdims=True)
    return L + L2


Jt = lo(s22t, sdt)


def fusi(idx, Y):
    pa = lr_fit_predict(Ev[idx], Y, [Evt])[0]
    pb = lr_fit_predict(Ec[idx], Y, [Ect])[0]
    g = (pa ** (1 / .33)) * 0 + np.exp(.5 * (np.log(np.clip(pa, 1e-9, None)) / .33 + np.log(np.clip(pb, 1e-9, None)) / .47))
    return g / g.sum(1, keepdims=True), pa, pb


t = pd.read_csv(OUT / "aturan_semua.csv", dtype={"image_id": str, "label_baru": str}).fillna({"label_baru": ""}).set_index("image_id")
s = np.load(OUT / "uji8_sisa.npz")
pa, pb = s["pa"], s["pb"]
ua = pa.argmax(1)
sama = ua == pb.argmax(1)
Vg = np.load(OUT / "uji5_gabung_A.npy")

Y0 = y.copy()
Y1 = y.copy()
for k, r in t.iterrows():
    i = np.where(ids == k)[0][0]
    if r.aksi == "koreksi":
        Y1[i] = NAMES.index(r.label_baru)
    elif r.aksi == "buang":
        Y1[i] = -1


def tambah(Y, th):
    Z = Y.copy()
    m = sama & (pa.max(1) >= th) & (pb.max(1) >= th)
    Z[isisa[m]] = ua[m]
    return Z


Yg = Y1.copy()
baru = (Vg.max(1) >= .75) & (Y1 < 0) & (y < 0)
Yg[baru] = Vg.argmax(1)[baru]
varian = {"L0_exp009": Y0, "L1_bersih": Y1, "L2_bersih_isi90": tambah(Y1, .9), "L3_bersih_isi95": tambah(Y1, .95), "L4_bersih_isi70": tambah(Y1, .7), "L5_bersih_bag_gabung": Yg}
g0, _, _ = fusi(np.where(Y0 >= 0)[0], Y0[Y0 >= 0])
u0 = g0.argmax(1)
rows = []
pred = {}
for n, Y in varian.items():
    i = np.where(Y >= 0)[0]
    g, _, _ = fusi(i, Y[i])
    u = g.argmax(1)
    pred[n] = u
    b = np.where(u != u0)[0]
    selisih = Jt[b, u[b]] - Jt[b, u0[b]]
    rows.append({"varian": n, "n_latih": len(i), "test_berubah": len(b), "wasit_dukung_baru": int((selisih >= MARGIN).sum()), "wasit_dukung_lama": int((selisih <= -MARGIN).sum()),
                 "ke_finetune": int((u[b] == ft[b]).sum()), "dari_finetune": int((u0[b] == ft[b]).sum()), "setuju_finetune": bulat((u == ft).mean()),
                 **{c: int((u == k).sum()) for k, c in enumerate(NAMES)}})
tb = pd.DataFrame(rows)
print(tb.to_string(index=False))
tb.to_csv(OUT / "uji9_efek.csv", index=False)
np.savez_compressed(OUT / "uji9_pred.npz", **pred)

# Batas bawah kesalahan dari perselisihan: dua model berbeda di D citra berarti err_a + err_b >= D.
o = {}
pv, pc = fusi(np.where(Y0 >= 0)[0], Y0[Y0 >= 0])[1:]
uv, uc = pv.argmax(1), pc.argmax(1)
o["beda_vit_cnn_lr"] = int((uv != uc).sum())
o["beda_finetune_vit_lr"] = int((ft != uv).sum())
o["beda_finetune_cnn_lr"] = int((ft != uc).sum())
o["beda_finetune_fusi_lr"] = int((ft != u0).sum())
tiga_sama = (uv == uc) & (uc == ft)
o["tiga_sepakat"] = int(tiga_sama.sum())
o["tidak_sepakat"] = int((~tiga_sama).sum())
print(o)
simpan_json("uji9_batas.json", o)
log("selesai")
