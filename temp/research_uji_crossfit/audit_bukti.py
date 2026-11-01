"""Audit independen: wasit dari ImageNet-22k (3 model) dan DMS46 material, keduanya tidak ikut membentuk label EXP-009.

Peta 22k dan DMS dibekukan sebelum inferensi oleh eda_112 (hash dicek di sini). Skor dinormalkan prior train dengan tau = 1.
Kolom elektronik DMS abstain. Wasit tidak dipakai melatih apa pun dan tidak mengubah daftar aturan.

Wasit punya akurasi sendiri yang tidak diketahui, jadi yang dilaporkan terutama perbandingan berpasangan: untuk dua kelas calon pada
citra yang sama, kelas mana yang lebih didukung. Citra yang wasitnya tidak yakin (selisih log-odds < MARGIN) dihitung terpisah.

    python temp/research_uji_crossfit/audit_bukti.py
"""
from common import *  # noqa: F401,F403
from common import E9, N_KELAS, NAMES, OUT, ROOT, Waktu, bulat, muat, simpan_json

import hashlib
import json

import numpy as np
import pandas as pd

MARGIN = .5
log = Waktu()
d = muat()
y, grp, ids = d["y"], d["grp"], d["ids"]
E8 = ROOT / "temp" / "exp_008_in22k_map" / "output"
P112 = ROOT / "temp" / "eda_112_label_v2" / "output"
M22, Md = np.load(P112 / "peta_in22k.npy"), np.load(P112 / "peta_dms.npy")
rek = json.loads((P112 / "peta_beku.json").read_text(encoding="utf-8"))
assert hashlib.sha256(M22.tobytes()).hexdigest() == rek["sha256_peta_in22k"]
assert hashlib.sha256(Md.tobytes()).hexdigest() == rek["sha256_peta_dms"]


def skor(split):
    s22 = np.mean([np.load(E8 / f"p22k_{split}_{m}.npy") @ M22.T for m in ("convnext_l", "swinv2_l", "beitv2_l")], axis=0)
    dm = .5 * np.load(E8 / f"dms_{split}.npy") + .5 * np.load(E8 / f"dms_tengah_{split}.npy")
    return s22.astype(np.float64), (dm @ Md.T).astype(np.float64)


s22, sd = skor("train")
s22_te, sd_te = skor("test")
pri22, prid = s22.mean(0) + 1e-9, sd.mean(0)


def logodds(a, b):
    r = a / pri22
    r = r / r.sum(1, keepdims=True)
    L = np.log(r + 1e-3)
    ada = prid > 1e-6
    q = np.divide(b, prid, out=np.zeros_like(b), where=ada)
    q = q / np.clip(q[:, ada].sum(1, keepdims=True), 1e-9, None)
    L2 = np.full(q.shape, np.nan)
    L2[:, ada] = np.log(q[:, ada] + 1e-3)
    L2[:, ~ada] = np.nanmean(L2[:, ada], axis=1, keepdims=True)  # abstain: netral
    return L + L2


J, J_te = logodds(s22, sd), logodds(s22_te, sd_te)
il = np.where(y >= 0)[0]
yl = y[il]
u = J.argmax(1)
aud = {"wasit_setuju_label": bulat((u[il] == yl).mean()), "wasit_per_kelas": {n: bulat((u[il][yl == c] == c).mean(), 3) for c, n in enumerate(NAMES)},
       "wasit_distribusi_train": dict(zip(NAMES, np.bincount(u, minlength=8).tolist()))}
print(aud)


def tanding(idx, c1, c2):
    """Kembalikan jumlah citra yang lebih mendukung c1, c2, dan seri (selisih < MARGIN)."""
    if len(idx) == 0:
        return {"n": 0}
    s = J[idx, c1] - J[idx, c2]
    return {"n": len(idx), "dukung_1": int((s >= MARGIN).sum()), "dukung_2": int((s <= -MARGIN).sum()), "seri": int((np.abs(s) < MARGIN).sum())}


# 1. Perselisihan ViT dan CNN pada OOF rata-rata 5 ulangan.
o = np.load(OUT / "uji3_oof.npz")
pv, pc = o["vit"].mean(0).argmax(1), o["cnn"].mean(0).argmax(1)
b = pv != pc
aud["oof_vit_vs_cnn"] = tanding(il[b], pv[b], pc[b])
la = b & (pv == yl)
aud["oof_vit_ikut_label_vs_cnn"] = tanding(il[la], pv[la], pc[la])
lb = b & (pc == yl)
aud["oof_cnn_ikut_label_vs_vit"] = tanding(il[lb], pc[lb], pv[lb])

# 2. Daftar aturan: label lama lawan alternatif.
t = pd.read_csv(OUT / "aturan_semua.csv", dtype={"image_id": str, "label_baru": str}).set_index("image_id")
pos = {k: i for i, k in enumerate(ids)}
kor = t[t.aksi == "koreksi"]
idx = np.array([pos[k] for k in kor.index], dtype=int)
c_new = np.array([NAMES.index(c) for c in kor.label_baru], dtype=int)
aud["koreksi_baru_vs_lama"] = tanding(idx, c_new, y[idx])
for a in ("buang", "periksa"):
    q = t[t.aksi == a]
    idq = np.array([pos[k] for k in q.index], dtype=int)
    aud[f"{a}_wasit_setuju_label"] = bulat((u[idq] == y[idq]).mean()) if len(idq) else None

# 3. Isi 294: prediksi LR dua pandangan lawan fine-tune EXP-009.
isisa = np.where(y < 0)[0]
ft = d["sisa"].argmax(1)
aud["sisa_finetune_setuju_wasit"] = bulat((u[isisa] == ft).mean())

# 4. Test: hanya prediksi. Mana yang didukung saat ViT LR, CNN LR, dan fine-tune EXP-009 berbeda.
te = np.load(OUT / "uji3_test.npz")
tv, tc, tf = te["vit"].argmax(1), te["cnn"].argmax(1), d["te_ft"].argmax(1)


def tanding_te(c1, c2):
    m = c1 != c2
    s = J_te[np.where(m)[0], c1[m]] - J_te[np.where(m)[0], c2[m]]
    return {"n": int(m.sum()), "dukung_1": int((s >= MARGIN).sum()), "dukung_2": int((s <= -MARGIN).sum()), "seri": int((np.abs(s) < MARGIN).sum())}


aud["test_vit_vs_cnn"] = tanding_te(tv, tc)
aud["test_finetune_vs_cnn"] = tanding_te(tf, tc)
aud["test_finetune_vs_vit"] = tanding_te(tf, tv)
aud["test_finetune_setuju_wasit"] = bulat((J_te.argmax(1) == tf).mean())
print(json.dumps(aud, indent=1, ensure_ascii=False))
simpan_json("audit_bukti.json", aud)
assert N_KELAS == 8 and E9.exists()
log("selesai")
