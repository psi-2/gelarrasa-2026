"""Langkah 5: label train v2 dengan aturan otomatis yang dibekukan di konstanta bawah.

Annotator (train-only):
  citra: c_in1k, c_in22k, c_dms (argmax per citra setelah normalisasi prior)
  cluster: b_in22k, b_dms (nama cluster bagging)
  embedding: e_lr (LR DINOv3 cross-fit 5 fold ber-group pada label lama)
b_in1k tidak dipakai sebagai annotator karena identik dengan pembentuk label lama, jadi label lama tidak bisa mengunci diri.
Prediksi EXP-009 dan semua data test tidak masuk fit. EXP-009 hanya pemeriksaan sesudahnya.

Dawid-Skene: inisialisasi mayoritas tertimbang, tanpa memihak label lama. Annotator dalam satu kanal diberi bobot 0,5.
DMS netral untuk electronic: baris electronic pada matriks DMS diganti distribusi marjinal, jadi DMS tidak menjadi bukti electronic.

Aturan keputusan (kanal: in1k, in22k, dms; kanal mendukung kelas c bila c_ atau b_ kanal itu sama dengan c):
  Koreksi label L ke C: C != L, q[C] >= Q_MIN, dukungan(C) >= DUKUNG_MIN, dukungan(C) > dukungan(L).
  Bila electronic ada pada {L, C}: dukungan dihitung pada in1k dan in22k saja, dan C wajib didukung keduanya.
  Tambah label: q[C] >= Q_MIN, dukungan(C) >= DUKUNG_MIN (aturan electronic sama), dan e_lr == C.

fold: citra lama tetap. Citra baru mengikuti fold group-nya bila group sudah ada, selain itu fold 1.

    python temp/eda_112_label_v2/label_v2.py
"""
import hashlib
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

# KONSTANTA BEKU
Q_MIN = 0.90
DUKUNG_MIN = 2
ITER_DS = 5
BOBOT = {"c_in1k": .5, "c_in22k": .5, "c_dms": .5, "b_in22k": .5, "b_dms": .5, "e_lr": 1.0}
SEED = 42

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
E9 = ROOT / "temp" / "exp_009_t1_dinov3" / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
ID = {c: i for i, c in enumerate(NAMES)}
EL = ID["electronic"]
nt = 1920
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)

ids = np.load(E6 / "ids_train.npy").astype(str)
lama = pd.read_csv(E9 / "label_folds.csv", dtype={"image_id": str})
assert lama.image_id.is_unique
d = pd.read_csv(OUT / "tabel_train.csv", dtype={"image_id": str})
assert (d.image_id.values == ids).all()
E_tr, E_te = np.load(E6 / "emb_train.npy"), np.load(E6 / "emb_test.npy")

# group identik dengan label_final.py
S = E_tr @ E_tr.T
np.fill_diagonal(S, 0)
a, b = np.where(S >= .9)
ph = pd.read_csv(ROOT / "temp" / "eda_104_duplikat" / "output" / "pasangan_hash.csv")
ph = ph[(ph.sa == "train") & (ph.sb == "train") & (ph.ham <= 4)]
ii, jj = np.r_[a, ph.a.values], np.r_[b, ph.b.values]
_, grp = connected_components(csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(nt, nt)), directed=False)
d["group"] = grp
cek = d.merge(lama[["image_id", "group"]], on="image_id", suffixes=("", "_l"))
assert (cek.group == cek.group_l).all(), "group tidak sama dengan berkas lama"
print("group identik dengan berkas lama:", len(cek))

# annotator embedding cross-fit
lab = d.label.notna().values
idx = np.where(lab)[0]
y = d.label[idx].map(ID).values
pe = np.zeros((nt, 8))
sg = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
for f, (tr_, va_) in enumerate(sg.split(idx, y, grp[idx])):
    m = LogisticRegression(C=3, max_iter=3000).fit(E_tr[idx[tr_]], y[tr_])
    pe[idx[va_]] = m.predict_proba(E_tr[idx[va_]])
    pe[~lab] += m.predict_proba(E_tr[~lab]) / 5
d["e_lr"] = np.array(NAMES)[pe.argmax(1)]
d["e_lr_p"] = pe.max(1).round(4)
print("LR cross-fit setuju label lama:", round(float((d.e_lr == d.label)[lab].mean()), 4))
m_full = LogisticRegression(C=3, max_iter=3000).fit(E_tr[idx], y)

ANN = list(BOBOT)
R = np.stack([d[k].map(ID).values for k in ANN], 1)
W = np.array([BOBOT[k] for k in ANN])
D_IDX = [i for i, k in enumerate(ANN) if k.endswith("_dms")]


def ds_fit(R):
    q = np.zeros((nt, 8))
    for j, w in enumerate(W):
        q[np.arange(nt), R[:, j]] += w
    q = q / q.sum(1, keepdims=True)
    for it in range(ITER_DS):
        pi = (q.sum(0) + 1) / (q.sum() + 8)
        C = np.zeros((len(ANN), 8, 8))
        for j in range(len(ANN)):
            for r in range(8):
                C[j, :, r] = q[R[:, j] == r].sum(0)
        C = (C + 1) / (C + 1).sum(2, keepdims=True)
        for j in D_IDX:
            nonel = [c for c in range(8) if c != EL]
            C[j, EL, :] = (C[j, nonel, :] * pi[nonel, None]).sum(0) / pi[nonel].sum()
        q = ds_post(R, pi, C)
    return q, pi, C


def ds_post(R, pi, C):
    L = np.tile(np.log(pi), (len(R), 1))
    for j, w in enumerate(W):
        L += w * np.log(C[j, :, R[:, j]])
    q = np.exp(L - L.max(1, keepdims=True))
    return q / q.sum(1, keepdims=True)


q, pi, C = ds_fit(R)
print("prior DS:", dict(zip(NAMES, pi.round(3).tolist())))
print("diagonal matriks per annotator:")
print(pd.DataFrame({k: np.diag(C[j]).round(3) for j, k in enumerate(ANN)}, index=NAMES).to_string())
d["q_max"] = q.max(1).round(4)
d["ds"] = np.array(NAMES)[q.argmax(1)]


def dukungan(r, c, tanpa_dms):
    n = int(r["c_in1k"] == c or r["b_in1k"] == c) + int(r["c_in22k"] == c or r["b_in22k"] == c)
    if not tanpa_dms:
        n += int(r["c_dms"] == c or r["b_dms"] == c)
    return n


keputusan, akhir, alasan = [], [], []
for i, r in enumerate(d.to_dict("records")):
    L = r["label"] if isinstance(r["label"], str) else None
    c = r["ds"]
    qc = q[i, ID[c]]
    el = c == "electronic" or L == "electronic"
    syarat = dukungan(r, c, el)
    cukup = (syarat == 2) if el else (syarat >= DUKUNG_MIN)
    if L is None:
        if qc >= Q_MIN and cukup and r["e_lr"] == c:
            keputusan.append("tambah"); akhir.append(c)
        else:
            keputusan.append("kosong"); akhir.append(None)
    elif c != L and qc >= Q_MIN and cukup and syarat > dukungan(r, L, el):
        keputusan.append("koreksi"); akhir.append(c)
    else:
        keputusan.append("tetap"); akhir.append(L)
d["keputusan"], d["akhir"] = keputusan, akhir
print("\nkeputusan:", d.keputusan.value_counts().to_dict())
k = d[d.keputusan == "koreksi"]
print("koreksi lama -> baru:")
print(pd.crosstab(k.label, k.akhir).to_string() if len(k) else "(tidak ada)")
print("koreksi per keluarga:", k.fam.value_counts().to_dict())
t = d[d.keputusan == "tambah"]
print("tambah per kelas:", t.akhir.value_counts().reindex(NAMES).fillna(0).astype(int).to_dict())
print("tambah per keluarga:", t.fam.value_counts().to_dict())
print("kosong:", int((d.keputusan == "kosong").sum()), "per keluarga", d[d.keputusan == "kosong"].fam.value_counts().to_dict())

# pemeriksaan: EXP-009 bukan masukan
print("\ncek EXP-009 pada tambah: setuju", round(float((t.e9 == t.akhir).mean()), 4) if len(t) else None, "n", len(t))
ko = k[k.e9.notna()]
print("cek EXP-009 pada koreksi di hold-out: n", len(ko), "setuju baru", int((ko.e9 == ko.akhir).sum()), "setuju lama", int((ko.e9 == ko.label).sum()))

# berkas v2
v = d[d.akhir.notna()][["image_id", "akhir", "group", "fam", "keputusan"]].rename(columns={"akhir": "label"}).copy()
v = v.merge(lama[["image_id", "suara", "fold"]], on="image_id", how="left")
qmap = dict(zip(d.image_id, d.q_max))
v["suara"] = [qmap[i] if (k_ != "tetap") else s for i, k_, s in zip(v.image_id, v.keputusan, v.suara)]
gf = lama.groupby("group").fold.first()
v["fold"] = [f if f == f else gf.get(g, 1) for f, g in zip(v.fold, v.group)]
v["fold"] = v.fold.astype(int)
v = v.sort_values("image_id")[["image_id", "label", "suara", "group", "fam", "fold"]]
assert v.image_id.is_unique
assert not (v.groupby("group").fold.nunique() > 1).any()
c = v.merge(lama[["image_id", "fold"]], on="image_id", suffixes=("", "_l"))
assert (c.fold == c.fold_l).all()
assert set(lama.image_id[lama.fold == 0]) == set(v.image_id[v.fold == 0]) & set(lama.image_id)
v.to_csv(OUT / "label_folds_v2.csv", index=False)
d.to_csv(OUT / "rincian_v2.csv", index=False)
print("\nv2:", len(v), "label, fold 0:", int((v.fold == 0).sum()), "fold 1:", int((v.fold == 1).sum()), "| fold 0 baru:", int((v.fold == 0).sum() - (lama.fold == 0).sum()))
print("per kelas v2:", v.label.value_counts().reindex(NAMES).to_dict())

# batas label
sisa = v.merge(d[["image_id", "c_in22k", "b_in22k", "c_dms", "b_dms", "c_in1k", "b_in1k"]], on="image_id")
ql = pd.DataFrame(q, columns=NAMES, index=d.image_id)
sisa["ql"] = [ql.at[i, c] for i, c in zip(sisa.image_id, sisa.label)]
sisa["luar"] = [int(r["c_in22k"] == r["label"] or r["b_in22k"] == r["label"]) + (0 if r["label"] == "electronic" else int(r["c_dms"] == r["label"] or r["b_dms"] == r["label"])) for r in sisa.to_dict("records")]
rows = []
for c_ in NAMES:
    s = sisa[sisa.label == c_]
    rows.append({"kelas": c_, "n": len(s), "harapan_salah": round(float((1 - s.ql).sum()), 1), "q<.5": int((s.ql < .5).sum()), "q<.9": int((s.ql < .9).sum()), "tanpa_dukungan_luar": int((s.luar == 0).sum())})
bt = pd.DataFrame(rows)
bt.loc[len(bt)] = ["TOTAL", bt.n.sum(), round(bt.harapan_salah.sum(), 1), bt["q<.5"].sum(), bt["q<.9"].sum(), bt.tanpa_dukungan_luar.sum()]
print("\nperkiraan label masih berpotensi salah:")
print(bt.to_string(index=False))
bt.to_csv(OUT / "batas_label.csv", index=False)

# test hanya pembandingan, tanpa fit
R_te = {}
for k_ in ("in1k", "in22k", "dms"):
    R_te["c_" + k_] = np.load(OUT / f"citra_{k_}.npy")[nt:]
for k_ in ("in22k", "dms"):
    R_te["b_" + k_] = np.load(OUT / f"vote_{k_}.npy")[nt:].argmax(1)
R_te["e_lr"] = m_full.predict_proba(E_te).argmax(1)
qt = ds_post(np.stack([R_te[k_] for k_ in ANN], 1), pi, C)
pt = np.load(E9 / "holdout" / "test_fold0.npy").argmax(1)
yd = qt.argmax(1)
print("\ntest, pembandingan saja. DS:", dict(zip(NAMES, np.bincount(yd, minlength=8).tolist())), "setuju EXP-009", round(float((yd == pt).mean()), 4))
kuat = (yd != pt) & (qt.max(1) >= Q_MIN)
print("beda dari EXP-009 dengan q>=.9:", int(kuat.sum()))
if kuat.any():
    print(pd.crosstab(pd.Series(np.array(NAMES)[pt[kuat]], name="e9"), pd.Series(np.array(NAMES)[yd[kuat]], name="ds")).to_string())
print("\nsha256 aturan:", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
