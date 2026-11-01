"""EM kosakata train-only. Sesuai docs/lock.md: test tidak ikut fit, prior, emisi, maupun tetangga kNN.

Inisialisasi dari peta kosakata (kosakata.py), tanpa nomor cluster dan tanpa nama cluster tertulis tangan.
Empat varian dilatih: semua kosakata, serta tiga varian yang masing-masing memakai satu kosakata saja untuk uji
kesepakatan antar pandangan. Tanpa temuan.

Keluaran: output/rt_{varian}.npy (2400 x 8), output/label_train_em.csv, output/kandidat_test_em.csv

    python temp/eda_110_fusi/em_train.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd

from kosakata import ADE, COCO, NAMES

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
E5 = ROOT / "temp" / "exp_005_imagenet_map" / "output"
EM_D = ROOT / "temp" / "eda_106_eomt" / "output"
pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 300)

g = pd.read_csv(ROOT / "temp" / "eda_102_sumber" / "output" / "meta_assign.csv", dtype={"image_id": str})
tr = g[g.split == "train"].sort_values("image_id")
te = g[g.split == "test"].sort_values("image_id")
d = pd.concat([tr, te], ignore_index=True)
nt, N = len(tr), len(d)
E = np.vstack([np.load(E6 / "emb_train.npy"), np.load(E6 / "emb_test.npy")]).astype(np.float32)

# tetangga hanya dari train
Sim = E @ E[:nt].T
Sim[np.arange(nt), np.arange(nt)] = -1
K = 10
idx = np.argpartition(-Sim, K, axis=1)[:, :K]
wk = np.clip(np.take_along_axis(Sim, idx, 1), 0, None) ** 4
wk = wk / wk.sum(1, keepdims=True)


def halus(X, alpha=.5):
    return alpha * X + (1 - alpha) * (X[idx] * wk[..., None]).sum(1)


LATAR_C = ["wall-other-merged", "floor-other-merged", "table-merged", "floor-wood", "wall-wood", "wall-tile", "wall-stone", "wall-brick", "dirt-merged", "rock-merged",
           "dining table", "counter", "mirror-stuff", "ceiling-merged", "cabinet-merged", "shelf", "person", "pavement-merged", "grass-merged", "sand", "snow",
           "sky-other-merged", "water-other", "window-other", "door-stuff", "bed", "gravel", "road", "platform"]
LATAR_A = ["wall", "floor", "table", "ceiling", "earth", "person", "desk", "cabinet", "sidewalk", "road", "grass", "sand", "windowpane", "door", "countertop", "shelf",
           "water", "rock", "building", "sky"]


def muat(m, latar):
    kel = json.loads((EM_D / f"{m}_kelas.json").read_text(encoding="utf-8"))
    A = np.vstack([np.load(EM_D / f"{m}_train.npy"), np.load(EM_D / f"{m}_test.npy")])
    C = .5 * A[:, :, 0] + .5 * A[:, :, 1]
    C[:, [kel.index(k) for k in latar]] = 0
    return C, kel


def peta_mat(kel, peta):
    M = np.zeros((8, len(kel)), dtype=np.float32)
    for k, v in peta.items():
        for c, w in v.items():
            M[NAMES.index(c), kel.index(k)] = w
    return M


p_csv = pd.concat([pd.read_csv(E5 / f"prob_{s}.csv", dtype={"image_id": str}).assign(split=s) for s in ("train", "test")])
p_csv = d[["split", "image_id"]].merge(p_csv, on=["split", "image_id"])
S0 = {"in1k": p_csv[NAMES].values * p_csv[["cakupan"]].values}
C_c, kel_c = muat("coco", LATAR_C)
C_a, kel_a = muat("ade", LATAR_A)
S0["coco"] = C_c @ peta_mat(kel_c, COCO).T
S0["ade"] = C_a @ peta_mat(kel_a, ADE).T

P1 = np.vstack([np.load(E5 / "imagenet_train.npy"), np.load(E5 / "imagenet_test.npy")])
X_all = {"in1k": P1, "coco": C_c, "ade": C_a}
for k in X_all:
    x = halus(X_all[k])
    X_all[k] = x / (x.sum(1, keepdims=True) + 1e-9)


def init_R(views):
    L = np.zeros((N, 8))
    for k in views:
        v = S0[k]
        prior = v[:nt].mean(0) + 1e-6
        r = v / prior
        r = .5 * r + .5 * (r[idx] * wk[..., None]).sum(1)
        r = r / (r.sum(1, keepdims=True) + 1e-9)
        L += np.log(r + 1e-3)
    R = np.exp(L - L.max(1, keepdims=True))
    return R / R.sum(1, keepdims=True)


def latih(views, iters=30, tajam=3.0):
    R = init_R(views)
    A0 = {}
    for it in range(iters):
        Em = {}
        for k in views:
            c = R[:nt].T @ X_all[k][:nt]  # hanya train
            if k not in A0:
                A0[k] = c / (c.sum(0, keepdims=True) + 1e-9)
            e = c + A0[k]
            Em[k] = e / e.sum(1, keepdims=True)
        p = R[:nt].sum(0) + 1
        L = np.tile(np.log(p / p.sum()), (N, 1))
        for k in views:
            L = L + tajam * (X_all[k] @ np.log(Em[k].T + 1e-12))
        Rn = np.exp(L - L.max(1, keepdims=True))
        Rn = Rn / Rn.sum(1, keepdims=True)
        delta = float(np.abs(Rn - R)[:nt].sum(1).mean() / 2)
        R = Rn
        if delta < 1e-3:
            break
    return R, Em, it


hasil = {}
for nama, views in (("semua", ["in1k", "coco", "ade"]), ("in1k", ["in1k"]), ("coco", ["coco"]), ("ade", ["ade"])):
    R, Em, it = latih(views)
    hasil[nama] = R
    np.save(OUT / f"rt_{nama}.npy", R.astype(np.float32))
    y = R.argmax(1)
    print(f"{nama:6s} iter {it:2d} test {dict(zip(NAMES, np.bincount(y[nt:], minlength=8).tolist()))}")
    print(f"       train {dict(zip(NAMES, np.bincount(y[:nt], minlength=8).tolist()))}", flush=True)

R = hasil["semua"]
y = R.argmax(1)
print("\ntest per keluarga, varian semua (bukan bagian fit)")
print(pd.crosstab(d.fam[nt:].values, np.array(NAMES)[y[nt:]]).reindex(columns=NAMES).fillna(0).astype(int).to_string())
print("\ntrain per keluarga, varian semua")
print(pd.crosstab(d.fam[:nt].values, np.array(NAMES)[y[:nt]]).reindex(columns=NAMES).fillna(0).astype(int).to_string())

# kesepakatan antar pandangan yang dilatih terpisah
a, b, c = (hasil[k].argmax(1) for k in ("in1k", "coco", "ade"))
for nama, m in (("train", np.arange(N) < nt), ("test", np.arange(N) >= nt)):
    print(f"{nama}: in1k=coco {(a == b)[m].mean():.3f}, in1k=ade {(a == c)[m].mean():.3f}, coco=ade {(b == c)[m].mean():.3f}, ketiganya {((a == b) & (b == c))[m].mean():.3f}, "
          f"semua=mayoritas {(y == np.where(a == b, a, np.where(b == c, b, np.where(a == c, a, -1))))[m].mean():.3f}")

kons = (a == y).astype(int) + (b == y).astype(int) + (c == y).astype(int)
print("jumlah pandangan setuju dengan varian semua, train:", np.bincount(kons[:nt], minlength=4).tolist())
print("per kelas, porsi train dengan minimal 2 pandangan setuju:")
print(pd.DataFrame({"kelas": np.array(NAMES)[y[:nt]], "ok": kons[:nt] >= 2}).groupby("kelas").ok.agg(["mean", "sum", "size"]).round(3).to_string())

lb = pd.DataFrame({"image_id": d.image_id[:nt], "label": np.array(NAMES)[y[:nt]], "yakin": R[:nt].max(1).round(4), "setuju": kons[:nt], "fam": d.fam[:nt]})
lb.to_csv(OUT / "label_train_em.csv", index=False)
pd.DataFrame({"image_id": d.image_id[nt:], "waste_category": np.array(NAMES)[y[nt:]]}).to_csv(OUT / "kandidat_test_em.csv", index=False)
lama = pd.read_csv(E6 / "draft_submission_dinov3_cluster.csv", dtype=str).set_index("image_id").loc[d.image_id[nt:]]["waste_category"].values
print("\nsetuju EM vs EXP-006 pada test:", round(float((np.array(NAMES)[y[nt:]] == lama).mean()), 4))
print(pd.crosstab(pd.Series(lama, name="exp006"), pd.Series(np.array(NAMES)[y[nt:]], name="em")).reindex(index=NAMES, columns=NAMES).fillna(0).astype(int).to_string())
