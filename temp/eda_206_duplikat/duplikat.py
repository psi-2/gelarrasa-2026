"""EDA 206: duplikat / near-duplicate (pHash dihedral + cos embedding ViT-L) dalam train dan train<->test.

Grup near-duplicate train-only dibentuk dengan union-find (Hamming <= H_MAX atau cos >= COS_MIN) dan disimpan
ke `output/grup_train.csv` untuk StratifiedGroupKFold. Pasangan train-test hanya diagnostik (test tidak ikut grup).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from eda_common import DATA, PHASH, emb, kelompok, montage, out_dir

OUT = out_dir(__file__)
Hb = np.load(PHASH)  # 2400 x 8 x 63, urut train lalu test
tr_e, te_e = emb("vit")
NTR = 1920


def hamming_min(A, B):
    """Hamming minimum atas 8 transformasi dihedral A terhadap basis B."""
    b = B[:, 0, :].astype(np.float32) * 2 - 1
    return np.min([(63 - (A[:, t, :].astype(np.float32) * 2 - 1) @ b.T) / 2 for t in range(8)], axis=0)


Htt = hamming_min(Hb[:NTR], Hb[:NTR]); Htt = np.minimum(Htt, Htt.T)
Hte = hamming_min(Hb[NTR:], Hb[:NTR])  # test x train
Ctt = tr_e @ tr_e.T
Cte = te_e @ tr_e.T
iu = np.triu_indices(NTR, 1)
lama = kelompok(8).label_lama.values

print("== train-train pasangan")
for h in (0, 2, 4, 6, 10):
    m = Htt[iu] <= h
    print(f"Hamming<={h:2d}: {int(m.sum()):5d} pasangan, label lama sama {np.mean(lama[iu[0][m]] == lama[iu[1][m]]):.3f}")
for c in (.99, .97, .95, .92, .90):
    m = Ctt[iu] >= c
    print(f"cos>={c:.2f}: {int(m.sum()):5d} pasangan, label lama sama {np.mean(lama[iu[0][m]] == lama[iu[1][m]]):.3f}")

print("\n== test -> train (diagnostik)")
nn_h, nn_c = Hte.min(1), Cte.max(1)
for h in (0, 4, 10):
    print(f"test dengan train Hamming<={h}: {np.mean(nn_h <= h):.3f} ({int((nn_h <= h).sum())})")
for c in (.99, .95, .90, .85):
    print(f"test dengan NN train cos>={c}: {np.mean(nn_c >= c):.3f}")
nn_tt = np.where(np.eye(NTR, dtype=bool), -1, Ctt).max(1)
print(f"median cos NN: train->train {np.median(nn_tt):.3f}  test->train {np.median(nn_c):.3f}")

print("\n== grup train-only per ambang (pilih yang grup terbesarnya tidak merantai)")
hasil = {}
for hmax, cmin in ((4, .95), (4, .93), (4, .92), (4, .90), (6, .92)):
    A = csr_matrix((Htt <= hmax) | (Ctt >= cmin))
    n, lab = connected_components(A, directed=False)
    uk = np.bincount(lab)
    pur = np.mean([np.mean(lama[lab == g] == pd.Series(lama[lab == g]).mode()[0]) for g in np.nonzero(uk > 1)[0]])
    hasil[(hmax, cmin)] = lab
    print(f"H<={hmax} | cos>={cmin}: {n} grup, terbesar {uk.max()}, citra di grup>1: {int(uk[lab].__gt__(1).sum())}, "
          f"kemurnian label lama dalam grup {pur:.3f}")
H_MAX, COS_MIN = 4, .92
lab = hasil[(H_MAX, COS_MIN)]
pd.DataFrame({"image_id": np.arange(1, NTR + 1), "grup": lab, "ukuran": np.bincount(lab)[lab]}).to_csv(OUT / "grup_train.csv", index=False)
print(f"dipakai H<={H_MAX} | cos>={COS_MIN}: simpan grup_train.csv")

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].hist(nn_tt, 60, alpha=.6, density=True, label="train->train"); ax[0].hist(nn_c, 60, alpha=.6, density=True, label="test->train")
ax[0].axvline(COS_MIN, c="k", ls="--"); ax[0].set(title="cos tetangga terdekat (ViT-L)"); ax[0].legend()
ax[1].hist(Htt[iu][Htt[iu] <= 20], bins=21, alpha=.6, label="train-train"); ax[1].hist(Hte.ravel()[Hte.ravel() <= 20], bins=21, alpha=.6, label="test-train")
ax[1].axvline(H_MAX + .5, c="k", ls="--"); ax[1].set(title="Hamming pHash dihedral (<=20)", yscale="log"); ax[1].legend()
uk = np.bincount(lab); ax[2].hist(uk, bins=np.arange(1, uk.max() + 2), log=True); ax[2].set(title=f"ukuran grup (H<={H_MAX}|cos>={COS_MIN})")
fig.tight_layout(); fig.savefig(OUT / "duplikat.png", dpi=120)
g = np.argmax(uk)
idx = np.nonzero(lab == g)[0][:24]
montage([next((DATA / "train").glob(f"{i + 1:04d}.*")) for i in idx], [f"{i + 1} {lama[i][:6]}" for i in idx], OUT / "montage_grup_terbesar.png")
print("simpan duplikat.png, montage_grup_terbesar.png di", OUT)
