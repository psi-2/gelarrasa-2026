"""Hipotesis C: duplikat dan near-duplicate train-test.

pHash 64 bit pada 8 transformasi dihedral (flip dan rotasi 90), dihitung sebagai sisi citra kecil abu-abu.
Tidak ada citra yang ditampilkan.

Keluaran: output/phash.npy (N x 8 uint64), output/pasangan_hash.csv, output/keluarga.csv

    python temp/eda_104_duplikat/hash_dup.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
DATA = ROOT / "data"
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)


def phash_dari(a):
    d = dctn(a, norm="ortho")[:8, :8].ravel()
    d = d[1:]
    return (d > np.median(d)).astype(np.uint8)


def hash_semua(p):
    with Image.open(p) as im:
        g = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float32)
    out = []
    for fl in (False, True):
        b = np.fliplr(g) if fl else g
        for r in range(4):
            out.append(phash_dari(np.rot90(b, r)))
    return np.array(out)  # 8 x 63


rows, H = [], []
for split in ("train", "test"):
    for p in sorted((DATA / split).iterdir()):
        rows.append((split, p.stem))
        H.append(hash_semua(p))
H = np.stack(H).astype(bool)  # N x 8 x 63
np.save(OUT / "phash_bits.npy", H)
ix = pd.DataFrame(rows, columns=["split", "image_id"])
N = len(ix)
base = H[:, 0, :].astype(np.float32) * 2 - 1
sel = []
for t in range(8):
    a = H[:, t, :].astype(np.float32) * 2 - 1
    ham = (63 - a @ base.T) / 2  # hamming(transformasi t dari i, basis j)
    sel.append(ham)
sel = np.stack(sel)  # 8 x N x N
best_t = sel.argmin(0)
best = sel.min(0)
np.fill_diagonal(best, 99)
iu = np.triu_indices(N, 1)
b = np.minimum(best[iu], best.T[iu])
print("distribusi jarak hamming dihedral terbaik (semua pasangan):")
for th in (0, 2, 4, 6, 8, 10, 12, 14, 16):
    print(th, int((b <= th).sum()))
TH = 10
ii, jj = np.where(np.triu(np.minimum(best, best.T) <= TH, 1))
p = pd.DataFrame({"a": ii, "b": jj})
p["ham"] = [min(best[i, j], best[j, i]) for i, j in zip(ii, jj)]
p["t"] = [best_t[i, j] if best[i, j] <= best[j, i] else best_t[j, i] for i, j in zip(ii, jj)]
p["sa"], p["sb"] = ix.split.values[ii], ix.split.values[jj]
p["ia"], p["ib"] = ix.image_id.values[ii], ix.image_id.values[jj]
p.to_csv(OUT / "pasangan_hash.csv", index=False)
print("pasangan ham<=10", len(p))
print(p.groupby(["sa", "sb"]).size().to_string())
print("transformasi terbaik (0-3 rot, 4-7 flip+rot):")
print(p.t.value_counts().sort_index().to_string())
print(p[p.ham <= 4].groupby(["sa", "sb"]).size().to_string())
g = csr_matrix((np.ones(len(p)), (ii, jj)), shape=(N, N))
_, lab = connected_components(g, directed=False)
ix["hash_kel"] = lab
sz = ix.groupby("hash_kel").size()
ix["kel_n"] = ix.hash_kel.map(sz)
print("citra dalam keluarga >1:", int((ix.kel_n > 1).sum()), "keluarga", int((sz > 1).sum()), "terbesar", int(sz.max()))
print("jumlah citra unik (komponen):", int(lab.max() + 1))
x = ix.groupby("hash_kel").split.agg(lambda s: tuple(sorted(set(s))))
print(x[sz > 1].value_counts().to_string())
print("citra test dengan pasangan di train:", ix[(ix.split == "test") & ix.hash_kel.isin(x[x.map(len) == 2].index)].shape[0])
print(ix.kel_n.value_counts().sort_index().to_string())
ix.to_csv(OUT / "keluarga.csv", index=False)
