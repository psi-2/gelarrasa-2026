"""Hipotesis B: autokorelasi ID berurutan terhadap cluster, sumber, dan embedding. Uji permutasi 2.000 kali.

Bila ada kebocoran, hanya dilaporkan.

    python temp/eda_103_urutan_id/urutan.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
df = pd.read_csv(ROOT / "temp" / "eda_102_sumber" / "output" / "meta_assign.csv", dtype={"image_id": str})
rng = np.random.default_rng(42)


def nilai(x, lag):
    return float((x[lag:] == x[:-lag]).mean())


rows = []
for split in ("train", "test"):
    d = df[df.split == split].sort_values("image_id").reset_index(drop=True)
    emb = np.load(E6 / f"emb_{split}.npy")
    ids = np.load(E6 / f"ids_{split}.npy").astype(str)
    assert (ids == d.image_id.values).all()
    for kol in ("cluster", "nama", "fam", "wh", "qhash"):
        x = d[kol].astype(str).values
        for lag in (1, 2, 3, 5, 10):
            o = nilai(x, lag)
            b = np.array([nilai(rng.permutation(x), lag) for _ in range(500)])
            rows.append({"split": split, "kolom": kol, "lag": lag, "amatan": round(o, 4), "acak": round(b.mean(), 4),
                         "z": round((o - b.mean()) / (b.std() + 1e-9), 2)})
    for lag in (1, 2, 3, 5, 10):
        o = float((emb[lag:] * emb[:-lag]).sum(1).mean())
        b = []
        for _ in range(500):
            e = emb[rng.permutation(len(emb))]
            b.append((e[lag:] * e[:-lag]).sum(1).mean())
        b = np.array(b)
        rows.append({"split": split, "kolom": "cos_emb", "lag": lag, "amatan": round(o, 4), "acak": round(b.mean(), 4),
                     "z": round((o - b.mean()) / (b.std() + 1e-9), 2)})
    # posisi: apakah distribusi sumber dan cluster berubah sepanjang ID
    m = len(d) // 8
    blok = (np.arange(len(d)) / m).astype(int).clip(0, 7)
    print(split, "share fam per blok ID")
    print(pd.crosstab(blok, d.fam, normalize="index").round(2).to_string())
    print(pd.crosstab(blok, d.nama, normalize="index").round(2).to_string())
    print("size vs ID spearman", round(d["size"].corr(pd.Series(np.arange(len(d))), method="spearman"), 4),
          "mp vs ID", round(d["mp"].corr(pd.Series(np.arange(len(d))), method="spearman"), 4))
    # tetangga terdekat embedding: jarak ID
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1)
    nn = sim.argmax(1)
    gap = np.abs(nn - np.arange(len(emb)))
    print("median jarak ID ke NN", float(np.median(gap)), "harapan acak", round(len(emb) / 3, 1), "NN dalam 5 ID:", round(float((gap <= 5).mean()), 4))

t = pd.DataFrame(rows)
t.to_csv(OUT / "autokorelasi.csv", index=False)
print(t.to_string(index=False))
print("|z| maksimum", t.z.abs().max())
