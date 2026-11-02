"""EDA 204: tipe latar dari cincin tepi (6% tiap sisi) + estimasi foreground (beda Lab dari median tepi).

Keluaran `output/latar.csv` dipakai ulang oleh verifikasi crop (prep_301). Montage per tipe latar untuk manusia.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda_common import buka, files, kelompok, montage, out_dir

OUT = out_dir(__file__)
S = 128
TEPI = 0.06
DE_FG = 18.0  # delta E (Lab) dari warna latar dianggap foreground


def fitur(p):
    a = np.asarray(buka(p).resize((S, S), 2))
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    t = max(1, int(S * TEPI))
    ring = np.concatenate([lab[:t].reshape(-1, 3), lab[-t:].reshape(-1, 3), lab[:, :t].reshape(-1, 3), lab[:, -t:].reshape(-1, 3)])
    med = np.median(ring, 0)
    de = np.linalg.norm(lab - med, axis=2)
    fg = de > DE_FG
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    ys, xs = np.nonzero(fg)
    box = (xs.min() / S, ys.min() / S, (xs.max() + 1) / S, (ys.max() + 1) / S) if len(xs) > 20 else (0, 0, 1, 1)
    sisi = [np.median(lab[:t], (0, 1)), np.median(lab[-t:], (0, 1)), np.median(lab[:, :t], (0, 1)), np.median(lab[:, -t:], (0, 1))]
    return dict(path=str(p), split=p.parent.name, image_id=int(p.stem), L_tepi=med[0], a_tepi=med[1], b_tepi=med[2],
                std_tepi=float(np.linalg.norm(ring.std(0))), beda_sisi=float(np.max([np.linalg.norm(x - sisi[0]) for x in sisi])),
                fg_frac=float(fg.mean()), fg_x0=box[0], fg_y0=box[1], fg_x1=box[2], fg_y1=box[3],
                de_pusat=float(np.median(de[S // 4:3 * S // 4, S // 4:3 * S // 4])))


def tipe(r):
    polos = r.std_tepi < 8 and r.beda_sisi < 12
    if polos and r.L_tepi > 85:
        return "putih_polos"
    if polos and r.L_tepi < 12:
        return "hitam_polos"
    if polos:
        return "polos_lain"
    return "bertekstur"


if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex:
        d = pd.DataFrame(list(ex.map(fitur, files("train") + files("test"), chunksize=32)))
    d["tipe"] = d.apply(tipe, axis=1)
    d.to_csv(OUT / "latar.csv", index=False)
    print(pd.crosstab(d.tipe, d.split, normalize="columns").round(3).to_string())
    print("\nfg_frac q05/q50/q95 per split:")
    print(d.groupby("split").fg_frac.quantile([.05, .5, .95]).unstack().round(3).to_string())
    d["fg_luas_box"] = (d.fg_x1 - d.fg_x0) * (d.fg_y1 - d.fg_y0)
    print(f"box foreground menyentuh tepi (x0<.03 atau x1>.97 dst): {np.mean((d.fg_x0 < .03) | (d.fg_y0 < .03) | (d.fg_x1 > .97) | (d.fg_y1 > .97)):.3f}")
    print(f"objek kecil (luas box < 0.25): {np.mean(d.fg_luas_box < .25):.3f}")
    g = kelompok(8)
    t = d[d.split == "train"].merge(g, on="image_id")
    dom = t.groupby("cluster").label_lama.agg(lambda x: x.value_counts().index[0])
    ct = pd.crosstab(t.cluster, t.tipe, normalize="index").round(2)
    ct["dominan"] = dom
    print("\ntipe latar per cluster K=8 (train):\n", ct.to_string())
    print("\ntipe latar vs label lama (porsi baris):\n", pd.crosstab(t.label_lama, t.tipe, normalize="index").round(2).to_string())

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    pd.crosstab(t.label_lama, t.tipe, normalize="index").plot.barh(stacked=True, ax=ax[0])
    ax[0].set(title="tipe latar per label lama (train)")
    for s, c in (("train", "C0"), ("test", "C3")):
        x = d[d.split == s]
        ax[1].hist(x.fg_frac, 40, alpha=.5, density=True, color=c, label=s)
        ax[2].hist(x.fg_luas_box, 40, alpha=.5, density=True, color=c, label=s)
    ax[1].set(title="porsi piksel foreground"); ax[2].set(title="luas box foreground")
    ax[1].legend(); ax[2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "background.png", dpi=120)
    rng = np.random.default_rng(42)
    for k, x in d[d.split == "train"].groupby("tipe"):
        x = x.iloc[rng.permutation(len(x))[:16]]
        montage(list(x.path), [f"{i}" for i in x.image_id], OUT / f"montage_latar_{k}.png")
    print("simpan background.png, latar.csv, montage_latar_*.png di", OUT)
