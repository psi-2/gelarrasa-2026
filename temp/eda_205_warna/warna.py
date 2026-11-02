"""EDA 205: distribusi warna (Lab dan HSV) per split dan per cluster, plus jarak warna antar-cluster.

Statistik dihitung di area pusat 50% (lebih mewakili objek daripada latar). `output/warna.csv` dipakai ulang
oleh verifikasi color jitter (prep_301): jarak antar label lama jadi patokan seberapa jauh jitter boleh menggeser warna.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from eda_common import buka, files, kelompok, out_dir

OUT = out_dir(__file__)


def stat_lab(a):
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    c = lab[32:96, 32:96].reshape(-1, 3)
    hsv = cv2.cvtColor(a[32:96, 32:96], cv2.COLOR_RGB2HSV).reshape(-1, 3)
    sat = hsv[:, 1] > 40
    hh = np.histogram(hsv[sat, 0], bins=12, range=(0, 180))[0] / max(1, sat.sum())
    return dict(L=c[:, 0].mean(), a=c[:, 1].mean(), b=c[:, 2].mean(), C=np.hypot(c[:, 1], c[:, 2]).mean(),
                L_std=c[:, 0].std(), sat=hsv[:, 1].mean() / 255, porsi_bersat=sat.mean(), **{f"h{i:02d}": hh[i] for i in range(12)})


def fitur(p):
    return dict(split=p.parent.name, image_id=int(p.stem), **stat_lab(np.asarray(buka(p).resize((128, 128), 2))))


if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex:
        d = pd.DataFrame(list(ex.map(fitur, files("train") + files("test"), chunksize=32)))
    d.to_csv(OUT / "warna.csv", index=False)
    tr, te = d[d.split == "train"], d[d.split == "test"]
    for c in ("L", "a", "b", "C", "sat"):
        print(f"{c:4s} train med {tr[c].median():7.2f}  test med {te[c].median():7.2f}  KS {ks_2samp(tr[c], te[c]).statistic:.3f}")
    t = tr.merge(kelompok(8), on="image_id")
    m = t.groupby("label_lama")[["L", "a", "b", "C", "sat", "porsi_bersat"]].median().round(2)
    print("\nmedian Lab pusat per label lama:\n", m.to_string())
    cen = t.groupby("label_lama")[["L", "a", "b"]].mean()
    D = np.linalg.norm(cen.values[:, None] - cen.values[None], axis=2)
    Dd = pd.DataFrame(D, cen.index, cen.index).round(1)
    print("\njarak Euclid Lab antar centroid label lama:\n", Dd.to_string())
    np.fill_diagonal(D, np.inf)
    print("\npasangan terdekat:", sorted([(round(D[i, j], 1), cen.index[i], cen.index[j]) for i in range(8) for j in range(i + 1, 8)])[:5])
    # sebaran dalam kelas: patokan jitter dibanding variasi alami
    print("\nstd dalam kelas (L, a, b) median antar kelas:", t.groupby("label_lama")[["L", "a", "b"]].std().median().round(2).to_dict())

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for k, x in t.groupby("label_lama"):
        ax[0].scatter(x.a, x.b, s=4, alpha=.35, label=k)
    ax[0].set(xlabel="a*", ylabel="b*", title="warna pusat per label lama (train)"); ax[0].legend(fontsize=7, markerscale=3)
    t.boxplot("L", by="label_lama", ax=ax[1], rot=45, showfliers=False); ax[1].set(title="L* pusat", xlabel="")
    H = t.groupby("label_lama")[[f"h{i:02d}" for i in range(12)]].mean()
    ax[2].imshow(H.values, aspect="auto", cmap="viridis"); ax[2].set_yticks(range(8), H.index)
    ax[2].set_xticks(range(12), [f"{i*15}" for i in range(12)], fontsize=7); ax[2].set(title="histogram hue (piksel bersaturasi)", xlabel="hue (derajat/2, OpenCV)")
    fig.suptitle(""); fig.tight_layout()
    fig.savefig(OUT / "warna.png", dpi=120)
    print("simpan warna.png, warna.csv di", OUT)
