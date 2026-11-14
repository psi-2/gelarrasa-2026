"""PREP 302 iterasi RRC: grid memperlihatkan objek tipis/kecil di latar polos hilang dari crop scale 0,4.

Ukur porsi foreground yang tersisa per crop memakai sampler RandomResizedCrop torchvision (algoritma sama dengan v2)
pada citra hasil muat_citra (trim + 512), mask foreground = jarak Lab ke median cincin tepi > 18 (EDA 204),
hanya pada citra berlatar polos (mask andal). Kriteria ditetapkan sebelum melihat hasil: pilih augmentasi terkuat
dengan p(sisa < 50%) <= 0,10 pada objek kecil (fg < 0,1) dan <= 0,02 pada semua latar polos.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import RandomResizedCrop

from eda_common import TEMP, out_dir, path_train
from pipa import muat_citra

OUT = out_dir(__file__)
torch.manual_seed(0)
rng = np.random.default_rng(0)
lat = pd.read_csv(TEMP / "eda_204_background/output/latar.csv").query("split == 'train' and tipe != 'bertekstur'")
kecil = lat[lat.fg_frac < .1].image_id.tolist()
semua = lat[lat.fg_frac.between(.05, .8)].sample(150, random_state=0).image_id.tolist()


def mask(i):
    a = cv2.resize(muat_citra(path_train(i)), (128, 128), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    t = max(1, int(128 * .06))
    ring = np.concatenate([lab[:t].reshape(-1, 3), lab[-t:].reshape(-1, 3), lab[:, :t].reshape(-1, 3), lab[:, -t:].reshape(-1, 3)])
    return np.linalg.norm(lab - np.median(ring, 0), axis=2) > 18


M = {i: mask(i) for i in set(kecil) | set(semua)}
M = {i: m for i, m in M.items() if m.sum() >= 20}
dummy = torch.zeros(3, 128, 128)
rows = []
for nama, smin, p in (("rrc0.4", .4, 1), ("rrc0.5", .5, 1), ("rrc0.6", .6, 1), ("rrc0.7", .7, 1), ("rrc0.8", .8, 1),
                      ("rrc0.4_p0.5", .4, .5), ("rrc0.5_p0.5", .5, .5)):
    for himp, ids in (("kecil", kecil), ("polos", semua)):
        ret, luas = [], []
        for i in ids:
            if i not in M:
                continue
            m = M[i]
            for _ in range(60):
                if rng.random() < p:
                    y, x, h, w = RandomResizedCrop.get_params(dummy, (smin, 1.0), (3 / 4, 4 / 3))
                else:
                    y, x, h, w = 0, 0, 128, 128
                ret.append(m[y:y + h, x:x + w].sum() / m.sum())
                luas.append(h * w / 128 ** 2)
        ret = np.array(ret)
        rows.append(dict(varian=nama, himpunan=himp, n_citra=sum(i in M for i in ids), p_hilang=np.mean(ret < .5).round(4),
                         median_sisa=np.median(ret).round(3), q10_sisa=np.quantile(ret, .1).round(3), luas_crop_mean=np.mean(luas).round(3)))
R = pd.DataFrame(rows)
print(R.to_string(index=False))
W = R.pivot(index="varian", columns="himpunan", values="p_hilang")
lolos = W[(W.kecil <= .10) & (W.polos <= .02)]
kuat = R[R.himpunan == "polos"].set_index("varian").luas_crop_mean
pilih = kuat.loc[lolos.index].idxmin() if len(lolos) else None
print("\nlolos:", lolos.index.tolist(), "| dipilih (crop rata-rata terkecil di antara yang lolos):", pilih)
R.to_csv(OUT / "iterasi_rrc.csv", index=False)
