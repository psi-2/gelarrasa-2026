"""Salinan persis fungsi preprocessing notebook (trim, load, transform) untuk verifikasi lokal tanpa model."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import v2 as T

PREP = json.loads((Path(__file__).resolve().parents[1] / "prep_301_verifikasi/output/prep_final.json").read_text())["final"]
UKURAN, UKURAN_SIMPAN, TRIM = PREP["ukuran"], 512, (12, 4, 0.03)
JITTER = tuple(PREP["color_jitter_bcsh"][:2])


def trim_hitam(a, ambang=TRIM[0], std_maks=TRIM[1], porsi_min=TRIM[2]):
    g = a.max(2).astype(np.float32)

    def tepi(pmax, pstd):
        n = 0
        while n < len(pmax) and pmax[n] < ambang and pstd[n] < std_maks:
            n += 1
        return n

    rmax, rstd, cmax, cstd = np.percentile(g, 99, axis=1), g.std(1), np.percentile(g, 99, axis=0), g.std(0)
    t, b, l, r = tepi(rmax, rstd), tepi(rmax[::-1], rstd[::-1]), tepi(cmax, cstd), tepi(cmax[::-1], cstd[::-1])
    H, W = g.shape
    t, b = (t, b) if (t + b) / H >= porsi_min else (0, 0)
    l, r = (l, r) if (l + r) / W >= porsi_min else (0, 0)
    if H - t - b < H * .3 or W - l - r < W * .3:
        return a
    return a[t:H - b, l:W - r]


def muat_citra(p):
    with Image.open(p) as im:
        a = trim_hitam(np.asarray(ImageOps.exif_transpose(im).convert("RGB")))
    return np.asarray(Image.fromarray(a).resize((UKURAN_SIMPAN, UKURAN_SIMPAN), Image.BICUBIC))


def aug(rrc_scale):
    return [
        T.RandomResizedCrop(UKURAN, scale=rrc_scale, ratio=(3 / 4, 4 / 3), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomChoice([T.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
        T.ColorJitter(brightness=JITTER[0], contrast=JITTER[1]),
    ]


tf_eval_tampil = T.Compose([T.ToImage(), T.Resize((UKURAN, UKURAN), antialias=True)])
