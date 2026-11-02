"""PREP 301b: iterasi 3 color jitter. Iterasi 2 tidak monoton (sampel cardboard ~160 percobaan, noise +-1,5 pp).
Di sini memakai SEMUA citra cardboard lama x 8 draw, metrik flip biner cardboard -> paper (centroid Lab pusat)
dan penurunan chroma C* (pembeda utama cardboard vs paper menurut eda_205: b* 20,8 vs 7,2).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

from eda_common import DATA, TEMP, buka, kelompok, out_dir

OUT = out_dir(__file__)
rng = np.random.default_rng(11)
g = kelompok(8)
w = pd.read_csv(TEMP / "eda_205_warna/output/warna.csv"); w = w[w.split == "train"].merge(g, on="image_id")
cc, cp_ = w[w.label_lama == "cardboard"][["L", "a", "b"]].mean().values, w[w.label_lama == "paper"][["L", "a", "b"]].mean().values


def lab(im):
    x = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2LAB).astype(np.float32)[32:96, 32:96].reshape(-1, 3)
    x[:, 0] *= 100 / 255; x[:, 1:] -= 128
    return x.mean(0), np.hypot(x[:, 1], x[:, 2]).mean()


def jitter(im, b, c, s, h):
    if b: im = ImageEnhance.Brightness(im).enhance(rng.uniform(1 - b, 1 + b))
    if c: im = ImageEnhance.Contrast(im).enhance(rng.uniform(1 - c, 1 + c))
    if s: im = ImageEnhance.Color(im).enhance(rng.uniform(1 - s, 1 + s))
    if h:
        hsv = np.array(im.convert("HSV")); hsv[..., 0] = (hsv[..., 0].astype(int) + int(rng.uniform(-h, h) * 255)) % 256
        im = Image.fromarray(hsv, "HSV").convert("RGB")
    return im


ims = [buka(next((DATA / "train").glob(f"{i:04d}.*"))).resize((128, 128), 2) for i in w[w.label_lama == "cardboard"].image_id]
is_card = lambda v: np.linalg.norm(v - cc) < np.linalg.norm(v - cp_)
hasil = []
for p in [(0, 0, 0, 0), (.05, .05, 0, 0), (.1, .1, 0, 0), (.15, .15, 0, 0), (.15, 0, 0, 0), (.2, .1, 0, 0), (.1, .1, .05, 0), (.2, .2, 0, 0)]:  # iterasi 4
    flip, dC = [], []
    for im in ims:
        v0, C0 = lab(im)
        if not is_card(v0):
            continue
        for _ in range(8):
            v1, C1 = lab(jitter(im, *p))
            flip.append(not is_card(v1)); dC.append(C1 / C0 - 1)
    flip = np.array(flip)
    se = np.sqrt(flip.mean() * (1 - flip.mean()) / len(flip))
    hasil.append({"bcsh": p, "n": len(flip), "cardboard_ke_paper": round(float(flip.mean()), 4), "se": round(float(se), 4),
                  "chroma_turun_q10": round(float(np.quantile(dC, .1)), 3)})
    print(hasil[-1])
ok = [r for r in hasil[1:] if r["cardboard_ke_paper"] + 2 * r["se"] <= .03 and r["chroma_turun_q10"] >= -.2]
pilih = max(ok, key=lambda r: sum(r["bcsh"]))["bcsh"] if ok else (0, 0, 0, 0)
print("dipilih (batas atas 95% flip <= 0,03 dan q10 chroma >= -20%):", pilih)
f = json.load(open(OUT / "prep_final.json"))
f["final"]["color_jitter_bcsh"] = list(pilih); f["log"]["jitter_iterasi4"] = hasil
json.dump(f, open(OUT / "prep_final.json", "w"), indent=1, default=str)

# grid before/after per cluster lama untuk diperiksa manusia: parameter final vs jitter lama (ditolak)
import matplotlib.pyplot as plt  # noqa: E402
contoh = g[g.suara >= .99].sample(frac=1, random_state=3).groupby("label_lama").head(3)
for nama, prm in ((f"jitter_final_{'_'.join(map(str, pilih))}", pilih), ("jitter_lama_0.2_0.2_0.2_0.02_ditolak", (.2, .2, .2, .02))):
    fig, ax = plt.subplots(8, 6, figsize=(9, 13))
    for r, (lb, d) in enumerate(contoh.groupby("label_lama")):
        for c, i in enumerate(d.image_id):
            im = buka(next((DATA / "train").glob(f"{i:04d}.*"))).resize((384, 384), Image.BICUBIC)
            ax[r, 2 * c].imshow(im); ax[r, 2 * c + 1].imshow(jitter(im, *prm))
            ax[r, 2 * c].set_title(f"{lb} {i} asli", fontsize=6); ax[r, 2 * c + 1].set_title("sesudah", fontsize=6)
    for a in ax.ravel(): a.axis("off")
    fig.suptitle(nama); fig.tight_layout(); fig.savefig(OUT / f"grid_{nama}.png", dpi=90); plt.close(fig)
print("simpan grid jitter final dan lama di", OUT)
