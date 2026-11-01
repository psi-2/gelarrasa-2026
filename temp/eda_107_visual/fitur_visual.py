"""Hipotesis E dan F serta forensik: fitur numerik per citra tanpa model deep.

Warna, saturasi, tekstur, tepi, transparansi semu, latar, buram, kontras, padding,
anisotropi (jejak stretch), dan periode grid JPEG (jejak resize dari JPEG lama).

Keluaran: output/visual.csv

    python temp/eda_107_visual/fitur_visual.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
DATA = ROOT / "data"


def rgb2hsv(a):
    mx, mn = a.max(2), a.min(2)
    d = mx - mn + 1e-6
    s = (mx - mn) / (mx + 1e-6)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    return h, s, mx


def periode(sig):
    """Puncak spektrum sinyal rata-rata beda kedua sepanjang satu sumbu."""
    s = sig - ndi.uniform_filter1d(sig, 21)
    n = len(s)
    if n < 64:
        return 0.0, 0.0
    S = np.abs(np.fft.rfft(s * np.hanning(n)))
    f = np.fft.rfftfreq(n)
    m = (f > 1 / 40) & (f < 0.45)
    if not m.any():
        return 0.0, 0.0
    v = S[m]
    i = int(v.argmax())
    return float(1 / f[m][i]), float(v[i] / (np.median(v) + 1e-9))


def fitur(p):
    with Image.open(p) as im:
        im.load()
        a0 = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
    H, W = a0.shape[:2]
    g0 = a0 @ np.array([.299, .587, .114], dtype=np.float32)
    r = {"image_id": p.stem}

    # forensik pada ukuran asli
    dx = np.abs(np.diff(g0, 2, axis=1)).mean(0)
    dy = np.abs(np.diff(g0, 2, axis=0)).mean(1)
    r["per_x"], r["per_x_kuat"] = periode(dx)
    r["per_y"], r["per_y_kuat"] = periode(dy)
    ex, ey = np.abs(np.diff(g0, axis=1)).mean(), np.abs(np.diff(g0, axis=0)).mean()
    r["aniso"] = float(np.log((ex + 1e-6) / (ey + 1e-6)))
    F = np.abs(np.fft.fftshift(np.fft.fft2(g0 - g0.mean()))) ** 2
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[:H, :W]
    fx, fy = (xx - cx) / W, (yy - cy) / H
    rad = np.sqrt(fx ** 2 + fy ** 2)
    tot = F.sum() + 1e-9
    r["hf_025"] = float(F[rad > .25].sum() / tot)
    r["hf_035"] = float(F[rad > .35].sum() / tot)
    E_x = F[np.abs(fx) > .25].sum() / tot
    E_y = F[np.abs(fy) > .25].sum() / tot
    r["hf_aniso"] = float(np.log((E_x + 1e-12) / (E_y + 1e-12)))
    # derau Immerkaer
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    r["noise"] = float(np.abs(ndi.convolve(g0, k)).mean() * np.sqrt(np.pi / 2) / 6)
    # blokiness pada grid 8 asli
    d = np.abs(np.diff(g0, axis=1))
    c = np.arange(d.shape[1]) % 8 == 7
    r["blk8"] = float(d[:, c].mean() / (d[:, ~c].mean() + 1e-6))

    # padding dan tepi
    def pita(x):
        return x.std(axis=1).mean() if x.ndim == 2 else 0
    tp = max(4, min(H, W) // 64)
    b = [a0[:tp], a0[-tp:], a0[:, :tp].transpose(1, 0, 2), a0[:, -tp:].transpose(1, 0, 2)]
    r["tepi_std"] = float(np.mean([x.std(axis=(0, 1)).mean() for x in b]))
    r["tepi_mean"] = float(np.mean([x.mean() for x in b]))
    r["hitam_baris"] = float(max((a0[:H // 8].max(2).mean(1) < .03).mean(), (a0[-H // 8:].max(2).mean(1) < .03).mean()))
    r["hitam_kolom"] = float(max((a0[:, :W // 8].max(2).mean(0) < .03).mean(), (a0[:, -W // 8:].max(2).mean(0) < .03).mean()))
    r["putih_baris"] = float(max((a0[:H // 8].min(2).mean(1) > .97).mean(), (a0[-H // 8:].min(2).mean(1) > .97).mean()))
    r["putih_kolom"] = float(max((a0[:, :W // 8].min(2).mean(0) > .97).mean(), (a0[:, -W // 8:].min(2).mean(0) > .97).mean()))
    k1 = a0[:tp, :tp].reshape(-1, 3).mean(0)
    k2 = a0[:tp, -tp:].reshape(-1, 3).mean(0)
    k3 = a0[-tp:, :tp].reshape(-1, 3).mean(0)
    k4 = a0[-tp:, -tp:].reshape(-1, 3).mean(0)
    kk = np.stack([k1, k2, k3, k4])
    r["sudut_mean"] = float(kk.mean())
    r["sudut_var"] = float(kk.std(0).mean())

    # sisanya pada 256 x 256
    with Image.open(p) as im:
        a = np.asarray(im.convert("RGB").resize((256, 256), Image.BILINEAR), dtype=np.float32) / 255
    g = a @ np.array([.299, .587, .114], dtype=np.float32)
    h, s, v = rgb2hsv(a)
    for n, x in (("r", a[..., 0]), ("g", a[..., 1]), ("b", a[..., 2]), ("s", s), ("v", v)):
        r[f"{n}_mean"], r[f"{n}_std"] = float(x.mean()), float(x.std())
    rg, yb = a[..., 0] - a[..., 1], .5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    r["warna"] = float(np.hypot(rg.std(), yb.std()) + .3 * np.hypot(rg.mean(), yb.mean()))
    ang = 2 * np.pi * h
    wt = s + 1e-6
    r["hue_cos"] = float((np.cos(ang) * wt).sum() / wt.sum())
    r["hue_sin"] = float((np.sin(ang) * wt).sum() / wt.sum())
    hh = np.histogram(h[s > .15], bins=12, range=(0, 1))[0].astype(float)
    hh = hh / (hh.sum() + 1e-9)
    r["hue_entropi"] = float(-(hh * np.log(hh + 1e-9)).sum())
    for i in range(12):
        r[f"hue{i:02d}"] = float(hh[i])
    r["putih"] = float(((v > .9) & (s < .1)).mean())
    r["hitam"] = float((v < .1).mean())
    r["abu"] = float((s < .1).mean())
    r["coklat"] = float(((h > .04) & (h < .13) & (s > .2) & (s < .75) & (v > .2) & (v < .85)).mean())
    r["specular"] = float(((v > .95) & (s < .15)).mean())
    # tepi dan tekstur
    gx, gy = ndi.sobel(g, 1), ndi.sobel(g, 0)
    m = np.hypot(gx, gy)
    r["tepi_rapat"] = float((m > .3).mean())
    r["tepi_mean2"] = float(m.mean())
    th = np.arctan2(gy, gx)[m > .2]
    if len(th) > 50:
        oh = np.histogram(np.mod(th, np.pi), bins=8, range=(0, np.pi))[0].astype(float)
        oh /= oh.sum()
        r["orient_entropi"] = float(-(oh * np.log(oh + 1e-9)).sum())
    else:
        r["orient_entropi"] = 0.0
    r["lap_var"] = float(ndi.laplace(g).var())
    mu = ndi.uniform_filter(g, 9)
    sd = np.sqrt(np.clip(ndi.uniform_filter(g * g, 9) - mu * mu, 0, None))
    r["lok_std_mean"] = float(sd.mean())
    r["lok_std_q10"] = float(np.quantile(sd, .1))
    r["lok_halus"] = float((sd < .015).mean())
    # LBP 8 tetangga
    c = g[1:-1, 1:-1]
    nb = [g[:-2, :-2], g[:-2, 1:-1], g[:-2, 2:], g[1:-1, 2:], g[2:, 2:], g[2:, 1:-1], g[2:, :-2], g[1:-1, :-2]]
    code = sum(((n >= c).astype(np.int32) << i) for i, n in enumerate(nb))
    lh = np.bincount(code.ravel(), minlength=256).astype(float)
    lh /= lh.sum()
    r["lbp_entropi"] = float(-(lh * np.log(lh + 1e-9)).sum())
    r["lbp_datar"] = float(lh[0] + lh[255])
    # kontras dan pusat vs pinggir (objek vs latar)
    r["kontras"] = float(np.quantile(g, .98) - np.quantile(g, .02))
    r["pusat_pinggir"] = float(np.abs(a[64:192, 64:192].mean((0, 1)) - np.concatenate([a[:32].reshape(-1, 3), a[-32:].reshape(-1, 3)]).mean(0)).mean())
    # transparansi semu: objek pusat rendah saturasi, kontras lokal tinggi, specular ada
    pus = (s[64:192, 64:192] < .15)
    r["pusat_rendah_sat"] = float(pus.mean())
    r["pusat_spec"] = float(((v[64:192, 64:192] > .95) & (s[64:192, 64:192] < .15)).mean())
    return r


def main():
    files = []
    for split in ("train", "test"):
        files += [(split, p) for p in sorted((DATA / split).iterdir())]
    hasil = Parallel(n_jobs=8, batch_size=16)(delayed(fitur)(p) for _, p in files)
    df = pd.DataFrame(hasil)
    df.insert(0, "split", [s for s, _ in files])
    df.to_csv(OUT / "visual.csv", index=False)
    print(df.shape)
    print(df.describe().T[["mean", "min", "max"]].round(3).head(30).to_string())


if __name__ == "__main__":
    main()
