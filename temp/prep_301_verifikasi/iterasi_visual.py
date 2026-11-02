"""PREP 301c: iterasi 5 setelah inspeksi visual grid (rule.md 2026-11-02 mengizinkan Claude melihat grid untuk QA).

Masalah yang terlihat di grid dan tidak tertangkap proksi angka:
1. pita hitam dari sumber (letterbox asli, banyak di cluster elektronik) -> uji trim pita hitam murni
2. stretch mendistorsi bentuk pada citra non-persegi -> uji letterbox (isi = median warna tepi) pada citra paling tidak persegi
3. isian sudut rotasi +-15 mencolok -> bandingkan dihedral saja, +-15 isian konstan, +-15 isian median tepi
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from eda_common import DATA, buka, files, kelompok, out_dir

OUT = out_dir(__file__)
UK = 384
rng = np.random.default_rng(5)


def trim_hitam(a, ambang=12, std_maks=4, porsi_min=0.03):
    """Buang baris/kolom tepi yang hitam murni dan seragam (pita letterbox sumber). Latar gelap bertekstur tidak lolos std."""
    g = a.max(2).astype(np.float32)
    def tepi(prof_max, prof_std):
        n = 0
        while n < len(prof_max) and prof_max[n] < ambang and prof_std[n] < std_maks:
            n += 1
        return n
    rmax, rstd = np.percentile(g, 99, axis=1), g.std(1)
    cmax, cstd = np.percentile(g, 99, axis=0), g.std(0)
    t, b = tepi(rmax, rstd), tepi(rmax[::-1], rstd[::-1])
    l, r = tepi(cmax, cstd), tepi(cmax[::-1], cstd[::-1])
    H, W = g.shape
    t, b = (t, b) if (t + b) / H >= porsi_min else (0, 0)
    l, r = (l, r) if (l + r) / W >= porsi_min else (0, 0)
    if H - t - b < H * .3 or W - l - r < W * .3:
        return a, 0.0
    return a[t:H - b, l:W - r], 1 - (H - t - b) * (W - l - r) / (H * W)


def letterbox(im, uk=UK):
    a = np.asarray(im)
    fill = tuple(int(x) for x in np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), 0))
    w, h = im.size
    s = uk / max(w, h)
    im2 = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    bg = Image.new("RGB", (uk, uk), fill)
    bg.paste(im2, ((uk - im2.size[0]) // 2, (uk - im2.size[1]) // 2))
    return bg


g = kelompok(8)
lab = dict(zip(g.image_id, g.label_lama))
fs = files("train") + files("test")
rows = []
for p in fs:
    a = np.asarray(buka(p))
    _, porsi = trim_hitam(a)
    rows.append(dict(split=p.parent.name, image_id=int(p.stem), porsi_trim=porsi, path=str(p), w=a.shape[1], h=a.shape[0]))
d = pd.DataFrame(rows)
d["trim"] = d.porsi_trim > 0
d["label_lama"] = [lab.get(i) if s == "train" else None for s, i in zip(d.split, d.image_id)]
print("== 1. trim pita hitam")
print(d.groupby("split").trim.agg(["sum", "mean"]).round(3).to_string())
print("porsi luas terbuang (median, q90) pada yang di-trim:", d[d.trim].porsi_trim.quantile([.5, .9]).round(3).to_dict())
print("label lama pada train yang di-trim:", d[(d.split == "train") & d.trim].label_lama.value_counts().to_dict())
d.to_csv(OUT / "trim_hitam.csv", index=False)
x = d[d.trim].sample(min(16, int(d.trim.sum())), random_state=1)
fig, ax = plt.subplots(4, 8, figsize=(14, 7.5))
for k, r in enumerate(x.itertuples()):
    a = np.asarray(buka(r.path)); t, _ = trim_hitam(a)
    ax.ravel()[2 * k].imshow(a); ax.ravel()[2 * k + 1].imshow(t)
    ax.ravel()[2 * k].set_title(f"{r.split[:2]} {r.image_id} asli", fontsize=6); ax.ravel()[2 * k + 1].set_title(f"trim {r.porsi_trim:.2f}", fontsize=6)
for a_ in ax.ravel(): a_.axis("off")
fig.tight_layout(); fig.savefig(OUT / "grid_trim_hitam.png", dpi=90); plt.close(fig)
# kontrol: citra berlatar gelap yang TIDAK di-trim (pastikan latar gelap bertekstur tidak terpotong)
lt = pd.read_csv(Path(__file__).resolve().parents[1] / "eda_204_background/output/latar.csv")
gelap = lt[(lt.L_tepi < 15)].merge(d[["split", "image_id", "trim"]], on=["split", "image_id"])
print(f"kontrol latar gelap (L tepi < 15): {len(gelap)} citra, di-trim {int(gelap.trim.sum())}")

print("\n== 2. stretch vs letterbox pada citra paling tidak persegi (train)")
tr = d[d.split == "train"].assign(dist=lambda z: np.maximum(z.w / z.h, z.h / z.w))
nonsq = tr.sort_values("dist", ascending=False).iloc[::12][:8]  # sebar dari paling ekstrem
fig, ax = plt.subplots(8, 3, figsize=(6, 16))
for r_, row in enumerate(nonsq.itertuples()):
    im = buka(row.path)
    ax[r_, 0].imshow(im); ax[r_, 1].imshow(im.resize((UK, UK), Image.BICUBIC)); ax[r_, 2].imshow(letterbox(im))
    ax[r_, 0].set_title(f"{row.image_id} ar {row.dist:.2f}", fontsize=7); ax[r_, 1].set_title("stretch", fontsize=7); ax[r_, 2].set_title("letterbox", fontsize=7)
for a_ in ax.ravel(): a_.axis("off")
fig.tight_layout(); fig.savefig(OUT / "grid_stretch_vs_letterbox.png", dpi=90); plt.close(fig)
print("distorsi contoh:", nonsq.dist.round(2).tolist())

print("\n== 3. varian rotasi")
contoh = g[g.suara >= .99].sample(frac=1, random_state=4).groupby("label_lama").head(1)
def rot_konst(im, deg): return im.rotate(deg, resample=Image.BICUBIC, fillcolor=(124, 116, 104))
def rot_median(im, deg):
    a = np.asarray(im); fill = tuple(int(v) for v in np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), 0))
    return im.rotate(deg, resample=Image.BICUBIC, fillcolor=fill)
fig, ax = plt.subplots(8, 4, figsize=(8, 16))
for r_, row in enumerate(contoh.itertuples()):
    im = letterbox(buka(next((DATA / "train").glob(f"{row.image_id:04d}.*"))))
    k = int(rng.integers(1, 4)); deg = float(rng.choice([-1, 1]) * rng.uniform(10, 15))
    for c, (judul, out) in enumerate((("asli", im), ("dihedral saja", im.rotate(90 * k)), (f"+{deg:.0f} isian konstan", rot_konst(im, deg)), (f"+{deg:.0f} isian median tepi", rot_median(im, deg)))):
        ax[r_, c].imshow(out); ax[r_, c].set_title(f"{row.label_lama} {row.image_id} {judul}" if c == 0 else judul, fontsize=6)
for a_ in ax.ravel(): a_.axis("off")
fig.tight_layout(); fig.savefig(OUT / "grid_rotasi_varian.png", dpi=90); plt.close(fig)
print("simpan grid_trim_hitam.png, grid_stretch_vs_letterbox.png, grid_rotasi_varian.png, trim_hitam.csv di", OUT)

# keputusan iterasi 5 (setelah melihat grid): trim pita hitam + stretch, rotasi dihedral saja.
# Letterbox ditolak: setelah trim, citra distorsi > 1,34 masih 31,5% electronic (basis 10,1%), jadi pita isian
# akan kembali berkorelasi dengan kelas. Rotasi +-15 ditolak: isian sudut mencolok, isian median tepi memunculkan
# sudut hitam pada citra electronic berpita.
import json  # noqa: E402
f = OUT / "prep_final.json"
j = json.loads(f.read_text())
j["final"].update({"trim_hitam": {"ambang": 12, "std_maks": 4, "porsi_min": 0.03}, "resize": "trim_hitam_lalu_stretch_bicubic",
                   "rotasi_deg": 0, "catatan": "iterasi 5 visual: letterbox dan rotasi kecil ditolak"})
j["log"]["iterasi5"] = {"trim_train": int(d[d.split == "train"].trim.sum()), "trim_test": int(d[d.split == "test"].trim.sum()),
                        "trim_train_electronic": int(((d.split == "train") & d.trim & (d.label_lama == "electronic")).sum())}
f.write_text(json.dumps(j, indent=1, default=str))
print("prep_final.json diperbarui:", j["final"])
