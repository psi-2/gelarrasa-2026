"""PREP 301: verifikasi preprocessing dan augmentasi usulan EDA, dengan iterasi parameter otomatis.

Claude TIDAK melihat grid (aturan no-VLM, docs/rule.md). Kerusakan diukur dengan proksi tanpa model:
- resize: porsi box foreground yang terpotong (center crop), distorsi aspek (stretch), porsi padding (letterbox)
- RandomResizedCrop: porsi piksel foreground yang tersisa, dihitung pada citra berlatar polos (mask andal)
- rotasi: porsi sudut kosong dan foreground yang keluar bingkai
- color jitter: perubahan assignment nearest-centroid Lab (centroid per label cluster lama), khususnya
  cardboard -> paper; patokannya jarak antar-centroid dari eda_205
- denoise: penurunan noise (Immerkaer) vs hilangnya tekstur (variance Laplacian) pada citra tidak bising
Grid before/after per cluster lama disimpan ke output/grid_*.png untuk diperiksa manusia.
Parameter final ditulis ke output/prep_final.json dan dipakai notebook.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda_204_background"))
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from scipy import ndimage as ndi

from eda_common import DATA, KELAS, TEMP, buka, kelompok, out_dir

OUT = out_dir(__file__)
rng = np.random.default_rng(42)
UK = 384
latar = pd.read_csv(TEMP / "eda_204_background/output/latar.csv")
warna = pd.read_csv(TEMP / "eda_205_warna/output/warna.csv")
g = kelompok(8)
tr = latar[latar.split == "train"].merge(g, on="image_id").reset_index(drop=True)
meta = pd.read_csv(TEMP / "eda_101_metadata/output/meta.csv")
meta = meta[meta.split == "train"].set_index("image_id")
tr["ar"] = tr.image_id.map(meta.w / meta.h)
LOG = {}


def path(i):
    return next((DATA / "train").glob(f"{i:04d}.*"))


# ---------- 1. resize ----------
ar = tr.ar.values
lo = np.where(ar >= 1, .5 - .5 / ar, 0); hi = np.where(ar >= 1, .5 + .5 / ar, 1)
lo_y = np.where(ar < 1, .5 - .5 * ar, 0); hi_y = np.where(ar < 1, .5 + .5 * ar, 1)
bw = (tr.fg_x1 - tr.fg_x0).clip(1e-3); bh = (tr.fg_y1 - tr.fg_y0).clip(1e-3)
ov = (np.clip(np.minimum(tr.fg_x1, hi) - np.maximum(tr.fg_x0, lo), 0, None) / bw) * (np.clip(np.minimum(tr.fg_y1, hi_y) - np.maximum(tr.fg_y0, lo_y), 0, None) / bh)
dist = np.maximum(ar, 1 / ar)
pad = 1 - 1 / dist
LOG["resize"] = {"centercrop_potong_box_gt10pct": float(np.mean(ov < .9)), "stretch_distorsi_gt1.2": float(np.mean(dist > 1.2)),
                 "stretch_distorsi_gt1.34": float(np.mean(dist > 1.34)), "letterbox_pad_median_nonpersegi": float(np.median(pad[dist > 1.02]))}
print("1. resize:", {k: round(v, 3) for k, v in LOG["resize"].items()})


# ---------- 2. RandomResizedCrop ----------
def mask_fg(i):
    from background import DE_FG, S, TEPI
    a = np.asarray(buka(path(i)).resize((S, S), 2))
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32); lab[..., 0] *= 100 / 255; lab[..., 1:] -= 128
    t = max(1, int(S * TEPI))
    ring = np.concatenate([lab[:t].reshape(-1, 3), lab[-t:].reshape(-1, 3), lab[:, :t].reshape(-1, 3), lab[:, -t:].reshape(-1, 3)])
    return np.linalg.norm(lab - np.median(ring, 0), axis=2) > DE_FG


def rrc_param(smin, n, r=(3 / 4, 4 / 3)):
    out = []
    for _ in range(n):
        for _t in range(10):
            a = rng.uniform(smin, 1); lr = np.exp(rng.uniform(np.log(r[0]), np.log(r[1])))
            w, h = np.sqrt(a * lr), np.sqrt(a / lr)
            if w <= 1 and h <= 1:
                x, y = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h); out.append((x, y, w, h)); break
        else:
            out.append((0, 0, 1, 1))
    return out


polos = tr[(tr.tipe != "bertekstur") & (tr.fg_frac.between(.05, .8))]
sampel = polos.sample(frac=1, random_state=42).groupby("label_lama").head(25)
masks = {i: mask_fg(i) for i in sampel.image_id}
LOG["rrc"] = {}
pilih = None
for smin in (0.08, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    ret = []
    for i, m in masks.items():
        S = m.shape[0]; tot = m.sum()
        for x, y, w, h in rrc_param(smin, 40):
            ret.append(m[int(y * S):int((y + h) * S) + 1, int(x * S):int((x + w) * S) + 1].sum() / tot)
    ret = np.array(ret)
    r = {"p_sisa_lt50": float(np.mean(ret < .5)), "median_sisa": float(np.median(ret)), "q10_sisa": float(np.quantile(ret, .1))}
    LOG["rrc"][smin] = r
    lolos = r["p_sisa_lt50"] <= .01 and r["median_sisa"] >= .9  # iterasi 2: mask hanya andal di latar polos, beri margin
    print(f"2. RRC scale=({smin},1): {r} -> {'LOLOS' if lolos else 'gagal'}")
    if lolos and pilih is None:
        pilih = smin
LOG["rrc_pilih"] = pilih


# ---------- 3. rotasi ----------
LOG["rotasi"] = {}
for deg in (10, 15, 20, 30, 45):
    kos, keluar = [], []
    for i, m in list(masks.items())[:120]:
        S = m.shape[0]
        th = rng.uniform(-deg, deg)
        isi = ndi.rotate(np.ones((S, S)), th, reshape=False, order=0)
        mr = ndi.rotate(m.astype(float), th, reshape=False, order=0)
        kos.append(1 - isi.mean()); keluar.append(1 - mr.sum() / m.sum())
    LOG["rotasi"][deg] = {"sudut_kosong_mean": float(np.mean(kos)), "fg_keluar_q90": float(np.quantile(keluar, .9))}
    print(f"3. rotasi +-{deg}: {LOG['rotasi'][deg]}")


# ---------- 4. color jitter ----------
def jitter(im, b, c, s, h):
    if b: im = ImageEnhance.Brightness(im).enhance(rng.uniform(1 - b, 1 + b))
    if c: im = ImageEnhance.Contrast(im).enhance(rng.uniform(1 - c, 1 + c))
    if s: im = ImageEnhance.Color(im).enhance(rng.uniform(1 - s, 1 + s))
    if h:
        hsv = np.array(im.convert("HSV")); hsv[..., 0] = (hsv[..., 0].astype(int) + int(rng.uniform(-h, h) * 255)) % 256
        im = Image.fromarray(hsv, "HSV").convert("RGB")
    return im


def lab_pusat(im):
    lab = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2LAB).astype(np.float32)[32:96, 32:96].reshape(-1, 3)
    lab[:, 0] *= 100 / 255; lab[:, 1:] -= 128
    return lab.mean(0)


wt = warna[warna.split == "train"].merge(g, on="image_id")
cen = wt.groupby("label_lama")[["L", "a", "b"]].mean().loc[KELAS].values
nc = lambda v: np.argmin(np.linalg.norm(cen - v, axis=1))
sj = wt.sample(frac=1, random_state=7).groupby("label_lama").head(40)
ims = {i: buka(path(i)).resize((128, 128), 2) for i in sj.image_id}
asli = {i: nc(lab_pusat(im)) for i, im in ims.items()}
lab_of = dict(zip(sj.image_id, sj.label_lama))
# iterasi 2: metrik 8-kelas terlalu sensitif (centroid glass/metal/paper/plastic hanya berjarak 3,6-7 dE, warna
# memang tidak membedakan keempatnya). Ganti: (a) flip biner cardboard vs paper, (b) geseran Lab vs std alami dalam kelas.
ci, pi = KELAS.index("cardboard"), KELAS.index("paper")
std_alami = wt.groupby("label_lama")[["L", "a", "b"]].std().median().values
biner = lambda v: np.linalg.norm(cen[ci] - v) < np.linalg.norm(cen[pi] - v)
LOG["jitter"] = []
for b, c, s, h in [(0, 0, 0, 0), (.1, .1, .05, 0), (.2, .2, .1, 0), (.2, .2, .2, .02), (.3, .3, .3, .05), (.4, .4, .4, .1), (.2, .2, 0, 0), (.3, .3, .1, 0), (.3, .3, .2, 0)]:
    geser, cp = [], []
    for i, im in ims.items():
        v0 = lab_pusat(im)
        for _ in range(4):
            v1 = lab_pusat(jitter(im, b, c, s, h))
            geser.append(np.abs(v1 - v0) / std_alami)
            if lab_of[i] == "cardboard" and biner(v0):
                cp.append(not biner(v1))
    geser = np.median(geser, 0)
    r = {"bcsh": (b, c, s, h), "geser_per_std_Lab": [round(float(x), 3) for x in geser], "cardboard_ke_paper": float(np.mean(cp))}
    LOG["jitter"].append(r)
    print(f"4. jitter b={b} c={c} s={s} h={h}: geser/std (L,a,b) {r['geser_per_std_Lab']}, cardboard->paper biner {r['cardboard_ke_paper']:.3f}")
ok = [r for r in LOG["jitter"][1:] if max(r["geser_per_std_Lab"]) <= .5 and r["cardboard_ke_paper"] <= .03]
LOG["jitter_pilih"] = max(ok, key=lambda r: sum(r["bcsh"]))["bcsh"] if ok else (0, 0, 0, 0)
print("   jitter dipilih (terkuat yang lolos: geser <= 0,5 std alami, cardboard->paper <= 0,03):", LOG["jitter_pilih"])


# ---------- 5. denoise ----------
vis = pd.read_csv(TEMP / "eda_107_visual/output/visual.csv")
vis = vis[vis.split == "train"].set_index("image_id")
bising = vis.noise.nlargest(20).index
bersih = vis[vis.noise < .005].sample(30, random_state=1).index
def noise_est(a):
    g0 = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    return np.abs(cv2.filter2D(g0, -1, k)).mean() * np.sqrt(np.pi / 2) / 6
lapv = lambda a: cv2.Laplacian(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var()
LOG["denoise"] = {}
for hh in (3, 6, 10):
    turun = np.median([1 - noise_est(cv2.fastNlMeansDenoisingColored(a, None, hh, hh, 7, 21)) / noise_est(a) for a in (np.asarray(buka(path(i), UK)) for i in bising)])
    tekstur = np.median([1 - lapv(cv2.fastNlMeansDenoisingColored(a, None, hh, hh, 7, 21)) / lapv(a) for a in (np.asarray(buka(path(i), UK)) for i in bersih)])
    LOG["denoise"][hh] = {"noise_turun_bising": float(turun), "tekstur_hilang_bersih": float(tekstur)}
    print(f"5. NLM h={hh}: noise turun {turun:.2f} pada citra bising, tekstur hilang {tekstur:.2f} pada citra bersih")


# ---------- grid before/after per cluster lama ----------
contoh = tr[tr.suara >= .99].sample(frac=1, random_state=3).groupby("label_lama").head(3)
def stretch(im): return im.resize((UK, UK), Image.BICUBIC)
def centercrop(im):
    w, h = im.size; s = UK / min(w, h); im = im.resize((max(UK, round(w * s)), max(UK, round(h * s))), Image.BICUBIC)
    w, h = im.size; l, t = (w - UK) // 2, (h - UK) // 2; return im.crop((l, t, l + UK, t + UK))
def letterbox(im):
    w, h = im.size; s = UK / max(w, h); im2 = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    a = np.asarray(im); fill = tuple(int(x) for x in np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), 0))
    bg = Image.new("RGB", (UK, UK), fill); bg.paste(im2, ((UK - im2.size[0]) // 2, (UK - im2.size[1]) // 2)); return bg
def rrc(im, smin):
    x, y, w, h = rrc_param(smin, 1)[0]; W, H = im.size
    return im.crop((int(x * W), int(y * H), int((x + w) * W), int((y + h) * H))).resize((UK, UK), Image.BICUBIC)
def dihedral(im, deg=15):
    k = rng.integers(4); im = im.rotate(90 * k, expand=True)
    if rng.random() < .5: im = im.transpose(Image.FLIP_LEFT_RIGHT)
    return stretch(im).rotate(rng.uniform(-deg, deg), resample=Image.BICUBIC, fillcolor=(124, 116, 104))
smin = LOG["rrc_pilih"] or .6
jp = LOG["jitter_pilih"]
LANGKAH = {"resize_stretch": stretch, "resize_centercrop": centercrop, "resize_letterbox": letterbox,
           f"rrc_scale{smin}": lambda im: rrc(im, smin), "rrc_scale0.08_ditolak": lambda im: rrc(im, .08),
           "rotasi_dihedral_15": dihedral, f"jitter_{'_'.join(map(str, jp))}": lambda im: jitter(stretch(im), *jp),
           "jitter_kuat_0.4_0.4_0.4_0.1_ditolak": lambda im: jitter(stretch(im), .4, .4, .4, .1)}
for nama, f in LANGKAH.items():
    fig, ax = plt.subplots(8, 6, figsize=(9, 13))
    for r, (lab, d) in enumerate(contoh.groupby("label_lama")):
        for c, i in enumerate(d.image_id):
            im = buka(path(i))
            ax[r, 2 * c].imshow(im); ax[r, 2 * c + 1].imshow(f(im))
            ax[r, 2 * c].set_title(f"{lab} {i} asli", fontsize=6); ax[r, 2 * c + 1].set_title("sesudah", fontsize=6)
    for a in ax.ravel(): a.axis("off")
    fig.suptitle(nama); fig.tight_layout(); fig.savefig(OUT / f"grid_{nama}.png", dpi=90); plt.close(fig)
fig, ax = plt.subplots(4, 4, figsize=(8, 8))
for k, i in enumerate(bising[:8]):
    a = np.asarray(buka(path(i), UK)); ax.ravel()[2 * k].imshow(a); ax.ravel()[2 * k + 1].imshow(cv2.fastNlMeansDenoisingColored(a, None, 6, 6, 7, 21))
    ax.ravel()[2 * k].set_title(f"{i} bising", fontsize=6); ax.ravel()[2 * k + 1].set_title("NLM h=6", fontsize=6)
for a in ax.ravel(): a.axis("off")
fig.tight_layout(); fig.savefig(OUT / "grid_denoise_nlm6.png", dpi=90); plt.close(fig)

final = {"ukuran": UK, "resize": "stretch_bicubic_antialias", "exif_transpose": True, "konversi": "RGB",
         "rrc_scale": [smin, 1.0], "rrc_ratio": [0.75, 1.3333], "hflip": 0.5, "vflip": 0.5, "rot90": True, "rotasi_deg": 15,
         "color_jitter_bcsh": list(jp), "denoise": None, "catatan": "lihat LOG; grid untuk verifikasi manusia"}
json.dump({"final": final, "log": LOG}, open(OUT / "prep_final.json", "w"), indent=1, default=str)
print("\nFINAL:", final)
print("simpan grid_*.png dan prep_final.json di", OUT)
