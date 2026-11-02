"""Helper bersama untuk skrip EDA 2xx (laptop, CPU, tanpa inference model).

Hanya membaca citra (piksel/header) dan cache lama di temp/. Claude tidak membuka gambar
keluaran: montage disimpan untuk dilihat manusia.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TEMP = ROOT / "temp"
SEED = 42
KELAS = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]

META = TEMP / "eda_101_metadata/output/meta.csv"
VISUAL = TEMP / "eda_107_visual/output/visual.csv"
FAM = TEMP / "eda_102_sumber/output/meta_assign.csv"
PHASH = TEMP / "eda_104_duplikat/output/phash_bits.npy"
EMB_VIT = TEMP / "nb_final/output/cache/emb_vit.npy"  # 2400 x 1024, train lalu test, L2-norm
EMB_CNN = TEMP / "nb_final/output/cache/emb_cnn.npy"
LABEL_LAMA = TEMP / "eda_110_fusi/output/label_train_bag.csv"  # label cluster otomatis lama, train-only


def out_dir(script_file):
    d = Path(script_file).resolve().parent / "output"
    d.mkdir(parents=True, exist_ok=True)
    return d


def files(split):
    return sorted((DATA / split).iterdir())


def buka(path, maks=None):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if maks:
        im.thumbnail((maks, maks), Image.BICUBIC)
    return im


def meta():
    m = pd.read_csv(META, dtype={"image_id": int})
    fam = pd.read_csv(FAM, usecols=["split", "image_id", "fam"])
    return m.merge(fam, on=["split", "image_id"], how="left")


def emb(nama="vit"):
    a = np.load(EMB_VIT if nama == "vit" else EMB_CNN)
    return a[:1920], a[1920:]


def kelompok(k=8):
    """Cluster KMeans train-only di ruang ViT-L (seed tetap) + label cluster lama untuk interpretasi."""
    from sklearn.cluster import KMeans

    tr, _ = emb("vit")
    c = KMeans(k, n_init=10, random_state=SEED).fit_predict(tr)
    lama = pd.read_csv(LABEL_LAMA).sort_values("image_id")
    return pd.DataFrame({"image_id": np.arange(1, 1921), "cluster": c,
                         "label_lama": lama.label.values, "suara": lama.suara.values})


def montage(paths, captions, out, ncol=8, sel=160):
    n = len(paths)
    nrow = max(1, int(np.ceil(n / ncol)))
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol * 1.6, nrow * 1.8))
    for a in np.atleast_1d(ax).ravel():
        a.axis("off")
    for a, p, c in zip(np.atleast_1d(ax).ravel(), paths, captions):
        a.imshow(buka(p, sel))
        a.set_title(c, fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
