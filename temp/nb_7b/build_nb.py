"""Builder notebook final DSC Gelar Rasa 2026: DINOv3 7B (clustering) + ConvNeXt-L DINOv3 (classifier tunggal).

    .venv/bin/python temp/nb_7b/build_nb.py

Output ke folder notebook/ di root repo. Notebook TIDAK dijalankan di lokal. Target: Kaggle T4 x2, path /kaggle/input.
"""
import json
import sys
from pathlib import Path
from textwrap import indent
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).resolve().parents[2] / "notebook" / "dsc_gelarrasa_2026_dinov3.ipynb"
HR = "---"
PREP = json.loads((Path(__file__).resolve().parents[1] / "prep_301_verifikasi/output/prep_final.json").read_text())["final"]


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": uuid4().hex[:8], "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "id": uuid4().hex[:8],
        "metadata": {}, "outputs": [], "source": source,
    }


def jaga(src: str, pesan: str = "Dilewati: NAMA_CLUSTER belum diisi (tahap 1).") -> str:
    """Bungkus sel tahap 2 supaya run tahap 1 selesai normal dan output tetap tersimpan."""
    return f"if not LABEL_SIAP:\n    print('{pesan}')\nelse:\n" + indent(src, "    ")


def section(title: str, anchor: str) -> dict:
    return md(
        f"{HR}\n\n"
        f"# {title} <a name=\"{anchor}\"></a>\n\n"
        f"{HR}"
    )


cells = []

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
cells.append(md(
    f"{HR}\n\n"
    "# DSC Gelar Rasa 2026: Klasifikasi Citra Sampah\n\n"
    "*Label train dari clustering embedding DINOv3, nama cluster diputuskan tim dengan bukti visual, "
    "lalu satu classifier ConvNeXt-L DINOv3 untuk 8 kategori sampah.*"
))
cells.append(md(
    f"{HR}\n\n"
    "## Nama Tim\n\n"
    "- Nama anggota 1\n"
    "- Nama anggota 2\n"
    "- Nama anggota 3"
))
cells.append(md(
    f"{HR}\n\n"
    "## Daftar Isi\n\n"
    "1. [**Pendahuluan**](#1)\n"
    "2. [**Inisialisasi**](#2)\n"
    "3. [**Exploratory Data Analysis**](#3)\n"
    "4. [**Ekstraksi Fitur**](#4)\n"
    "5. [**Perbandingan Ruang Embedding**](#5)\n"
    "6. [**Clustering Train**](#6)\n"
    "7. [**Penamaan Cluster**](#7)\n"
    "8. [**Training Classifier**](#8)\n"
    "9. [**Validasi Lokal**](#9)\n"
    "10. [**Inference Test**](#10)\n"
    "11. [**Submission**](#11)\n"
))

# ---------------------------------------------------------------------------
# 1. Pendahuluan
# ---------------------------------------------------------------------------
cells.append(section("Pendahuluan", "1"))
cells.append(md(
    "## Overview\n\n"
    "Kompetisi meminta klasifikasi citra sampah ke 8 kategori: glass, paper, electronic, metal, plastic, "
    "cardboard, textile, dan foodwaste. Data train berisi 1.920 citra tanpa label dan data test berisi 480 citra. "
    "Menurut jawaban resmi panitia, label train wajib diperoleh lewat clustering, lalu cluster boleh dinamai secara manual."
))
cells.append(md(
    "## Metrik\n\n"
    "Submission dinilai dengan accuracy, yaitu porsi prediksi benar terhadap seluruh citra test. Public leaderboard "
    "memuat sekitar 30% test (sekitar 145 citra), sehingga satu kesalahan bernilai sekitar 0,7 poin persen. "
    "Notebook ini tidak memakai skor leaderboard untuk keputusan apa pun."
))
cells.append(md(
    "## Kepatuhan\n\n"
    "| Aturan | Penerapan di notebook |\n"
    "|---|---|\n"
    "| Tanpa VLM | Hanya DINOv3 (self-supervised, citra saja) dan ConvNeXtV2 ImageNet-1k (citra saja) |\n"
    "| Label train dari clustering | KMeans pada embedding train, nama cluster diisi tim di bagian 7 |\n"
    "| Test hanya untuk inference | Test tidak ikut PCA, clustering, grup, normalisasi, atau seleksi model |\n"
    "| Satu model | Submission dari satu ConvNeXt-L yang dilatih ulang pada seluruh label, tanpa ensemble dan tanpa TTA |\n"
    "| Tanpa LB probing | Semua keputusan dari aturan yang ditetapkan sebelum run dan dari validasi lokal |"
))
cells.append(md(
    "## Pendekatan\n\n"
    "1. EDA ringkas: resolusi, kualitas JPEG, noise dan blur, warna, serta duplikat.\n"
    "2. Embedding train dari DINOv3 ViT-7B (bobot fp16 dibagi ke dua T4) dan DINOv3 ViT-L sebagai pembanding.\n"
    "3. Ruang embedding dipilih dengan aturan beku berbasis metrik tanpa label.\n"
    "4. KMeans train-only, stabilitas per citra dari bootstrap, citra rapuh tidak diberi label.\n"
    "5. Tim menamai tiap cluster dari montage dan top-5 kelas ImageNet sebagai bukti pendukung.\n"
    "6. ConvNeXt-L DINOv3 di-fine-tune, divalidasi dengan StratifiedGroupKFold berbasis grup near-duplicate.\n"
    "7. Model final dilatih ulang pada seluruh label, lalu memprediksi test sekali.\n\n"
    "Embedding 7B hanya dibutuhkan untuk train. Classifier final membaca piksel, jadi test tidak perlu diekstrak dengan 7B."
))

# ---------------------------------------------------------------------------
# 2. Inisialisasi
# ---------------------------------------------------------------------------
cells.append(section("Inisialisasi", "2"))
cells.append(md(
    "## Environment Setup\n\n"
    "Notebook dirancang untuk Kaggle GPU T4 x2 dengan Internet aktif. Bila dataset cache embedding dan cluster "
    "dilampirkan, `Settings.PAKAI_CACHE = True` melewati ekstraksi 7B. Perkiraan waktu: ekstraksi 7B train dua "
    "pandangan sekitar 30 menit, cross-validation 5 fold sekitar 1 jam, model final sekitar 15 menit."
))
cells.append(code("!nvidia-smi"))
cells.append(md("Sel berikut memasang pustaka yang versinya perlu dikunci."))
cells.append(code("%pip install -q timm==1.0.29"))

cells.append(md(
    "## Import Libraries\n\n"
    "Semua import dikumpulkan di satu sel."
))
cells.append(code(
    "import gc\n"
    "import hashlib\n"
    "import json\n"
    "import math\n"
    "import os\n"
    "import random\n"
    "import shutil\n"
    "import time\n"
    "from concurrent.futures import ThreadPoolExecutor\n"
    "from pathlib import Path\n"
    "\n"
    "import cv2\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import timm\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "from accelerate import init_empty_weights\n"
    "from accelerate.utils import set_module_tensor_to_device\n"
    "from huggingface_hub import hf_hub_download\n"
    "from IPython.display import display\n"
    "from kaggle_secrets import UserSecretsClient\n"
    "from PIL import Image, ImageOps\n"
    "from safetensors import safe_open\n"
    "from scipy.fft import dctn\n"
    "from scipy.sparse import csr_matrix\n"
    "from scipy.sparse.csgraph import connected_components\n"
    "from scipy.stats import ks_2samp\n"
    "from sklearn.cluster import KMeans\n"
    "from sklearn.decomposition import PCA\n"
    "from sklearn.metrics import adjusted_rand_score, confusion_matrix, silhouette_score\n"
    "from sklearn.model_selection import StratifiedGroupKFold\n"
    "from timm.data import ImageNetInfo\n"
    "from timm.optim import create_optimizer_v2\n"
    "from torch.utils.data import DataLoader, Dataset\n"
    "from torchvision.transforms import v2 as T\n"
    "\n"
    "print('timm', timm.__version__, '| torch', torch.__version__, '| GPU', torch.cuda.device_count())"
))

cells.append(md("## Seed Everything"))
cells.append(code(
    "def seed_everything(seed: int = 42):\n"
    "    random.seed(seed)\n"
    "    os.environ['PYTHONHASHSEED'] = str(seed)\n"
    "    np.random.seed(seed)\n"
    "    torch.manual_seed(seed)\n"
    "    torch.cuda.manual_seed_all(seed)\n"
    "    torch.backends.cudnn.benchmark = False\n"
    "    torch.backends.cudnn.deterministic = True\n"
    "\n"
    "\n"
    "def seed_worker(worker_id):\n"
    "    s = torch.initial_seed() % 2**32\n"
    "    np.random.seed(s)\n"
    "    random.seed(s)\n"
    "\n"
    "\n"
    "seed_everything(42)"
))

cells.append(md(
    "## Settings\n\n"
    "Semua path, revisi bobot, ambang, dan hyperparameter dikumpulkan di sini. Token Hugging Face dibaca dari "
    "Kaggle Secrets bernama `HF_TOKEN` bila dilampirkan. Token tidak wajib karena repo bobot tidak gated, tetapi "
    "mempercepat unduhan dan menghindari batas laju. Parameter augmentasi berasal dari "
    "verifikasi preprocessing lokal (`temp/prep_301_verifikasi`), bukan dari skor test. Aturan pemilihan ruang "
    "embedding, K, dan ambang stabilitas ditetapkan sebelum run."
))
cells.append(code(
    "class Settings:\n"
    "    SEED = 42\n"
    "    KELAS = ['glass', 'paper', 'electronic', 'metal', 'plastic', 'cardboard', 'textile', 'foodwaste']\n"
    "    INPUT = Path('/kaggle/input')\n"
    "    OUTPUT_DIR = Path('/kaggle/working')\n"
    "    CACHE_DIR = OUTPUT_DIR / 'cache'\n"
    "    FIG_DIR = OUTPUT_DIR / 'gambar'\n"
    "    HF_CACHE = Path('/tmp/hf')\n"
    "    PAKAI_CACHE = True\n"
    "    ULANG_CV = False\n"
    "\n"
    "    M7B = ('timm/vit_7b_patch16_dinov3.lvd1689m', 'b40f6c827cb02b13c7a23a76cf598489bb0f94e5', 'vit_7b_patch16_dinov3')\n"
    "    MVITL = ('timm/vit_large_patch16_dinov3.lvd1689m', '30c1109559f65dea34316b0d4842d35c5771fe11', 'vit_large_patch16_dinov3')\n"
    "    MCNN = ('timm/convnext_large.dinov3_lvd1689m', '0f8b496db5dea10c7533e85304400f0eddad55ac', 'convnext_large.dinov3_lvd1689m')\n"
    "    MHEAD = ('timm/convnextv2_huge.fcmae_ft_in22k_in1k_384', 'e2ddce36b729b827f01e8c3fd72be459d212c95c', 'convnextv2_huge.fcmae_ft_in22k_in1k_384')\n"
    "    POLA_7B_FP16 = 'dinov3_7b_fp16*.safetensors'\n"
    "    DISK_MIN_GIB = 30\n"
    "\n"
    "    UKURAN_SIMPAN = 512\n"
    f"    UKURAN = {PREP['ukuran']}\n"
    "    MEAN = (0.485, 0.456, 0.406)\n"
    "    STD = (0.229, 0.224, 0.225)\n"
    "    BS_EMB = 16\n"
    "    LAPIS_7B = (19, 29)\n"
    "\n"
    "    PCA_DIM = 128\n"
    "    K_BANDING = 48\n"
    "    K_KANDIDAT = (32, 48, 64)\n"
    "    UK_MIN_CLUSTER = 10\n"
    "    N_BOOT = 20\n"
    "    STAB_MIN = 0.7\n"
    "    SELISIH_ARI_7B = 0.03\n"
    "    H_MAX = 4\n"
    "    PORSI_PASANGAN = 0.000721\n"
    "\n"
    "    N_FOLD = 5\n"
    "    EPOCH = 8\n"
    "    BS = 16\n"
    "    LR = 1e-4\n"
    "    HEAD_MULT = 10.0\n"
    "    LAYER_DECAY = 0.75\n"
    "    WD = 0.05\n"
    "    WARMUP = 1\n"
    "    SMOOTH = 0.1\n"
    "    DROP_PATH = 0.1\n"
    f"    RRC_SCALE = ({PREP['rrc_scale'][0]}, 1.0)\n"
    f"    JITTER = ({PREP['color_jitter_bcsh'][0]}, {PREP['color_jitter_bcsh'][1]})\n"
    "    TRIM = (12, 4, 0.03)\n"
    "    NUM_WORKERS = 4\n"
    "\n"
    "\n"
    "CFG = Settings()\n"
    "for d in (CFG.CACHE_DIR, CFG.FIG_DIR, CFG.HF_CACHE):\n"
    "    d.mkdir(parents=True, exist_ok=True)\n"
    "try:\n"
    "    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')\n"
    "    print('HF_TOKEN dari Kaggle Secrets dipakai')\n"
    "except Exception:\n"
    "    print('HF_TOKEN tidak terpasang; repo timm DINOv3 tidak gated, unduhan tetap jalan tanpa token')"
))

cells.append(md(
    "## Load Dataset\n\n"
    "Lokasi data dicari dari berkas `SampleSubmission_DSC.csv`, sehingga nama folder dataset tidak perlu ditulis "
    "tangan. Nama berkas diurutkan sebagai angka karena ekstensinya campur (`.jpg`, `.jpeg`, `.png`)."
))
cells.append(code(
    "SUB_PATH = next(CFG.INPUT.rglob('SampleSubmission_DSC.csv'))\n"
    "TRAIN_DIR = next(p for p in SUB_PATH.parent.rglob('train') if p.is_dir())\n"
    "TEST_DIR = next(p for p in SUB_PATH.parent.rglob('test') if p.is_dir())\n"
    "f_train = sorted(TRAIN_DIR.iterdir(), key=lambda p: int(p.stem))\n"
    "f_test = sorted(TEST_DIR.iterdir(), key=lambda p: int(p.stem))\n"
    "id_train = np.array([int(p.stem) for p in f_train])\n"
    "id_test = np.array([int(p.stem) for p in f_test])\n"
    "assert len(f_train) == 1920 and len(f_test) == 480, (len(f_train), len(f_test))\n"
    "print(TRAIN_DIR, len(f_train), '|', TEST_DIR, len(f_test))"
))

# ---------------------------------------------------------------------------
# 3. EDA
# ---------------------------------------------------------------------------
cells.append(section("Exploratory Data Analysis", "3"))
cells.append(md(
    "EDA di notebook ini adalah versi ringkas dari EDA lokal (`temp/eda_201` sampai `temp/eda_209`), yang memakai "
    "seluruh fitur berkas, warna, latar, duplikat, outlier, dan shift. Piksel test tidak dibaca di bagian ini. Dari "
    "test hanya header berkas (ukuran dan tabel kuantisasi JPEG) yang dipakai, dan itu pun hanya sebagai diagnostik."
))
cells.append(md(
    "## Resolusi dan Format\n\n"
    "Setiap citra train dibuka sekali: orientasi EXIF diterapkan, mode dikonversi ke RGB (ada PNG mode P dan RGBA), "
    "pita hitam murni dari sumber dipotong, lalu disimpan di RAM sebagai 512 x 512. Ukuran 512 cukup untuk crop acak sebelum resize ke 384. Kualitas JPEG "
    "ditaksir dari tabel kuantisasi luminans terhadap tabel standar IJG."
))
cells.append(code(
    "IJG_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,\n"
    "                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,\n"
    "                     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], float)\n"
    "\n"
    "\n"
    "def kualitas_jpeg(im):\n"
    "    q = getattr(im, 'quantization', None)\n"
    "    if not q:\n"
    "        return np.nan\n"
    "    s = np.mean(q[0]) / IJG_LUMA.mean() * 100\n"
    "    return (200 - s) / 2 if s <= 100 else 5000 / s\n"
    "\n"
    "\n"
    "def header(p):\n"
    "    with Image.open(p) as im:\n"
    "        return dict(w=im.width, h=im.height, fmt=im.format, mode=im.mode, q=kualitas_jpeg(im))\n"
    "\n"
    "\n"
    "def trim_hitam(a, ambang=CFG.TRIM[0], std_maks=CFG.TRIM[1], porsi_min=CFG.TRIM[2]):\n"
    "    g = a.max(2).astype(np.float32)\n"
    "\n"
    "    def tepi(pmax, pstd):\n"
    "        n = 0\n"
    "        while n < len(pmax) and pmax[n] < ambang and pstd[n] < std_maks:\n"
    "            n += 1\n"
    "        return n\n"
    "\n"
    "    rmax, rstd, cmax, cstd = np.percentile(g, 99, axis=1), g.std(1), np.percentile(g, 99, axis=0), g.std(0)\n"
    "    t, b, l, r = tepi(rmax, rstd), tepi(rmax[::-1], rstd[::-1]), tepi(cmax, cstd), tepi(cmax[::-1], cstd[::-1])\n"
    "    H, W = g.shape\n"
    "    t, b = (t, b) if (t + b) / H >= porsi_min else (0, 0)\n"
    "    l, r = (l, r) if (l + r) / W >= porsi_min else (0, 0)\n"
    "    if H - t - b < H * .3 or W - l - r < W * .3:\n"
    "        return a\n"
    "    return a[t:H - b, l:W - r]\n"
    "\n"
    "\n"
    "def muat_citra(p):\n"
    "    with Image.open(p) as im:\n"
    "        a = trim_hitam(np.asarray(ImageOps.exif_transpose(im).convert('RGB')))\n"
    "    return np.asarray(Image.fromarray(a).resize((CFG.UKURAN_SIMPAN, CFG.UKURAN_SIMPAN), Image.BICUBIC))\n"
    "\n"
    "\n"
    "with ThreadPoolExecutor(8) as ex:\n"
    "    X_train = np.stack(list(ex.map(muat_citra, f_train)))\n"
    "meta = pd.concat([pd.DataFrame([header(p) for p in f_train]).assign(split='train'),\n"
    "                  pd.DataFrame([header(p) for p in f_test]).assign(split='test')], ignore_index=True)\n"
    "meta['pendek'] = meta[['w', 'h']].min(axis=1)\n"
    "meta['ar'] = meta.w / meta.h\n"
    "ringkas = meta.groupby('split').agg(n=('w', 'size'), pendek_median=('pendek', 'median'),\n"
    "                                    pendek_lt384=('pendek', lambda s: (s < 384).mean()),\n"
    "                                    persegi=('ar', lambda s: (abs(s - 1) < .02).mean()), q_median=('q', 'median'))\n"
    "display(ringkas.round(3))\n"
    "display(meta.groupby(['split', 'fmt', 'mode']).size().rename('n').reset_index())\n"
    "for c in ('pendek', 'ar', 'q'):\n"
    "    r = ks_2samp(meta.loc[meta.split == 'train', c].dropna(), meta.loc[meta.split == 'test', c].dropna())\n"
    "    print(f'KS {c}: D={r.statistic:.3f} (diagnostik shift, bukan untuk tuning)')\n"
    "\n"
    "fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))\n"
    "for s, c in (('train', 'C0'), ('test', 'C3')):\n"
    "    d = meta[meta.split == s]\n"
    "    ax[0].hist(np.log2(d.pendek), 40, alpha=.5, density=True, color=c, label=s)\n"
    "    ax[1].hist(np.clip(d.ar, .3, 2.5), 50, alpha=.5, density=True, color=c, label=s)\n"
    "ax[0].axvline(np.log2(CFG.UKURAN), c='k', ls='--')\n"
    "ax[0].set(title='log2 sisi pendek (garis: 384)')\n"
    "ax[1].set(title='aspect ratio w/h')\n"
    "for a in ax:\n"
    "    a.legend()\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(md(
    "#### Insights\n\n"
    "> Median sisi pendek 416 px, 88% train berukuran minimal 384 px, dan 77% train persegi (test 82%). Input 384 "
    "membuat hampir semua citra di-downscale. Input 448 akan meng-upsample sekitar 53% citra (ukuran 416), dan ini "
    "sejalan dengan catatan lama bahwa ConvNeXt 448 lebih buruk dari 384.\n\n"
    "> Center crop memotong lebih dari 10% box objek pada 16,7% citra (EDA lokal `prep_301`). Letterbox menjaga "
    "bentuk, tetapi grid visual memperlihatkan masalah lain: 123 citra train (113 berlabel lama electronic) dan 45 "
    "citra test membawa pita hitam letterbox dari sumbernya. Pita ini adalah jalan pintas menuju kelas electronic "
    "dan memboroskan rata-rata 30% luas citra. Pita hitam murni dipotong lebih dulu. Setelah dipotong, citra dengan "
    "distorsi di atas 1,34 masih 31,5% electronic (basis 10,1%), sehingga letterbox akan memunculkan pita lagi pada "
    "kelas yang sama. Karena itu resize memakai stretch bicubic ke persegi, dan distorsi di atas 1,5 hanya terjadi "
    "pada 7,9% citra.\n\n"
    "> Sebanyak 87% train dan 94% test sudah di-encode ulang di kualitas sekitar 75 oleh encoder yang sama. "
    "Perbedaan sumber berkas antar split adalah shift utama: AUC adversarial pada metadata 0,66 dan pada embedding "
    "0,65, keduanya hanya diagnostik."
))

cells.append(md(
    "## Noise, Blur, Warna, dan Latar\n\n"
    "Noise ditaksir dengan metode Immerkaer, blur dengan variance Laplacian pada skala abu-abu 256 px, warna dari "
    "rata-rata Lab area pusat 50%, dan keseragaman latar dari simpangan baku Lab cincin tepi 6%."
))
cells.append(code(
    "K_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)\n"
    "\n"
    "\n"
    "def statistik(x):\n"
    "    g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255\n"
    "    noise = np.abs(cv2.filter2D(g, -1, K_IMMERKAER)).mean() * math.sqrt(math.pi / 2) / 6\n"
    "    blur = cv2.Laplacian(cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA), cv2.CV_32F).var()\n"
    "    lab = cv2.cvtColor(cv2.resize(x, (128, 128), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB).astype(np.float32)\n"
    "    lab[..., 0] *= 100 / 255\n"
    "    lab[..., 1:] -= 128\n"
    "    pusat = lab[32:96, 32:96].reshape(-1, 3)\n"
    "    tepi = np.concatenate([lab[:8].reshape(-1, 3), lab[-8:].reshape(-1, 3), lab[:, :8].reshape(-1, 3), lab[:, -8:].reshape(-1, 3)])\n"
    "    return dict(noise=noise, blur=blur, L=pusat[:, 0].mean(), chroma=np.hypot(pusat[:, 1], pusat[:, 2]).mean(),\n"
    "                std_tepi=np.linalg.norm(tepi.std(0)), L_tepi=np.median(tepi[:, 0]))\n"
    "\n"
    "\n"
    "stat = pd.DataFrame([statistik(x) for x in X_train])\n"
    "stat['latar'] = np.where(stat.std_tepi >= 8, 'bertekstur', np.where(stat.L_tepi > 85, 'putih_polos', 'polos_lain'))\n"
    "print(f\"noise > 0,03: {(stat.noise > .03).mean():.3f} | blur < 5e-4: {(stat.blur < 5e-4).mean():.3f}\")\n"
    "print(stat.latar.value_counts(normalize=True).round(3).to_dict())\n"
    "fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))\n"
    "ax[0].hist(np.log10(stat.noise + 1e-5), 40)\n"
    "ax[0].axvline(np.log10(.03), c='k', ls='--')\n"
    "ax[0].set(title='log10 noise')\n"
    "ax[1].hist(np.log10(stat.blur + 1e-6), 40)\n"
    "ax[1].axvline(np.log10(5e-4), c='k', ls='--')\n"
    "ax[1].set(title='log10 variance Laplacian')\n"
    "ax[2].scatter(stat.L, stat.chroma, s=3, alpha=.4)\n"
    "ax[2].set(xlabel='L* pusat', ylabel='chroma pusat', title='warna area pusat')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(md(
    "#### Insights\n\n"
    "> Pada resolusi asli (EDA lokal), taksiran noise tinggi hanya sekitar 2,4% dan blur sekitar 2%, terkumpul di "
    "citra hasil unduhan web. Angka di atas lebih kecil karena dihitung pada cache 512 px, dan downscale meredam noise. "
    "Inspeksi visual menunjukkan citra dengan noise tertinggi sebenarnya adalah tumpukan pakaian atau sampah yang "
    "teksturnya padat, beberapa dengan artefak blok upscale dan watermark situs stok foto. Jadi estimator Immerkaer "
    "di sini lebih banyak mengukur tekstur daripada noise sensor. Denoise NLM h=6 menghaluskan tumpukan kain dan "
    "menghapus 17% energi tekstur pada citra bersih, padahal tekstur adalah isyarat material. Tidak ada denoise.\n\n"
    "> Cardboard dibedakan dari paper terutama oleh chroma (b* median 20,8 lawan 7,2), sedangkan glass, metal, paper, "
    "dan plastic hanya berjarak 3,6 sampai 7 dE. Augmentasi warna yang menurunkan chroma berisiko membuat cardboard "
    "mirip paper.\n\n"
    "> Sekitar 77% citra berlatar bertekstur dan 17% berlatar putih polos. Porsi latar putih berbeda per cluster "
    "(sampai 52% pada cluster yang didominasi kaca), jadi cluster bisa ikut menangkap latar. Montage di bagian 7 "
    "perlu diperiksa untuk pola ini."
))

cells.append(md(
    "## Near-duplicate pHash\n\n"
    "pHash 63 bit dihitung pada 8 transformasi dihedral karena data memuat salinan hasil flip dan rotasi 90 derajat. "
    "Jarak Hamming minimum atas 8 transformasi dipakai. Ambang 4 dipilih di EDA lokal: kesepakatan label lama pada "
    "pasangan Hamming 4 atau kurang 0,99, sedangkan pada ambang 10 hanya 0,64 karena citra berlatar polos merantai."
))
cells.append(code(
    "def phash8(x):\n"
    "    g = cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)\n"
    "    out = []\n"
    "    for b in (g, np.fliplr(g)):\n"
    "        for r in range(4):\n"
    "            d = dctn(np.rot90(b, r), norm='ortho')[:8, :8].ravel()[1:]\n"
    "            out.append(d > np.median(d))\n"
    "    return np.array(out)\n"
    "\n"
    "\n"
    "PH = np.stack([phash8(x) for x in X_train])\n"
    "basis = PH[:, 0, :].astype(np.float32) * 2 - 1\n"
    "HAM = np.min([(63 - (PH[:, t, :].astype(np.float32) * 2 - 1) @ basis.T) / 2 for t in range(8)], axis=0)\n"
    "HAM = np.minimum(HAM, HAM.T)\n"
    "iu = np.triu_indices(len(HAM), 1)\n"
    "for h in (0, 2, 4, 10):\n"
    "    print(f'pasangan train-train Hamming <= {h}: {int((HAM[iu] <= h).sum())}')"
))
cells.append(md(
    "#### Insights\n\n"
    "> EDA lokal menemukan 100 pasangan train dengan Hamming 4 atau kurang dan 7,3% citra test punya pasangan "
    "train pada ambang yang sama. Split validasi acak akan menaruh salinan di train dan validasi sekaligus, jadi "
    "validasi memakai StratifiedGroupKFold dengan grup near-duplicate (bagian 6)."
))

cells.append(md(
    "## Verifikasi Augmentasi\n\n"
    "Parameter augmentasi dipilih lewat iterasi lokal (`temp/prep_301_verifikasi`) dengan proksi tanpa model. "
    "RandomResizedCrop scale minimum 0,4 adalah nilai terkecil dengan peluang sisa objek di bawah 50% tidak lebih "
    "dari 1% (diukur pada citra berlatar polos). Rotasi 90 derajat dan flip meniru salinan dihedral yang memang ada "
    "di data. Rotasi kecil 15 derajat ditolak setelah inspeksi grid: isian sudut konstan mencolok, dan isian warna "
    "tepi memunculkan sudut hitam pada citra electronic berpita. Color jitter dibatasi pada brightness 0,2 dan "
    "contrast 0,1 tanpa saturation dan hue. Jitter lama (0,2; 0,2; 0,2; 0,02) memindahkan 4,7% cardboard ke sisi "
    "paper, sedangkan parameter final 1,7%. Grid di bawah memperlihatkan hasil augmentasi untuk pemeriksaan visual."
))
cells.append(code(
    "AUG = [\n"
    "    T.RandomResizedCrop(CFG.UKURAN, scale=CFG.RRC_SCALE, ratio=(3 / 4, 4 / 3), antialias=True),\n"
    "    T.RandomHorizontalFlip(),\n"
    "    T.RandomVerticalFlip(),\n"
    "    T.RandomChoice([T.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),\n"
    "    T.ColorJitter(brightness=CFG.JITTER[0], contrast=CFG.JITTER[1]),\n"
    "]\n"
    "NORMAL = [T.ToDtype(torch.float32, scale=True), T.Normalize(CFG.MEAN, CFG.STD)]\n"
    "tf_train = T.Compose([T.ToImage(), *AUG, *NORMAL])\n"
    "tf_eval = T.Compose([T.ToImage(), T.Resize((CFG.UKURAN, CFG.UKURAN), antialias=True), *NORMAL])\n"
    "tf_tampil = T.Compose([T.ToImage(), *AUG])\n"
    "\n"
    "rng = np.random.default_rng(CFG.SEED)\n"
    "fig, ax = plt.subplots(6, 4, figsize=(8, 12))\n"
    "for r, i in enumerate(rng.choice(len(X_train), 6, replace=False)):\n"
    "    ax[r, 0].imshow(X_train[i])\n"
    "    ax[r, 0].set_title(f'{id_train[i]} asli', fontsize=7)\n"
    "    for c in range(1, 4):\n"
    "        ax[r, c].imshow(tf_tampil(X_train[i]).permute(1, 2, 0).numpy())\n"
    "        ax[r, c].set_title('augmentasi', fontsize=7)\n"
    "for a in ax.ravel():\n"
    "    a.axis('off')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(md(
    "#### Insights\n\n"
    "> Augmentasi menjaga isyarat material: crop tidak lebih kecil dari 40% luas, warna kardus tidak dipudarkan "
    "lewat saturation, tekstur tidak dihaluskan, dan tidak ada sudut isian yang bisa menjadi artefak. Grid lengkap per cluster lama (resize, trim pita hitam, crop, "
    "rotasi, jitter final, serta contoh parameter yang ditolak) ada di `temp/prep_301_verifikasi/output` untuk diperiksa tim."
))

# ---------------------------------------------------------------------------
# 4. Ekstraksi fitur
# ---------------------------------------------------------------------------
cells.append(section("Ekstraksi Fitur", "4"))
cells.append(md(
    "Embedding train diambil dari DINOv3 ViT-7B dan DINOv3 ViT-L dengan input dan preprocessing yang sama. Fitur "
    "yang disimpan untuk tiap pandangan (asli dan flip horizontal): token CLS lapis akhir, rata-rata patch token "
    "lapis akhir, dan, khusus 7B, CLS lapis 20 dan 30. Semua array disimpan fp16 di `cache/` beserta manifest "
    "(revisi bobot, waktu, VRAM, hash), sehingga run berikutnya cukup memuat cache."
))
cells.append(md(
    "## Utilitas Bobot dan Cache\n\n"
    "Bobot diunduh dengan revisi yang dikunci. Bobot 7B dicari lebih dulu sebagai dataset fp16 terlampir "
    "(`dinov3_7b_fp16*.safetensors`, sekitar 13,4 GB). Bila tidak ada, berkas fp32 resmi timm (26,9 GB, tidak gated) "
    "diunduh ke `/tmp`, dengan pemeriksaan ruang disk lebih dulu."
))
cells.append(code(
    "def unduh(repo, rev, nama='model.safetensors'):\n"
    "    return Path(hf_hub_download(repo, nama, revision=rev, cache_dir=CFG.HF_CACHE))\n"
    "\n"
    "\n"
    "def cari_cache(nama):\n"
    "    for akar in (CFG.CACHE_DIR, CFG.INPUT):\n"
    "        hit = sorted(akar.rglob(nama))\n"
    "        if hit:\n"
    "            return hit[0]\n"
    "    return None\n"
    "\n"
    "\n"
    "def berkas_7b():\n"
    "    lokal = sorted(CFG.INPUT.rglob(CFG.POLA_7B_FP16))\n"
    "    if lokal:\n"
    "        return lokal\n"
    "    bebas = shutil.disk_usage(CFG.HF_CACHE).free / 2**30\n"
    "    assert bebas >= CFG.DISK_MIN_GIB, f'Disk {bebas:.0f} GiB tidak cukup untuk bobot fp32 26,9 GB. Lampirkan dataset bobot fp16.'\n"
    "    return [unduh(*CFG.M7B[:2])]\n"
    "\n"
    "\n"
    "class Citra(Dataset):\n"
    "    def __init__(self, X, tf, idx=None, y=None):\n"
    "        self.X, self.tf, self.y = X, tf, y\n"
    "        self.idx = np.arange(len(X)) if idx is None else np.asarray(idx)\n"
    "\n"
    "    def __len__(self):\n"
    "        return len(self.idx)\n"
    "\n"
    "    def __getitem__(self, i):\n"
    "        x = self.tf(self.X[self.idx[i]])\n"
    "        return x if self.y is None else (x, int(self.y[i]))"
))

cells.append(md(
    "## DINOv3 di Dua GPU\n\n"
    "Bobot 7B fp16 sekitar 13,4 GB. Satu T4 (sekitar 15 GB terpakai) hanya menyisakan sekitar 1 GB untuk aktivasi, "
    "jadi 20 blok pertama ditaruh di GPU 0 dan 20 blok berikutnya di GPU 1. Model dibuat kosong dengan "
    "`init_empty_weights`, lalu tiap tensor dibaca satu per satu dari safetensors, dikonversi ke fp16, dan langsung "
    "dikirim ke GPU tujuan. Dengan cara ini RAM tidak pernah memuat seluruh bobot fp32.\n\n"
    "T4 tidak mendukung bf16, padahal DINOv3 dilatih dengan bf16. Untuk mencegah overflow fp16, residual stream "
    "disimpan dalam fp32, sedangkan matmul di attention dan MLP berjalan di fp16 lewat autocast. Bila sebuah blok "
    "tetap menghasilkan nilai non-finite, blok itu otomatis dijalankan ulang dalam fp32. `cek_presisi` membandingkan "
    "keluaran tiap blok fp16 dengan fp32 pada 4 citra sebelum ekstraksi."
))
cells.append(code(
    "def muat_dinov3(arsitektur, berkas, dua_gpu):\n"
    "    with init_empty_weights():\n"
    "        m = timm.create_model(arsitektur, pretrained=False, num_classes=0)\n"
    "    n = len(m.blocks)\n"
    "    batas = n // 2 if dua_gpu and torch.cuda.device_count() > 1 else n\n"
    "    akhir = 'cuda:1' if batas < n else 'cuda:0'\n"
    "\n"
    "    def perangkat(k):\n"
    "        if k.startswith('blocks.'):\n"
    "            return 'cuda:0' if int(k.split('.')[1]) < batas else 'cuda:1'\n"
    "        return akhir if k.startswith(('norm.', 'fc_norm.', 'head.')) else 'cuda:0'\n"
    "\n"
    "    param = dict(m.named_parameters())\n"
    "    kunci_model = set(m.state_dict())\n"
    "    terisi, asing = set(), []\n"
    "    for f in berkas:\n"
    "        with safe_open(str(f), 'pt', device='cpu') as sf:\n"
    "            for k in sf.keys():\n"
    "                if k not in kunci_model:\n"
    "                    asing.append(k)\n"
    "                    continue\n"
    "                t = sf.get_tensor(k)\n"
    "                fp16 = k in param and t.is_floating_point()\n"
    "                set_module_tensor_to_device(m, k, perangkat(k), value=t, dtype=torch.float16 if fp16 else None)\n"
    "                terisi.add(k)\n"
    "    kurang = set(param) - terisi\n"
    "    assert not kurang, f'parameter tanpa bobot: {sorted(kurang)[:5]}'\n"
    "    for nama, b in m.named_buffers():\n"
    "        if b.device.type != 'cuda':\n"
    "            set_module_tensor_to_device(m, nama, perangkat(nama), value=b)\n"
    "    m.eval()\n"
    "    m.batas = batas\n"
    "    m.blok32 = set()\n"
    "    print(f'{arsitektur}: {len(terisi)} tensor, {len(asing)} kunci asing dilewati, blok GPU0 {batas}, GPU1 {n - batas}')\n"
    "    return m\n"
    "\n"
    "\n"
    "def awal(m, x):\n"
    "    with torch.autocast('cuda', dtype=torch.float16):\n"
    "        h = m.patch_embed(x.to('cuda:0'))\n"
    "        h, rope = m._pos_embed(h)\n"
    "        h = m.norm_pre(h)\n"
    "    return h.float(), rope\n"
    "\n"
    "\n"
    "def lewat_blok(m, i, h, rope):\n"
    "    d = 'cuda:0' if i < m.batas else 'cuda:1'\n"
    "    h = h.to(d)\n"
    "    r = None if rope is None else rope.to(d)\n"
    "    blk = m.blocks[i]\n"
    "    if i not in m.blok32:\n"
    "        with torch.autocast('cuda', dtype=torch.float16):\n"
    "            y = blk(h, rope=r)\n"
    "        if torch.isfinite(y).all():\n"
    "            return y.float()\n"
    "        blk.float()\n"
    "        m.blok32.add(i)\n"
    "    with torch.autocast('cuda', enabled=False):\n"
    "        return blk(h, rope=None if r is None else r.float()).float()\n"
    "\n"
    "\n"
    "def normal_akhir(m, h):\n"
    "    with torch.autocast('cuda', dtype=torch.float16):\n"
    "        return m.norm(h.to(m.norm.weight.device)).float()\n"
    "\n"
    "\n"
    "@torch.no_grad()\n"
    "def maju_dinov3(m, x, lapis=()):\n"
    "    h, rope = awal(m, x)\n"
    "    antara = {}\n"
    "    for i in range(len(m.blocks)):\n"
    "        h = lewat_blok(m, i, h, rope)\n"
    "        if i in lapis:\n"
    "            antara[i] = normal_akhir(m, h)[:, 0]\n"
    "    h = normal_akhir(m, h)\n"
    "    return h[:, 0], h[:, m.num_prefix_tokens:].mean(1), antara\n"
    "\n"
    "\n"
    "@torch.no_grad()\n"
    "def cek_presisi(m, x):\n"
    "    h, rope = awal(m, x)\n"
    "    rows = []\n"
    "    for i, blk in enumerate(m.blocks):\n"
    "        d = 'cuda:0' if i < m.batas else 'cuda:1'\n"
    "        h = h.to(d)\n"
    "        r = None if rope is None else rope.to(d)\n"
    "        with torch.autocast('cuda', dtype=torch.float16):\n"
    "            y16 = blk(h, rope=r).float()\n"
    "        blk.float()\n"
    "        with torch.autocast('cuda', enabled=False):\n"
    "            y32 = blk(h, rope=None if r is None else r.float()).float()\n"
    "        if i not in m.blok32:\n"
    "            blk.half()\n"
    "        rows.append(dict(blok=i, galat_relatif=((y16 - y32).norm() / y32.norm()).item(),\n"
    "                         maks_abs=y32.abs().max().item(), finite16=bool(torch.isfinite(y16).all())))\n"
    "        h = y32\n"
    "    return pd.DataFrame(rows)"
))

cells.append(md(
    "## Ekstraksi Embedding Train\n\n"
    "Ekstraksi memakai `tf_eval` (stretch ke 384, tanpa augmentasi). Hanya train yang diekstrak. Pada 384 px, satu "
    "citra menghasilkan 576 patch, 1 CLS, dan 4 register. Perkiraan komputasi 7B sekitar 8 TFLOP per citra per "
    "pandangan, jadi 1.920 citra x 2 pandangan butuh sekitar 30 menit di T4 karena blok GPU 0 dan GPU 1 bergantian."
))
cells.append(code(
    "@torch.no_grad()\n"
    "def ekstrak(m, X, lapis=()):\n"
    "    dl = DataLoader(Citra(X, tf_eval), batch_size=CFG.BS_EMB, num_workers=CFG.NUM_WORKERS)\n"
    "    out = {}\n"
    "    for x in dl:\n"
    "        for v, xx in (('asli', x), ('flip', x.flip(3))):\n"
    "            cls, patch, antara = maju_dinov3(m, xx, lapis)\n"
    "            out.setdefault(f'{v}_cls', []).append(cls.half().cpu())\n"
    "            out.setdefault(f'{v}_patch', []).append(patch.half().cpu())\n"
    "            for i, a in antara.items():\n"
    "                out.setdefault(f'{v}_l{i + 1}', []).append(a.half().cpu())\n"
    "    return {k: torch.cat(v).numpy() for k, v in out.items()}\n"
    "\n"
    "\n"
    "def embedding(nama, arsitektur, berkas_fn, dua_gpu, lapis=()):\n"
    "    kunci = [f'{v}_{f}' for v in ('asli', 'flip') for f in ['cls', 'patch'] + [f'l{i + 1}' for i in lapis]]\n"
    "    nama_berkas = {k: f'emb_{nama}_train_{k}.npy' for k in kunci}\n"
    "    if CFG.PAKAI_CACHE and all(cari_cache(b) for b in nama_berkas.values()):\n"
    "        print(nama, 'dimuat dari cache', cari_cache(nama_berkas[kunci[0]]).parent)\n"
    "        return {k: np.load(cari_cache(b)).astype(np.float32) for k, b in nama_berkas.items()}\n"
    "    for i in range(torch.cuda.device_count()):\n"
    "        torch.empty(0, device=f'cuda:{i}')\n"
    "        torch.cuda.reset_peak_memory_stats(i)\n"
    "    t0 = time.time()\n"
    "    m = muat_dinov3(arsitektur, berkas_fn(), dua_gpu)\n"
    "    t_muat = time.time() - t0\n"
    "    presisi = cek_presisi(m, torch.stack([tf_eval(X_train[i]) for i in range(4)]))\n"
    "    presisi.to_csv(CFG.CACHE_DIR / f'presisi_{nama}.csv', index=False)\n"
    "    print(f'{nama}: galat relatif fp16 vs fp32 maks {presisi.galat_relatif.max():.2e}, '\n"
    "          f'blok non-finite di fp16 {int((~presisi.finite16).sum())}, |aktivasi| maks {presisi.maks_abs.max():.0f}')\n"
    "    t1 = time.time()\n"
    "    hasil = ekstrak(m, X_train, lapis)\n"
    "    manifest = dict(model=nama, arsitektur=arsitektur, ukuran=CFG.UKURAN, presisi='bobot fp16, residual fp32',\n"
    "                    blok_fp32=sorted(m.blok32), detik_muat=round(t_muat), detik_ekstrak=round(time.time() - t1),\n"
    "                    vram_gib=[round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())],\n"
    "                    timm=timm.__version__, sha256={k: hashlib.sha256(a.tobytes()).hexdigest()[:16] for k, a in hasil.items()})\n"
    "    for k, a in hasil.items():\n"
    "        np.save(CFG.CACHE_DIR / nama_berkas[k], a)\n"
    "    json.dump(manifest, open(CFG.CACHE_DIR / f'manifest_{nama}.json', 'w'), indent=1)\n"
    "    print(manifest)\n"
    "    del m\n"
    "    gc.collect()\n"
    "    torch.cuda.empty_cache()\n"
    "    return {k: a.astype(np.float32) for k, a in hasil.items()}\n"
    "\n"
    "\n"
    "E_VITL = embedding('vitl', CFG.MVITL[2], lambda: [unduh(*CFG.MVITL[:2])], dua_gpu=False)\n"
    "E_7B = embedding('7b', CFG.M7B[2], berkas_7b, dua_gpu=True, lapis=CFG.LAPIS_7B)\n"
    "print({k: v.shape for k, v in E_7B.items()})"
))

cells.append(md(
    "## Head ImageNet-1k sebagai Bukti Pendukung\n\n"
    "ConvNeXtV2-Huge (pretraining FCMAE, fine-tune ImageNet-1k, citra saja) memberi probabilitas 1.000 kelas "
    "ImageNet untuk tiap citra train. Keluarannya dipakai untuk dua hal: referensi independen pada perbandingan ruang "
    "embedding, dan daftar top-5 nama kelas ImageNet per cluster sebagai bukti pendukung penamaan. Tidak ada pemetaan "
    "kelas ImageNet ke 8 kategori. Nama mentah ditampilkan, dan tim yang menafsirkannya."
))
cells.append(code(
    "def prob_head():\n"
    "    b = 'head_in1k_train.npy'\n"
    "    if CFG.PAKAI_CACHE and cari_cache(b):\n"
    "        return np.load(cari_cache(b)).astype(np.float32)\n"
    "    m = timm.create_model(CFG.MHEAD[2], pretrained=True, pretrained_cfg_overlay={'file': str(unduh(*CFG.MHEAD[:2]))})\n"
    "    m = m.cuda().eval().half()\n"
    "    dl = DataLoader(Citra(X_train, tf_eval), batch_size=32, num_workers=CFG.NUM_WORKERS)\n"
    "    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):\n"
    "        p = np.concatenate([torch.softmax(m(x.cuda()).float(), 1).cpu().numpy() for x in dl])\n"
    "    np.save(CFG.CACHE_DIR / b, p.astype(np.float16))\n"
    "    del m\n"
    "    gc.collect()\n"
    "    torch.cuda.empty_cache()\n"
    "    return p\n"
    "\n"
    "\n"
    "P_HEAD = prob_head()\n"
    "INFO_IN = ImageNetInfo()\n"
    "NAMA_IN = [INFO_IN.index_to_description(i) for i in range(1000)]\n"
    "TOP1 = P_HEAD.argmax(1)\n"
    "print('kelas ImageNet top-1 terbanyak:', pd.Series([NAMA_IN[i].split(',')[0] for i in TOP1]).value_counts().head(8).to_dict())"
))

# ---------------------------------------------------------------------------
# 5. Perbandingan ruang
# ---------------------------------------------------------------------------
cells.append(section("Perbandingan Ruang Embedding", "5"))
cells.append(md(
    "Bagian ini menjawab apakah 7B benar-benar memberi cluster yang lebih baik daripada ViT-L. Resep fitur sama untuk "
    "keduanya: CLS dan rata-rata patch, masing-masing dirata-rata atas pandangan asli dan flip, dinormalisasi L2, "
    "digabung, lalu PCA 128 yang di-fit pada train saja.\n\n"
    "| Metrik | Arti | Bisa dibandingkan antar ruang? |\n"
    "|---|---|---|\n"
    "| `ari_boot` | ARI partisi subsample 80% terhadap partisi acuan, rata-rata 20 bootstrap | Ya |\n"
    "| `rapuh` | jumlah cluster acuan dengan Jaccard bootstrap < 0,6 | Ya |\n"
    "| `knn_top1` | porsi 10 tetangga terdekat yang berbagi kelas top-1 ImageNet (referensi independen) | Ya |\n"
    "| `ari_seed` | ARI antar 5 seed KMeans | Ya |\n"
    "| `silhouette` | silhouette cosine | Tidak, bergantung geometri ruang |\n\n"
    "Aturan beku, dievaluasi pada K = 48: 7B dipakai bila `ari_boot` 7B minimal 0,03 di atas ViT-L, `knn_top1` "
    "tidak lebih rendah, dan `rapuh` tidak lebih banyak. Bila salah satu gagal, ViT-L dipakai karena lebih murah. "
    "Baseline lokal ViT-L pada K = 48 (resep CLS satu pandangan, `temp/eda_208`): `ari_boot` 0,638, `rapuh` 18 dari 48."
))
cells.append(code(
    "def norml2(a):\n"
    "    return a / np.linalg.norm(a, axis=1, keepdims=True)\n"
    "\n"
    "\n"
    "def resep(e):\n"
    "    return norml2(np.hstack([norml2(e['asli_cls'] + e['flip_cls']), norml2(e['asli_patch'] + e['flip_patch'])]))\n"
    "\n"
    "\n"
    "def siapkan(x):\n"
    "    return norml2(PCA(CFG.PCA_DIM, random_state=CFG.SEED).fit_transform(x))\n"
    "\n"
    "\n"
    "def km(z, k, seed, n_init=4):\n"
    "    return KMeans(k, n_init=n_init, random_state=seed).fit(z)\n"
    "\n"
    "\n"
    "def jaccard_cluster(acuan, lab, idx):\n"
    "    out = []\n"
    "    for c in np.unique(acuan):\n"
    "        a = set(idx[acuan[idx] == c])\n"
    "        out.append(max(len(a & set(idx[lab[idx] == d])) / len(a | set(idx[lab[idx] == d])) for d in np.unique(lab[idx])))\n"
    "    return np.array(out)\n"
    "\n"
    "\n"
    "def metrik_ruang(z, k):\n"
    "    acuan = km(z, k, CFG.SEED).labels_\n"
    "    rng = np.random.default_rng(CFG.SEED)\n"
    "    ari_seed = np.mean([adjusted_rand_score(acuan, km(z, k, CFG.SEED + s).labels_) for s in range(1, 5)])\n"
    "    boot, jac = [], []\n"
    "    for b in range(CFG.N_BOOT):\n"
    "        idx = np.sort(rng.choice(len(z), int(.8 * len(z)), replace=False))\n"
    "        lab = km(z[idx], k, CFG.SEED + 100 + b).predict(z)\n"
    "        boot.append(adjusted_rand_score(acuan, lab))\n"
    "        jac.append(jaccard_cluster(acuan, lab, idx))\n"
    "    jac = np.mean(jac, 0)\n"
    "    S = z @ z.T\n"
    "    np.fill_diagonal(S, -np.inf)\n"
    "    nn10 = np.argsort(-S, 1)[:, :10]\n"
    "    uk = np.bincount(acuan)\n"
    "    return dict(k=k, silhouette=silhouette_score(z, acuan, metric='cosine'), ari_seed=ari_seed, ari_boot=np.mean(boot),\n"
    "                ari_boot_std=np.std(boot), jaccard_med=np.median(jac), rapuh=int((jac < .6).sum()),\n"
    "                knn_top1=np.mean(TOP1[nn10] == TOP1[:, None]), uk_min=int(uk.min()), uk_max=int(uk.max()))\n"
    "\n"
    "\n"
    "Z = {'vitl': siapkan(resep(E_VITL)), '7b': siapkan(resep(E_7B))}\n"
    "banding = pd.DataFrame([dict(ruang=r, **metrik_ruang(z, CFG.K_BANDING)) for r, z in Z.items()]).set_index('ruang')\n"
    "display(banding.round(3))\n"
    "a, b = banding.loc['7b'], banding.loc['vitl']\n"
    "syarat = {'ari_boot naik >= 0,03': a.ari_boot - b.ari_boot >= CFG.SELISIH_ARI_7B,\n"
    "          'knn_top1 tidak turun': a.knn_top1 >= b.knn_top1, 'rapuh tidak bertambah': a.rapuh <= b.rapuh}\n"
    "PAKAI = '7b' if all(syarat.values()) else 'vitl'\n"
    "print(syarat)\n"
    "print('ARI partisi K=48 antara 7B dan ViT-L:',\n"
    "      round(adjusted_rand_score(km(Z['7b'], CFG.K_BANDING, CFG.SEED).labels_, km(Z['vitl'], CFG.K_BANDING, CFG.SEED).labels_), 3))\n"
    "lama = cari_cache('label_lama.csv')\n"
    "if lama:\n"
    "    lab_lama = pd.read_csv(lama).sort_values('image_id').label.values\n"
    "    for r, z in Z.items():\n"
    "        print(f'ARI {r} K=48 vs label lama (melingkar untuk ViT-L):',\n"
    "              round(adjusted_rand_score(lab_lama, km(z, CFG.K_BANDING, CFG.SEED).labels_), 3))\n"
    "print('RUANG DIPAKAI:', PAKAI)"
))
cells.append(md(
    "#### Insights\n\n"
    "> Isi setelah run: tulis angka `ari_boot`, `rapuh`, dan `knn_top1` kedua ruang beserta ruang yang terpilih. "
    "Bila 7B tidak lolos, ekstraksi 7B tetap tercatat di notebook sebagai bukti bahwa keputusan diambil dari metrik."
))

# ---------------------------------------------------------------------------
# 6. Clustering
# ---------------------------------------------------------------------------
cells.append(section("Clustering Train", "6"))
cells.append(md(
    "## Pemilihan K dan Partisi Acuan\n\n"
    "Train sengaja di-overcluster (K lebih besar dari 8) supaya tiap cluster cukup murni untuk diberi satu nama. "
    "K dipilih dari {32, 48, 64} dengan `ari_boot` tertinggi, dengan syarat cluster terkecil minimal 10 citra. Partisi "
    "acuan memakai KMeans `n_init=20`, lalu nomor cluster diurutkan menurut ID citra terkecil di dalamnya, sehingga "
    "nomornya tidak bergantung pada urutan label internal KMeans. Partisi disimpan ke `cluster_train.csv`. Bila cache "
    "ada, partisi dimuat dari cache supaya nama cluster di bagian 7 tetap menunjuk citra yang sama."
))
cells.append(code(
    "def cluster_acuan(z, k):\n"
    "    lab = km(z, k, CFG.SEED, n_init=20).labels_\n"
    "    urut = np.argsort([np.flatnonzero(lab == c).min() for c in range(k)])\n"
    "    peta = np.empty(k, int)\n"
    "    peta[urut] = np.arange(k)\n"
    "    return peta[lab]\n"
    "\n"
    "\n"
    "def peta_bootstrap(z, acuan, k):\n"
    "    rng = np.random.default_rng(CFG.SEED + 7)\n"
    "    out = []\n"
    "    for b in range(CFG.N_BOOT):\n"
    "        idx = np.sort(rng.choice(len(z), int(.8 * len(z)), replace=False))\n"
    "        lab = km(z[idx], k, CFG.SEED + 200 + b).predict(z)\n"
    "        out.append(pd.crosstab(lab, acuan).idxmax(axis=1).loc[lab].values)\n"
    "    return np.array(out)\n"
    "\n"
    "\n"
    "ZP = Z[PAKAI]\n"
    "cache_cl = cari_cache('cluster_train.csv')\n"
    "if CFG.PAKAI_CACHE and cache_cl:\n"
    "    CL = pd.read_csv(cache_cl)\n"
    "    assert CL.ruang.iloc[0] == PAKAI, f'cache cluster dari ruang {CL.ruang.iloc[0]}, run ini memilih {PAKAI}'\n"
    "    K = int(CL.cluster.max() + 1)\n"
    "    BOOT = np.load(cari_cache('boot_train.npy'))\n"
    "    print('partisi dimuat dari cache; ARI dengan hitung ulang:', round(adjusted_rand_score(CL.cluster, cluster_acuan(ZP, K)), 4))\n"
    "else:\n"
    "    kandidat = pd.DataFrame([metrik_ruang(ZP, k) for k in CFG.K_KANDIDAT]).set_index('k')\n"
    "    display(kandidat.round(3))\n"
    "    lolos = kandidat[kandidat.uk_min >= CFG.UK_MIN_CLUSTER]\n"
    "    K = int(lolos.ari_boot.idxmax()) if len(lolos) else CFG.K_BANDING\n"
    "    acuan = cluster_acuan(ZP, K)\n"
    "    BOOT = peta_bootstrap(ZP, acuan, K)\n"
    "    CL = pd.DataFrame({'image_id': id_train, 'cluster': acuan, 'stab': (BOOT == acuan).mean(0), 'ruang': PAKAI})\n"
    "    CL.to_csv(CFG.CACHE_DIR / 'cluster_train.csv', index=False)\n"
    "    np.save(CFG.CACHE_DIR / 'boot_train.npy', BOOT)\n"
    "print(f'K = {K} | citra stabil di level cluster (stab >= {CFG.STAB_MIN}): {(CL.stab >= CFG.STAB_MIN).mean():.3f}')\n"
    "fig, ax = plt.subplots(1, 2, figsize=(13, 3.5))\n"
    "CL.cluster.value_counts().sort_index().plot.bar(ax=ax[0], fontsize=6)\n"
    "ax[0].set(title='ukuran cluster', xlabel='cluster')\n"
    "ax[1].hist(CL.stab, 20)\n"
    "ax[1].axvline(CFG.STAB_MIN, c='k', ls='--')\n"
    "ax[1].set(title='stabilitas per citra (bootstrap)')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(md(
    "#### Insights\n\n"
    "> Pada ViT-L lokal, sekitar separuh cluster rapuh untuk K 32 sampai 64 (Jaccard bootstrap < 0,6), artinya batas "
    "cluster bergeser antar run. Matriks pemetaan 20 bootstrap disimpan (`boot_train.npy`). Setelah penamaan, label "
    "hanya diberikan pada citra yang jatuh ke cluster bernama sama pada minimal 70% bootstrap (`stab_nama`). Citra "
    "yang hanya berpindah di antara dua cluster bernama sama tetap berlabel. Ukuran cluster ViT-L K = 48 "
    "berkisar 15 sampai 83 citra (rasio 5,5), jadi imbalance antar cluster jauh lebih besar daripada imbalance antar "
    "kelas pada label lama (rasio 1,49)."
))

cells.append(md(
    "## Grup Near-duplicate\n\n"
    "Grup untuk validasi dibentuk dari train saja dengan union-find: pasangan pHash dihedral Hamming 4 atau kurang, "
    "atau pasangan dengan cosine embedding tertinggi. Ambang cosine diambil dari kuantil supaya kepadatan pasangannya "
    "sama dengan cos 0,92 pada ViT-L di EDA lokal (0,0721% dari semua pasangan). Alasannya, nilai cosine mentah tidak "
    "setara antar ruang embedding."
))
cells.append(code(
    "EG = resep(E_7B if PAKAI == '7b' else E_VITL)\n"
    "C = EG @ EG.T\n"
    "ambang_cos = np.sort(C[iu])[::-1][int(CFG.PORSI_PASANGAN * len(iu[0]))]\n"
    "_, GRUP = connected_components(csr_matrix((HAM <= CFG.H_MAX) | (C >= ambang_cos)), directed=False)\n"
    "uk_grup = np.bincount(GRUP)\n"
    "CL['grup'] = GRUP\n"
    "print(f'ambang cos {ambang_cos:.4f} | {len(uk_grup)} grup | terbesar {uk_grup.max()} | citra dalam grup > 1: {int((uk_grup[GRUP] > 1).sum())}')\n"
    "del C"
))
cells.append(md(
    "#### Insights\n\n"
    "> Di EDA lokal, gabungan Hamming 4 dan cos 0,92 menghasilkan 1.471 grup dengan grup terbesar 52 citra (2,7% "
    "data), dan 628 citra berada dalam grup berisi lebih dari satu. Grup sebesar ini masih bisa dibagi rata oleh "
    "StratifiedGroupKFold 5 fold tanpa merusak stratifikasi."
))

# ---------------------------------------------------------------------------
# 7. Penamaan cluster
# ---------------------------------------------------------------------------
cells.append(section("Penamaan Cluster", "7"))
cells.append(md(
    "Penamaan dilakukan manual oleh tim, sesuai jawaban resmi panitia. Untuk tiap cluster, notebook menampilkan "
    "montage berisi 20 citra inti (stabilitas tertinggi, diambil acak) dan 6 citra paling pinggir (bingkai merah), "
    "serta tabel ukuran, stabilitas rata-rata, dan top-5 kelas ImageNet. Skor head hanya bukti pendukung. Keputusan "
    "nama ada pada tim, dan cluster campuran boleh dipecah atau ditandai `campur`. Asisten AI yang membantu menulis "
    "kode hanya melihat grid preprocessing untuk QA, dan tidak memberi label maupun nama cluster."
))
cells.append(md(
    "## Bukti Visual per Cluster\n\n"
    "Montage juga disimpan sebagai `gambar/cluster_<kode>.png` supaya bisa diperiksa di luar notebook."
))
cells.append(code(
    "def urut_kode(kode):\n"
    "    return sorted(set(kode), key=lambda s: (int(''.join(c for c in s if c.isdigit())), s))\n"
    "\n"
    "\n"
    "def top_in(idx, n=5):\n"
    "    p = P_HEAD[idx].mean(0)\n"
    "    return ', '.join(f\"{NAMA_IN[i].split(',')[0]} {p[i]:.2f}\" for i in np.argsort(-p)[:n])\n"
    "\n"
    "\n"
    "def montage(kd, inti, pinggir):\n"
    "    idx = list(inti) + list(pinggir)\n"
    "    fig, ax = plt.subplots(2, 13, figsize=(16, 2.9))\n"
    "    for a in ax.ravel():\n"
    "        a.axis('off')\n"
    "    for j, (a, i) in enumerate(zip(ax.ravel(), idx)):\n"
    "        a.imshow(cv2.resize(X_train[i], (112, 112), interpolation=cv2.INTER_AREA))\n"
    "        a.set_title(f'{id_train[i]} {CL.stab[i]:.2f}', fontsize=6, color='r' if j >= len(inti) else 'k')\n"
    "    fig.suptitle(f'cluster {kd} (n={int((CL.kode == kd).sum())}) | merah = paling pinggir', fontsize=9)\n"
    "    fig.tight_layout()\n"
    "    fig.savefig(CFG.FIG_DIR / f'cluster_{kd}.png', dpi=80)\n"
    "    plt.show()\n"
    "\n"
    "\n"
    "def bukti(daftar_kode):\n"
    "    rows = []\n"
    "    rng = np.random.default_rng(CFG.SEED)\n"
    "    for kd in daftar_kode:\n"
    "        d = CL[CL.kode == kd].sort_values('stab', ascending=False)\n"
    "        inti = rng.permutation(d.index.values[:max(20, len(d) // 2)])[:20]\n"
    "        pinggir = d.index.values[::-1][:min(6, max(0, len(d) - len(inti)))]\n"
    "        montage(kd, inti, pinggir)\n"
    "        rows.append(dict(kode=kd, n=len(d), stab_mean=round(d.stab.mean(), 3), n_pinggir=int((d.stab < CFG.STAB_MIN).sum()),\n"
    "                         top5_imagenet=top_in(d.index.values)))\n"
    "    return pd.DataFrame(rows)\n"
    "\n"
    "\n"
    "CL['kode'] = CL.cluster.astype(str)\n"
    "BUKTI = bukti(urut_kode(CL.kode))\n"
    "BUKTI.to_csv(CFG.OUTPUT_DIR / 'bukti_cluster.csv', index=False)\n"
    "with pd.option_context('display.max_colwidth', 200, 'display.max_rows', 200):\n"
    "    display(BUKTI)"
))
cells.append(md(
    "## Pecah Cluster Campuran\n\n"
    "Bila montage sebuah cluster memuat dua material atau lebih, cluster itu bisa dipecah dengan KMeans lokal. "
    "Contoh: `PECAH = {12: 2}` memecah cluster 12 menjadi `12a` dan `12b`, lalu montage keduanya ditampilkan. "
    "Biarkan kosong bila tidak ada cluster yang perlu dipecah."
))
cells.append(code(
    "PECAH = {}\n"
    "\n"
    "CL['kode'] = CL.cluster.astype(str)\n"
    "for c, n in PECAH.items():\n"
    "    idx = CL.index[CL.cluster == c]\n"
    "    sub = KMeans(n, n_init=10, random_state=CFG.SEED).fit_predict(ZP[idx])\n"
    "    CL.loc[idx, 'kode'] = [f'{c}{chr(97 + s)}' for s in sub]\n"
    "if PECAH:\n"
    "    display(bukti(urut_kode(CL.kode[CL.cluster.isin(list(PECAH))])))\n"
    "print('templat NAMA_CLUSTER:')\n"
    "print('{' + ', '.join(f\"'{k}': ''\" for k in urut_kode(CL.kode)) + '}')"
))
cells.append(md(
    "## Nama Cluster dari Tim\n\n"
    "Isi `NAMA_CLUSTER` dengan salah satu nama kelas (`glass`, `paper`, `electronic`, `metal`, `plastic`, "
    "`cardboard`, `textile`, `foodwaste`) atau `campur` untuk setiap kode, berdasarkan montage di atas. Salin templat "
    "dari sel sebelumnya. Citra pada cluster `campur` tidak diberi label. Selama kamus ini belum lengkap, run "
    "tetap selesai normal: sel berikutnya mencetak pesan tahap 1, dan bagian 8 sampai 11 dilewati supaya cache "
    "serta montage tetap tersimpan sebagai output."
))
cells.append(code(
    "NAMA_CLUSTER = {\n"
    "}"
))
cells.append(code(
    "belum = [k for k in urut_kode(CL.kode) if k not in NAMA_CLUSTER]\n"
    "LABEL_SIAP = not belum\n"
    + jaga(
    "salah = {k: v for k, v in NAMA_CLUSTER.items() if v not in CFG.KELAS + ['campur']}\n"
    "assert not salah, f'nama tidak valid: {salah}'\n"
    "CL['nama'] = CL.kode.map(NAMA_CLUSTER)\n"
    "nama_cluster = CL.groupby('cluster').nama.agg(lambda s: s.mode()[0]).reindex(range(K)).to_numpy(dtype=object)\n"
    "# cluster yang dipecah (PECAH) bisa berisi beberapa nama; jatuh ke cluster sendiri tetap dihitung konsisten\n"
    "CL['stab_nama'] = ((nama_cluster[BOOT] == CL.nama.to_numpy(dtype=object)) | (BOOT == CL.cluster.values)).mean(0)\n"
    "CL['berlabel'] = CL.nama.isin(CFG.KELAS) & (CL.stab_nama >= CFG.STAB_MIN)\n"
    "CL.to_csv(CFG.OUTPUT_DIR / 'label_train.csv', index=False)\n"
    "L = np.flatnonzero(CL.berlabel.values)\n"
    "Y = np.array([CFG.KELAS.index(n) for n in CL.nama.values[L]])\n"
    "print(f'berlabel {len(L)} dari 1.920 | campur {int((CL.nama == \"campur\").sum())} | ambigu antar nama {int((CL.stab_nama < CFG.STAB_MIN).sum())}')\n"
    "display(pd.Series(Y).map(dict(enumerate(CFG.KELAS))).value_counts().rename('n').to_frame().T)\n"
    "\n"
    "rng = np.random.default_rng(CFG.SEED)\n"
    "fig, ax = plt.subplots(8, 10, figsize=(13, 11))\n"
    "for r, kls in enumerate(CFG.KELAS):\n"
    "    pilih = rng.permutation(L[Y == r])[:10]\n"
    "    for c in range(10):\n"
    "        ax[r, c].axis('off')\n"
    "        if c < len(pilih):\n"
    "            ax[r, c].imshow(cv2.resize(X_train[pilih[c]], (96, 96), interpolation=cv2.INTER_AREA))\n"
    "    ax[r, 0].set_title(kls, fontsize=8, loc='left')\n"
    "plt.tight_layout()\n"
    "plt.show()",
        "TAHAP 1 SELESAI: NAMA_CLUSTER belum diisi. Lihat gambar/cluster_*.png dan bukti_cluster.csv, isi nama, lalu jalankan ulang dengan cache.")
))
cells.append(md(
    "#### Insights\n\n"
    "> Isi setelah penamaan: jumlah citra berlabel per kelas, cluster yang ditandai `campur` beserta alasannya, dan "
    "cluster yang nama dari montage berbeda dari top-5 ImageNet."
))

# ---------------------------------------------------------------------------
# 8. Training
# ---------------------------------------------------------------------------
cells.append(section("Training Classifier", "8"))
cells.append(md(
    "Classifier final adalah ConvNeXt-L DINOv3 yang di-fine-tune pada piksel, bukan head di atas embedding 7B. Ada "
    "tiga alasan. Pertama, head linear pada ruang yang sama dengan ruang clustering hanya mereproduksi batas cluster, "
    "termasuk kesalahannya, dan skor validasinya melingkar. Kedua, uji lama menunjukkan fine-tune menambah informasi "
    "di atas head linear. Ketiga, inference test tidak membutuhkan 7B sehingga notebook final lebih ringan.\n\n"
    "Resep mengikuti EXP-010 yang sudah teruji di data ini: AdamW, LR 1e-4 dengan pengali head 10, layer decay 0,75, "
    "weight decay 0,05, warmup 1 epoch lalu cosine, label smoothing 0,1, drop path 0,1, bobot kelas akar-invers "
    "frekuensi, dan 8 epoch. Karena T4 tidak punya bf16, training memakai AMP fp16 dengan GradScaler. Batch 16 "
    "dibagi ke dua GPU lewat DataParallel."
))
cells.append(code(
    "def buat_cnn():\n"
    "    return timm.create_model(CFG.MCNN[2], pretrained=True, num_classes=len(CFG.KELAS), drop_path_rate=CFG.DROP_PATH,\n"
    "                             pretrained_cfg_overlay={'file': str(unduh(*CFG.MCNN[:2]))})\n"
    "\n"
    "\n"
    "def bungkus(model):\n"
    "    return nn.DataParallel(model) if torch.cuda.device_count() > 1 else model\n"
    "\n"
    "\n"
    "def bobot_kelas(y):\n"
    "    w = 1 / np.sqrt(np.bincount(y, minlength=len(CFG.KELAS)).clip(1))\n"
    "    return torch.tensor(w / w.mean(), dtype=torch.float32)\n"
    "\n"
    "\n"
    "@torch.no_grad()\n"
    "def prediksi(net, X, idx=None):\n"
    "    net.eval()\n"
    "    dl = DataLoader(Citra(X, tf_eval, idx), batch_size=32, num_workers=CFG.NUM_WORKERS)\n"
    "    out = []\n"
    "    with torch.autocast('cuda', dtype=torch.float16):\n"
    "        for x in dl:\n"
    "            out.append(torch.softmax(net(x.cuda().to(memory_format=torch.channels_last)).float(), 1).cpu())\n"
    "    return torch.cat(out).numpy()\n"
    "\n"
    "\n"
    "def latih(idx_tr, y_tr, idx_va=None, y_va=None):\n"
    "    seed_everything(CFG.SEED)\n"
    "    model = buat_cnn().cuda().to(memory_format=torch.channels_last)\n"
    "    opt = create_optimizer_v2(model, opt='adamw', lr=CFG.LR, weight_decay=CFG.WD, layer_decay=CFG.LAYER_DECAY)\n"
    "    puncak = max(g.get('lr_scale', 1.0) for g in opt.param_groups)\n"
    "    for g in opt.param_groups:\n"
    "        g['lr_scale'] = g.get('lr_scale', 1.0) * (CFG.HEAD_MULT if g.get('lr_scale', 1.0) >= puncak else 1.0)\n"
    "    net = bungkus(model)\n"
    "    dl = DataLoader(Citra(X_train, tf_train, idx_tr, y_tr), batch_size=CFG.BS, shuffle=True, drop_last=True,\n"
    "                    num_workers=CFG.NUM_WORKERS, worker_init_fn=seed_worker,\n"
    "                    generator=torch.Generator().manual_seed(CFG.SEED), persistent_workers=True)\n"
    "    total, warm = len(dl) * CFG.EPOCH, len(dl) * CFG.WARMUP\n"
    "    kriteria = nn.CrossEntropyLoss(weight=bobot_kelas(y_tr).cuda(), label_smoothing=CFG.SMOOTH)\n"
    "    scaler = torch.amp.GradScaler('cuda')\n"
    "    riwayat, langkah, pv = [], 0, None\n"
    "    for ep in range(CFG.EPOCH):\n"
    "        net.train()\n"
    "        rugi = []\n"
    "        t0 = time.time()\n"
    "        for x, y in dl:\n"
    "            if langkah < warm:\n"
    "                f = langkah / warm\n"
    "            else:\n"
    "                f = 0.5 * (1 + math.cos(math.pi * (langkah - warm) / max(1, total - warm)))\n"
    "            for g in opt.param_groups:\n"
    "                g['lr'] = CFG.LR * g['lr_scale'] * f\n"
    "            with torch.autocast('cuda', dtype=torch.float16):\n"
    "                loss = kriteria(net(x.cuda(non_blocking=True).to(memory_format=torch.channels_last)), y.cuda())\n"
    "            opt.zero_grad(set_to_none=True)\n"
    "            scaler.scale(loss).backward()\n"
    "            scaler.step(opt)\n"
    "            scaler.update()\n"
    "            rugi.append(loss.item())\n"
    "            langkah += 1\n"
    "        r = dict(epoch=ep + 1, loss=round(float(np.mean(rugi)), 4), detik=round(time.time() - t0))\n"
    "        if idx_va is not None:\n"
    "            pv = prediksi(net, X_train, idx_va)\n"
    "            r['val_acc'] = round(float((pv.argmax(1) == y_va).mean()), 4)\n"
    "        riwayat.append(r)\n"
    "        print(r, flush=True)\n"
    "    return model, riwayat, pv"
))
cells.append(md(
    "## Cross-validation\n\n"
    "StratifiedGroupKFold 5 fold memakai grup near-duplicate, sehingga salinan sebuah citra tidak muncul di train dan "
    "validasi sekaligus. Hasil OOF disimpan ke `cache/cv_oof.npz`. Cache dipakai ulang hanya bila himpunan label sama "
    "persis dan `ULANG_CV = False`."
))
cells.append(code(jaga(
    "sgkf = StratifiedGroupKFold(n_splits=CFG.N_FOLD, shuffle=True, random_state=CFG.SEED)\n"
    "FOLD = np.full(len(L), -1)\n"
    "for f, (_, b) in enumerate(sgkf.split(L, Y, GRUP[L])):\n"
    "    FOLD[b] = f\n"
    "cache_cv = cari_cache('cv_oof.npz')\n"
    "cv = np.load(cache_cv) if cache_cv and not CFG.ULANG_CV else None\n"
    "if cv is not None and np.array_equal(cv['L'], L) and np.array_equal(cv['Y'], Y) and np.array_equal(cv['fold'], FOLD):\n"
    "    OOF, RIWAYAT = cv['oof'], json.loads(str(cv['riwayat']))\n"
    "    print('OOF dimuat dari cache')\n"
    "else:\n"
    "    OOF, RIWAYAT = np.zeros((len(L), len(CFG.KELAS))), []\n"
    "    for f in range(CFG.N_FOLD):\n"
    "        a, b = np.flatnonzero(FOLD != f), np.flatnonzero(FOLD == f)\n"
    "        print(f'fold {f}: latih {len(a)}, validasi {len(b)}')\n"
    "        model, riw, pv = latih(L[a], Y[a], L[b], Y[b])\n"
    "        OOF[b] = pv\n"
    "        RIWAYAT += [dict(fold=f, **r) for r in riw]\n"
    "        del model\n"
    "        gc.collect()\n"
    "        torch.cuda.empty_cache()\n"
    "    np.savez(CFG.CACHE_DIR / 'cv_oof.npz', oof=OOF, L=L, Y=Y, fold=FOLD, riwayat=json.dumps(RIWAYAT))"
)))

# ---------------------------------------------------------------------------
# 9. Validasi lokal
# ---------------------------------------------------------------------------
cells.append(section("Validasi Lokal", "9"))
cells.append(md(
    "Accuracy di sini adalah kesepakatan model dengan label cluster yang dinamai tim, bukan accuracy terhadap ground "
    "truth. Simpangan baku antar fold menunjukkan seberapa besar skor bisa bergeser hanya karena sampel yang berbeda. "
    "Sebagai pembanding, galat baku binomial pada accuracy 0,95 dengan 145 citra (public) sekitar 1,8 poin persen, "
    "dan dengan 335 citra (private) sekitar 1,2 poin persen."
))
cells.append(code(jaga(
    "acc = np.array([(OOF[FOLD == f].argmax(1) == Y[FOLD == f]).mean() for f in range(CFG.N_FOLD)])\n"
    "print('accuracy per fold:', acc.round(4))\n"
    "print(f'accuracy CV: {acc.mean():.4f} +- {acc.std(ddof=1):.4f} (mean +- std antar fold)')\n"
    "for n in (145, 335):\n"
    "    print(f'galat baku binomial pada n={n}: {math.sqrt(acc.mean() * (1 - acc.mean()) / n):.4f}')\n"
    "pred = OOF.argmax(1)\n"
    "cm = confusion_matrix(Y, pred, labels=range(len(CFG.KELAS)))\n"
    "recall = pd.Series(cm.diagonal() / cm.sum(1), CFG.KELAS).round(3)\n"
    "display(recall.to_frame('recall OOF').T)\n"
    "riw = pd.DataFrame(RIWAYAT)\n"
    "fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))\n"
    "im = ax[0].imshow(cm, cmap='Blues')\n"
    "ax[0].set_xticks(range(8), CFG.KELAS, rotation=45, fontsize=7)\n"
    "ax[0].set_yticks(range(8), CFG.KELAS, fontsize=7)\n"
    "for i in range(8):\n"
    "    for j in range(8):\n"
    "        ax[0].text(j, i, cm[i, j], ha='center', va='center', fontsize=7)\n"
    "ax[0].set(xlabel='prediksi OOF', ylabel='label cluster', title='confusion matrix OOF')\n"
    "for f, d in riw.groupby('fold'):\n"
    "    ax[1].plot(d.epoch, d.val_acc, marker='o', label=f'fold {f}')\n"
    "ax[1].set(xlabel='epoch', ylabel='accuracy validasi', title='kurva validasi per fold')\n"
    "ax[1].legend()\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "beda = CL.iloc[L].assign(pred=np.array(CFG.KELAS)[pred], benar=pred == Y)\n"
    "tinjau = beda.groupby('kode').agg(nama=('nama', 'first'), n=('benar', 'size'), setuju=('benar', 'mean'),\n"
    "                                  pred_lain=('pred', lambda s: s.value_counts().index[0]))\n"
    "display(tinjau[tinjau.setuju < .9].sort_values('setuju').round(3))"
)))
cells.append(md(
    "#### Insights\n\n"
    "> Isi setelah run: mean +- std accuracy per fold, kelas dengan recall terendah, dan cluster di tabel terakhir. "
    "Cluster dengan kesepakatan di bawah 0,9 adalah kandidat nama yang perlu ditinjau ulang lewat montage-nya. "
    "Penggantian nama tetap keputusan tim berdasarkan citra train. Bila std antar fold lebih besar dari sekitar 1,5 "
    "poin persen, selisih skor public dan private yang besar wajar terjadi."
))

# ---------------------------------------------------------------------------
# 10. Inference
# ---------------------------------------------------------------------------
cells.append(section("Inference Test", "10"))
cells.append(md(
    "Model final dilatih ulang pada seluruh citra berlabel dengan resep dan jumlah epoch yang sama dengan "
    "cross-validation. Model fold tidak dirata-rata karena itu termasuk ensemble. Test baru dimuat di sini, dengan "
    "preprocessing yang identik dengan validasi (stretch ke 384, tanpa TTA), lalu diprediksi satu kali."
))
cells.append(code(jaga(
    "MODEL, RIW_FINAL, _ = latih(L, Y)\n"
    "with ThreadPoolExecutor(8) as ex:\n"
    "    X_test = np.stack(list(ex.map(muat_citra, f_test)))\n"
    "P_TEST = prediksi(bungkus(MODEL), X_test)\n"
    "np.save(CFG.CACHE_DIR / 'prob_test.npy', P_TEST.astype(np.float16))\n"
    "print('distribusi prediksi test:', pd.Series(np.array(CFG.KELAS)[P_TEST.argmax(1)]).value_counts().to_dict())"
)))

# ---------------------------------------------------------------------------
# 11. Submission
# ---------------------------------------------------------------------------
cells.append(section("Submission", "11"))
cells.append(md(
    "Berkas submission mengikuti format `image_id,waste_category` dengan ID tiga digit, sesuai sample submission. "
    "Urutan baris disamakan dengan sample submission."
))
cells.append(code(jaga(
    "contoh = pd.read_csv(SUB_PATH, dtype=str)\n"
    "pred = pd.DataFrame({'image_id': [f'{i:03d}' for i in id_test], 'waste_category': np.array(CFG.KELAS)[P_TEST.argmax(1)]})\n"
    "assert set(pred.image_id) == set(contoh.image_id)\n"
    "submission = contoh[['image_id']].merge(pred, on='image_id', how='left')\n"
    "assert submission.waste_category.isin(CFG.KELAS).all() and len(submission) == 480\n"
    "submission.to_csv(CFG.OUTPUT_DIR / 'submission.csv', index=False)\n"
    "print(submission.waste_category.value_counts().to_dict())\n"
    "display(submission.head())"
)))

# ---------------------------------------------------------------------------
# Notebook writer
# ---------------------------------------------------------------------------
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
    },
    "cells": cells,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(cells)} cells -> {OUT}")
