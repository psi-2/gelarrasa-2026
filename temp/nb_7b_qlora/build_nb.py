"""Builder notebook final DSC Gelar Rasa 2026: DINOv3 ViT-7B + LoRA (QLoRA) untuk Jupyter Vast.ai.

    .venv/Scripts/python.exe temp/nb_7b_qlora/build_nb.py

Output: notebook/dsc_gelarrasa_2026_dinov3_7b_qlora.ipynb. Notebook TIDAK dijalankan di lokal.
Target: 1 GPU Blackwell (RTX 5090 32 GB -> NF4, RTX PRO 6000 96 GB -> bf16), satu submission dari satu model.
Label train tidak dibuat ulang: results/label_train.csv dari notebook v2 (temp/nb_7b/build_nb.py).
"""
import hashlib
import json
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AKAR = Path(__file__).resolve().parents[2]
OUT = AKAR / "notebook" / "dsc_gelarrasa_2026_dinov3_7b_qlora.ipynb"
HR = "---"
PREP = json.loads((AKAR / "temp/prep_302_verifikasi_run1/output/prep_final.json").read_text())["final"]
_LABEL = pd.read_csv(AKAR / "results/label_train.csv")
LABEL_SHA = hashlib.sha256(
    _LABEL[["image_id", "nama", "berlabel", "grup"]].to_csv(index=False, lineterminator="\n").encode()
).hexdigest()
assert int(_LABEL.berlabel.sum()) == 1750 and len(_LABEL) == 1920


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": uuid4().hex[:8], "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "id": uuid4().hex[:8],
        "metadata": {}, "outputs": [], "source": source,
    }


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
    "# DSC Gelar Rasa 2026: Klasifikasi Citra Sampah dengan DINOv3 ViT-7B\n\n"
    "*Label train dari cluster yang dinamai tim (notebook v2), lalu satu model DINOv3 ViT-7B yang di-fine-tune "
    "dengan LoRA di atas bobot beku terkuantisasi (QLoRA) untuk 8 kategori sampah.*"
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
    "3. [**Pemeriksaan GPU**](#3)\n"
    "4. [**Data dan Label**](#4)\n"
    "5. [**Model DINOv3 ViT-7B dengan LoRA**](#5)\n"
    "6. [**Training**](#6)\n"
    "7. [**Model Final dan Inference Test**](#7)\n"
    "8. [**Submission**](#8)\n"
    "9. [**Validasi Lokal**](#9)\n"
    "10. [**Pseudo-label Train (Opsional)**](#10)\n"
    "11. [**Ringkasan Run**](#11)\n"
))

# ---------------------------------------------------------------------------
# 1. Pendahuluan
# ---------------------------------------------------------------------------
cells.append(section("Pendahuluan", "1"))
cells.append(md(
    "## Overview\n\n"
    "*Kompetisi meminta klasifikasi citra sampah ke 8 kategori: glass, paper, electronic, metal, plastic, cardboard, "
    "textile, dan foodwaste. Data train berisi 1.920 citra tanpa label dan data test berisi 480 citra. Menurut QnA "
    "resmi panitia, label train diperoleh lewat clustering dengan model yang dijalankan sendiri, lalu cluster boleh "
    "dinamai secara manual.*"
))
cells.append(md(
    "## Tujuan\n\n"
    "*Notebook ini melatih satu classifier final di atas label train yang sudah final, lalu menulis tepat satu "
    "berkas `submission.csv` dari satu model. Clustering dan penamaan cluster tidak diulang di sini.*"
))
cells.append(md(
    "## Metrik\n\n"
    "***Accuracy** adalah porsi citra test yang kategorinya ditebak benar. Public leaderboard memuat sekitar 30% "
    "test (sekitar 145 citra), sehingga satu kesalahan bernilai sekitar 0,7 poin persen. Skor leaderboard tidak "
    "dipakai untuk keputusan apa pun di notebook ini.*\n\n"
    "$$\\text{Accuracy} = \\frac{\\text{jumlah prediksi benar}}{\\text{jumlah citra test}}$$"
))
cells.append(md(
    "## Dataset\n\n"
    "*Data berasal dari panitia DSC Gelar Rasa 2026 di Kaggle. Train berisi 1.920 citra dalam satu folder tanpa "
    "label, test berisi 480 citra. Ekstensi berkas campur, sehingga berkas dicari dari nama (stem) lalu diurutkan "
    "sebagai angka. Nama train empat digit (`0001`), nama test tiga digit (`001`).*\n\n"
    "*Data dipakai untuk melatih classifier pada train berlabel dan memprediksi test satu kali.*\n\n"
    "```\n"
    "data/\n"
    "+-- train/                     1.920 citra .jpg/.jpeg, tanpa label\n"
    "+-- test/                      480 citra .jpg/.jpeg/.png\n"
    "+-- SampleSubmission_DSC.csv   image_id (3 digit), waste_category\n"
    "results/\n"
    "+-- label_train.csv            label final dari notebook v2\n"
    "```"
))
cells.append(md(
    "## Asal Label\n\n"
    "*Label train tidak dibuat ulang di notebook ini. Label berasal dari notebook v2 "
    "(`notebook/dsc_gelarrasa_2026_dinov3.ipynb`, builder `temp/nb_7b/build_nb.py`, Kaggle "
    "`riantamaputra/dsc-gelarrasa-2026-dinov3` versi 4), yang menjalankan seluruh alur pelabelan pada train saja:*\n\n"
    "1. *Embedding train dari DINOv3 ViT-L dan DINOv3 ViT-7B. Aturan yang ditetapkan sebelum run memilih ruang ViT-L "
    "untuk clustering karena 7B tidak lebih stabil (ARI bootstrap 0,665 lawan 0,692 pada ViT-L).*\n"
    "2. *KMeans K = 48 pada PCA 128, stabilitas tiap citra dari 20 bootstrap, dan usulan pecah k = 2 per cluster.*\n"
    "3. *Tim menamai tiap cluster hanya dari montage citra. Enam cluster dinamai per sub-cluster karena berisi dua "
    "material: 5, 7, 23, 27, 44, dan 47. Nama tersimpan di `results/nama_cluster_templat.csv`. Tidak ada label tangan.*\n"
    "4. *Tiga aturan program hanya membuang citra dan tidak pernah memindah kelas: stabilitas nama bootstrap minimal "
    "0,7, minimal separuh dari 10 tetangga terdekat di ruang 7B bernama sama, dan tanpa konflik nama di dalam grup "
    "near-duplicate.*\n\n"
    "*Hasilnya `results/label_train.csv`: 1.750 dari 1.920 citra berlabel (189 sampai 261 per kelas), 51 citra "
    "bernama `campur`, dan 119 citra dibuang aturan. Kolom `grup` adalah grup near-duplicate (pHash dihedral dan "
    "cosine embedding) yang dipakai validasi. Isi berkas dikunci dengan hash SHA-256 di Settings, sehingga notebook "
    "berhenti bila label berbeda dari versi final. Label yang ada tidak diganti, karena QnA bertentangan soal "
    "penggantian label berbasis confidence model (#19.4 membolehkan, #21 melarang).*"
))
cells.append(md(
    "## Kepatuhan\n\n"
    "| Aturan (nomor pos QnA Padlet) | Penerapan di notebook |\n"
    "|---|---|\n"
    "| Tanpa VLM, termasuk image encoder saja (#4, #22.4, #23) | Hanya DINOv3, model self-supervised citra tanpa encoder teks |\n"
    "| Model pretrained wajib di-fine-tune (#5, #17) | Adapter LoRA di 40 blok dan head dilatih pada train panitia. Model 7B tidak pernah menebak test tanpa fine-tune |\n"
    "| Label dari clustering, nama cluster manual (#1, #2, #12, #14) | `label_train.csv` dari notebook v2, nama cluster diputuskan tim |\n"
    "| Buang citra salah cluster dengan aturan program (#19.3) | Tiga aturan buang di notebook v2, tanpa relabel |\n"
    "| Label tangan maksimal 5 per kelas, total 40 (#22.1) | Tidak ada label tangan |\n"
    "| Bantuan AI untuk pelabelan dilarang (#16, #20) | Asisten AI hanya memeriksa grid preprocessing setelah nama diputuskan tim. Tidak ada label yang diubah karena pemeriksaan itu |\n"
    "| Test hanya untuk prediksi akhir (#24) | Piksel test baru dibaca di bagian 7. Tidak ada fitting, normalisasi, atau pseudo-label dari test. Mean dan std memakai konstanta ImageNet bawaan bobot |\n"
    "| Tanpa dataset eksternal (#22.6) | Hanya data panitia. Bobot pretrained non-VLM diperbolehkan (#5) |\n"
    "| Pseudo-label hanya dari train (#22.3) | Opsional, hanya untuk 170 citra train tak berlabel |\n"
    "| Tanpa ensemble (batasan internal tim) | Satu model final, tanpa rata-rata fold dan tanpa TTA |"
))
cells.append(md(
    "## Pendekatan: QLoRA DINOv3 ViT-7B\n\n"
    "*DINOv3 ViT-7B/16 adalah vision transformer self-supervised berukuran 6,7 miliar parameter. Bobot pretrained "
    "versi timm (`timm/vit_7b_patch16_dinov3.lvd1689m`, revisi dikunci) menjadi base model dan di-fine-tune pada train "
    "panitia.*\n\n"
    "*Fine-tune penuh tidak muat di GPU yang tersedia. Bobot bf16 saja sudah 13,4 GB, sedangkan AdamW butuh salinan "
    "fp32, dua momen, dan gradien, sehingga totalnya di atas 100 GB sebelum aktivasi. Bobot terkuantisasi 4-bit juga "
    "tidak bisa diperbarui dengan gradien. Karena itu bobot 7B dibekukan, dalam NF4 4-bit di GPU 32 GB atau bf16 di GPU "
    "96 GB, dan yang dilatih hanya adapter LoRA rank 16 di kelima lapis linear setiap blok beserta head linear 8 kelas. "
    "Pola ini disebut QLoRA bila base model terkuantisasi.*\n\n"
    "```\n"
    "citra -> trim pita hitam -> stretch 512 x 512 -> augmentasi -> 384 x 384\n"
    "    |\n"
    "    +-- DINOv3 ViT-7B/16: 40 blok, dimensi 4096, bobot beku (NF4 atau bf16)\n"
    "    |     +-- LoRA r = 16 pada qkv, proj, fc1_g, fc1_x, fc2 di setiap blok (39,3 juta parameter latih)\n"
    "    |\n"
    "    +-- fitur: token CLS (4096) + rata-rata 576 patch token (4096) = 8192\n"
    "    |\n"
    "    Linear 8192 -> 8 kelas -> softmax\n"
    "```\n\n"
    "*Head diinisialisasi dari linear probe (regresi logistik pada fitur beku), lalu LoRA dan head dilatih bersama "
    "selama 4 epoch dengan AdamW, warmup lalu cosine, label smoothing 0,1, bobot kelas akar-invers, dan gradient "
    "checkpointing per blok. Semua hyperparameter ditetapkan sebelum run. Submission berasal dari satu model yang "
    "dilatih pada seluruh 1.750 citra berlabel, ditulis sebelum cross-validation. Model fold hanya untuk mengukur "
    "dan tidak dirata-rata.*"
))

# ---------------------------------------------------------------------------
# 2. Inisialisasi
# ---------------------------------------------------------------------------
cells.append(section("Inisialisasi", "2"))
cells.append(md(
    "## Environment Setup\n\n"
    "Notebook ditujukan untuk Jupyter di Vast.ai dengan satu GPU Blackwell: RTX 5090 (32 GB) atau RTX PRO 6000 "
    "(96 GB). Pilih template PyTorch dengan CUDA 12.8 atau lebih baru, karena arsitektur sm_120 butuh PyTorch 2.7 ke "
    "atas build cu128. Siapkan disk minimal 80 GB, karena bobot resmi 7B berukuran 26,9 GB (fp32) dan diunduh sekali "
    "ke cache Hugging Face. Clone repo ke instance, lalu buka notebook dari folder `notebook/`. Notebook mencari "
    "`data/SampleSubmission_DSC.csv` dan `results/label_train.csv` di folder kerja, folder induknya, atau "
    "`/workspace`. Keluaran ditulis ke `keluaran_7b/` di folder kerja, atau ke folder di variabel lingkungan "
    "`DSC_KELUARAN`.\n\n"
    "Token Hugging Face tidak wajib karena repo bobot timm tidak gated. Bila ada, token dibaca dari variabel "
    "lingkungan `HF_TOKEN` atau dari berkas `.env` di root repo (baris `HF_TOKEN=...`), dan nilainya tidak pernah "
    "dicetak. Perkiraan kasar di RTX 5090: unduh dan muat bobot 10 sampai 15 menit, fitur beku sekitar 3 menit, model "
    "final 4 epoch di 384 px sekitar 40 menit. Angka pastinya dicetak dari uji 1 batch di bagian 5. Bila kernel mati, "
    "jalankan ulang semua sel: training model final dilanjutkan dari checkpoint epoch terakhir, dan fold "
    "cross-validation yang selesai dimuat dari disk."
))
cells.append(code("!nvidia-smi"))
cells.append(md(
    "Sel berikut memasang pustaka yang belum ada di image PyTorch, dengan versi yang sudah diuji. timm dipasang tanpa "
    "dependensi supaya pip tidak mengganti torch atau torchvision bawaan image, yang sudah cocok dengan driver GPU."
))
cells.append(code(
    "%pip install -q --no-deps timm==1.0.29\n"
    "%pip install -q peft==0.21.3 bitsandbytes==0.50.2 accelerate==1.15.0 safetensors huggingface_hub pyyaml scikit-learn pandas matplotlib"
))

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
    "import platform\n"
    "import random\n"
    "import shutil\n"
    "import time\n"
    "from concurrent.futures import ThreadPoolExecutor\n"
    "from pathlib import Path\n"
    "\n"
    "# harus sebelum import torch: import bitsandbytes sudah menginisialisasi CUDA dan allocator-nya\n"
    "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')\n"
    "\n"
    "import bitsandbytes as bnb\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import peft\n"
    "import timm\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "from accelerate import init_empty_weights\n"
    "from accelerate.utils import set_module_tensor_to_device\n"
    "from huggingface_hub import hf_hub_download\n"
    "from huggingface_hub.constants import HF_HUB_CACHE\n"
    "from IPython.display import display\n"
    "from peft import LoraConfig, inject_adapter_in_model\n"
    "from PIL import Image, ImageOps\n"
    "from safetensors import safe_open\n"
    "from safetensors.torch import save_file\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.metrics import confusion_matrix, precision_recall_fscore_support\n"
    "from sklearn.model_selection import StratifiedGroupKFold\n"
    "from torch.utils.checkpoint import checkpoint\n"
    "from torch.utils.data import DataLoader, Dataset\n"
    "from torchvision.transforms import v2 as T\n"
    "\n"
    "print('torch', torch.__version__, '| timm', timm.__version__, '| peft', peft.__version__, '| bitsandbytes', bnb.__version__)"
))

cells.append(md("## Seed Everything"))
cells.append(code(
    "def seed_everything(seed: int = 42):\n"
    "    random.seed(seed)\n"
    "    os.environ['PYTHONHASHSEED'] = str(seed)\n"
    "    np.random.seed(seed)\n"
    "    torch.manual_seed(seed)\n"
    "    torch.cuda.manual_seed_all(seed)\n"
    "    # cuDNN deterministik membuat training notebook v2 sekitar 30x lebih lambat, jadi hanya seed yang dikunci\n"
    "    torch.backends.cudnn.benchmark = True\n"
    "    torch.backends.cudnn.deterministic = False\n"
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
    "Semua path, revisi bobot, hyperparameter, batas waktu, dan flag dikumpulkan di sini. Parameter preprocessing "
    "berasal dari verifikasi lokal `temp/prep_302_verifikasi_run1`. Hyperparameter training ditetapkan sebelum run "
    "dan tidak disetel terhadap skor apa pun. `MUAT = 'auto'` memakai bf16 bila VRAM minimal 60 GiB dan NF4 bila "
    "kurang. `UKURAN_CADANGAN` dan `BS_MIKRO` adalah urutan fallback bila 384 px atau batch 16 tidak muat di VRAM "
    "atau melewati `BATAS_FINAL_JAM`. Batch efektif tetap 16 lewat gradient accumulation. `STAB_MIN` dan `KNN_MIN` "
    "hanya dipakai untuk melaporkan alasan pembuangan label dari notebook v2."
))
cells.append(code(
    "def cari_berkas(nama):\n"
    "    for akar in (Path.cwd(), Path.cwd().parent, Path('/workspace')):\n"
    "        for pola in (nama, f'*/{nama}', f'*/*/{nama}'):\n"
    "            hit = sorted(akar.glob(pola))\n"
    "            if hit:\n"
    "                return hit[0].resolve()\n"
    "    raise FileNotFoundError(f'{nama} tidak ditemukan di folder kerja, folder induk, atau /workspace')\n"
    "\n"
    "\n"
    "def pasang_token():\n"
    "    if os.environ.get('HF_TOKEN'):\n"
    "        return 'variabel lingkungan'\n"
    "    for f in (Path.cwd() / '.env', Path.cwd().parent / '.env', Path('/workspace/.env')):\n"
    "        if f.is_file():\n"
    "            for baris in f.read_text(encoding='utf-8').splitlines():\n"
    "                kunci, _, nilai = baris.partition('=')\n"
    "                if kunci.strip().removeprefix('export ').strip() == 'HF_TOKEN' and nilai.strip():\n"
    "                    os.environ['HF_TOKEN'] = nilai.strip().strip('\\'\"')\n"
    "                    return f'berkas {f}'\n"
    "    return None\n"
    "\n"
    "\n"
    "class Settings:\n"
    "    SEED = 42\n"
    "    KELAS = ['glass', 'paper', 'electronic', 'metal', 'plastic', 'cardboard', 'textile', 'foodwaste']\n"
    "    SUB_PATH = cari_berkas('SampleSubmission_DSC.csv')\n"
    "    LABEL_PATH = cari_berkas('label_train.csv')\n"
    f"    LABEL_SHA256 = '{LABEL_SHA}'\n"
    "    OUTPUT_DIR = Path(os.environ.get('DSC_KELUARAN', Path.cwd() / 'keluaran_7b'))\n"
    "    FIG_DIR = OUTPUT_DIR / 'gambar'\n"
    "    STAB_MIN = 0.7\n"
    "    KNN_MIN = 0.5\n"
    "\n"
    "    M7B = ('timm/vit_7b_patch16_dinov3.lvd1689m', 'b40f6c827cb02b13c7a23a76cf598489bb0f94e5', 'vit_7b_patch16_dinov3')\n"
    "    DISK_MIN_GIB = 30\n"
    "    MUAT = 'auto'\n"
    "    VRAM_BF16_GIB = 60\n"
    "    LORA_R = 16\n"
    "    LORA_ALPHA = 32\n"
    "    LORA_DROPOUT = 0.05\n"
    "    LORA_TARGET = r'blocks\\.\\d+\\.(attn\\.(qkv|proj)|mlp\\.(fc1_g|fc1_x|fc2))'\n"
    "\n"
    "    UKURAN_SIMPAN = 512\n"
    f"    UKURAN = {PREP['ukuran']}\n"
    "    UKURAN_CADANGAN = (320, 256, 224)\n"
    "    MEAN = (0.485, 0.456, 0.406)\n"
    "    STD = (0.229, 0.224, 0.225)\n"
    f"    RRC_SCALE = ({PREP['rrc_scale'][0]}, 1.0)\n"
    f"    JITTER = ({PREP['color_jitter_bcsh'][0]}, {PREP['color_jitter_bcsh'][1]})\n"
    "    TRIM = (12, 4, 0.03)\n"
    "\n"
    "    BS = 16\n"
    "    BS_MIKRO = (16, 8, 4, 2)\n"
    "    EPOCH = 4\n"
    "    LR_LORA = 1e-4\n"
    "    LR_KEPALA = 1e-3\n"
    "    WD = 0.05\n"
    "    WARMUP = 0.1\n"
    "    SMOOTH = 0.1\n"
    "    LP_C = 0.1\n"
    "    LP_MIN = 0.8\n"
    "    VRAM_MAKS = 0.85\n"
    "    BATAS_FINAL_JAM = 1.5\n"
    "    JALANKAN_CV = True\n"
    "    N_FOLD = 5\n"
    "    MIN_FOLD = 2\n"
    "    BATAS_CV_JAM = 2.0\n"
    "    JALANKAN_PSEUDO = False\n"
    "    AMBANG_PSEUDO = 0.85\n"
    "    # /dev/shm bawaan Docker hanya 64 MB, terlalu kecil untuk batch yang dikirim worker DataLoader\n"
    "    SHM_GIB = shutil.disk_usage('/dev/shm').free / 2**30 if Path('/dev/shm').exists() else 0.0\n"
    "    NUM_WORKERS = min(8, os.cpu_count() or 1) if SHM_GIB >= 2 else 0\n"
    "\n"
    "\n"
    "CFG = Settings()\n"
    "for d in (CFG.OUTPUT_DIR, CFG.FIG_DIR):\n"
    "    d.mkdir(parents=True, exist_ok=True)\n"
    "sumber_token = pasang_token()\n"
    "print('data:', CFG.SUB_PATH.parent, '| label:', CFG.LABEL_PATH, '| keluaran:', CFG.OUTPUT_DIR)\n"
    "print(f'/dev/shm bebas {CFG.SHM_GIB:.1f} GiB -> worker DataLoader {CFG.NUM_WORKERS}')\n"
    "print('HF_TOKEN:', f'terpasang dari {sumber_token}' if sumber_token else 'tidak ada, repo timm tidak gated sehingga unduhan tetap jalan')"
))

cells.append(md(
    "## Load Dataset\n\n"
    "Berkas citra dikumpulkan dari nama (stem) lalu diurutkan sebagai angka karena ekstensinya campur. Label dibaca "
    "dari `label_train.csv`, diperiksa hash-nya, lalu diselaraskan dengan urutan berkas train. Bagian ini belum "
    "membaca piksel."
))
cells.append(code(
    "EKSTENSI = ('.jpg', '.jpeg', '.png')\n"
    "TRAIN_DIR = next(p for p in sorted(CFG.SUB_PATH.parent.rglob('train')) if p.is_dir())\n"
    "TEST_DIR = next(p for p in sorted(CFG.SUB_PATH.parent.rglob('test')) if p.is_dir())\n"
    "f_train = sorted((p for p in TRAIN_DIR.iterdir() if p.suffix.lower() in EKSTENSI), key=lambda p: int(p.stem))\n"
    "f_test = sorted((p for p in TEST_DIR.iterdir() if p.suffix.lower() in EKSTENSI), key=lambda p: int(p.stem))\n"
    "id_train = np.array([int(p.stem) for p in f_train])\n"
    "id_test = np.array([int(p.stem) for p in f_test])\n"
    "assert len(f_train) == 1920 and len(f_test) == 480, (len(f_train), len(f_test))\n"
    "assert len(set(id_train)) == 1920 and len(set(id_test)) == 480, 'stem ganda'\n"
    "\n"
    "LABEL = pd.read_csv(CFG.LABEL_PATH)\n"
    "kanon = LABEL[['image_id', 'nama', 'berlabel', 'grup']].to_csv(index=False, lineterminator='\\n')\n"
    "assert hashlib.sha256(kanon.encode()).hexdigest() == CFG.LABEL_SHA256, 'label_train.csv berbeda dari versi final notebook v2'\n"
    "LABEL = LABEL.set_index('image_id').loc[id_train].reset_index()\n"
    "BERLABEL = LABEL.berlabel.to_numpy(bool)\n"
    "L = np.flatnonzero(BERLABEL)\n"
    "U = np.flatnonzero(~BERLABEL)\n"
    "Y = np.array([CFG.KELAS.index(n) for n in LABEL.nama.to_numpy()[L]])\n"
    "GRUP = LABEL.grup.to_numpy()\n"
    "print(f'train {len(f_train)} | test {len(f_test)} | berlabel {len(L)} | tak berlabel {len(U)} | '\n"
    "      f'grup near-duplicate {len(np.unique(GRUP))}')"
))

# ---------------------------------------------------------------------------
# 3. Pemeriksaan GPU
# ---------------------------------------------------------------------------
cells.append(section("Pemeriksaan GPU", "3"))
cells.append(md(
    "Blackwell (RTX 5090 dan RTX PRO 6000, kapabilitas sm_120) hanya didukung PyTorch 2.7 ke atas build CUDA 12.8 dan "
    "bitsandbytes yang membawa kernel sm_120. Sel berikut mencetak versi semua komponen, memeriksa apakah build PyTorch "
    "memuat kernel untuk GPU ini, lalu menjalankan forward dan backward kecil untuk matmul bf16, scaled dot-product "
    "attention, dan lapis linear NF4. Bila salah satu gagal, sel berhenti dengan perintah perbaikannya. Mode bobot 7B "
    "dipilih dari VRAM: bf16 tanpa kuantisasi bila VRAM minimal 60 GiB, NF4 bila kurang."
))
cells.append(code(
    "assert torch.cuda.is_available(), 'CUDA tidak tersedia: pilih instance GPU dengan image PyTorch build CUDA'\n"
    "DEV = torch.device('cuda:0')\n"
    "prop = torch.cuda.get_device_properties(DEV)\n"
    "VRAM_GIB = prop.total_memory / 2**30\n"
    "SM = f'sm_{prop.major}{prop.minor}'\n"
    "INFO_GPU = {'gpu': prop.name, 'kapabilitas': SM, 'vram_gib': round(VRAM_GIB, 1), 'python': platform.python_version(),\n"
    "            'torch': torch.__version__, 'cuda_torch': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),\n"
    "            'bitsandbytes': bnb.__version__, 'timm': timm.__version__, 'peft': peft.__version__,\n"
    "            'arsitektur_build': ' '.join(torch.cuda.get_arch_list())}\n"
    "display(pd.Series(INFO_GPU, dtype=object).to_frame('nilai'))\n"
    "if SM not in torch.cuda.get_arch_list():\n"
    "    print(f'PERINGATAN: build PyTorch ini tidak memuat kernel {SM}. Untuk Blackwell jalankan '\n"
    "          '%pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128 lalu restart kernel.')\n"
    "\n"
    "\n"
    "def uji_kernel():\n"
    "    x = torch.randn(4, 64, 256, device=DEV, requires_grad=True)\n"
    "    lin = nn.Linear(256, 768).to(DEV)\n"
    "    with torch.autocast('cuda', dtype=torch.bfloat16):\n"
    "        q, k, v = lin(x).view(4, 64, 3, 4, 64).unbind(2)\n"
    "        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))\n"
    "    y.float().sum().backward()\n"
    "    q4 = bnb.nn.Linear4bit(256, 512, compute_dtype=torch.bfloat16, quant_type='nf4')\n"
    "    q4.load_state_dict(nn.Linear(256, 512).state_dict())\n"
    "    q4 = q4.to(DEV)\n"
    "    x4 = torch.randn(8, 256, device=DEV, requires_grad=True)\n"
    "    q4(x4).float().sum().backward()\n"
    "    torch.cuda.synchronize()\n"
    "    return bool(torch.isfinite(x.grad).all() and torch.isfinite(x4.grad).all())\n"
    "\n"
    "\n"
    "try:\n"
    "    print('uji forward + backward bf16, SDPA, dan NF4:', 'OK' if uji_kernel() else 'GAGAL, nilai non-finite')\n"
    "except Exception as e:\n"
    "    raise RuntimeError('Kernel GPU gagal. Blackwell butuh PyTorch >= 2.7 cu128 '\n"
    "                       '(%pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128) '\n"
    "                       'dan bitsandbytes >= 0.46, lalu restart kernel.') from e\n"
    "MUAT = CFG.MUAT if CFG.MUAT != 'auto' else ('bf16' if VRAM_GIB >= CFG.VRAM_BF16_GIB else 'nf4')\n"
    "bebas, total = torch.cuda.mem_get_info(DEV)\n"
    "print(f'mode bobot 7B: {MUAT} | VRAM bebas {bebas / 2**30:.1f} dari {total / 2**30:.1f} GiB')"
))

# ---------------------------------------------------------------------------
# 4. Data dan label
# ---------------------------------------------------------------------------
cells.append(section("Data dan Label", "4"))
cells.append(md(
    "Bagian ini memeriksa label final, memuat piksel train dengan preprocessing yang sama seperti notebook v2, lalu "
    "menampilkan seluruh citra train per kelas dan contoh augmentasi. Piksel test belum dibaca."
))
cells.append(md(
    "## Sebaran Label\n\n"
    "Tabel memisahkan citra berlabel, citra bernama `campur`, dan citra bernama yang dibuang aturan program notebook "
    "v2. Kolom `stab_nama`, `knn_nama`, dan `konflik_grup` berasal dari berkas label dan hanya dipakai untuk laporan."
))
cells.append(code(
    "status = np.where(BERLABEL, 'berlabel', np.where(LABEL.nama.eq('campur'), 'campur', 'dibuang aturan'))\n"
    "display(pd.crosstab(LABEL.nama, status, colnames=['status']).reindex(CFG.KELAS + ['campur']).fillna(0).astype(int).T)\n"
    "dibuang = LABEL[status == 'dibuang aturan']\n"
    "print(f'dibuang aturan {len(dibuang)}: stab_nama < {CFG.STAB_MIN} {int((dibuang.stab_nama < CFG.STAB_MIN).sum())}, '\n"
    "      f'knn_nama < {CFG.KNN_MIN} {int((dibuang.knn_nama < CFG.KNN_MIN).sum())}, konflik grup {int(dibuang.konflik_grup.sum())}')\n"
    "uk = pd.Series(GRUP).value_counts()\n"
    "print(f'grup near-duplicate: {len(uk)} grup, {int((uk > 1).sum())} grup berisi lebih dari 1 citra, terbesar {uk.max()}')\n"
    "n_kelas = np.bincount(Y, minlength=len(CFG.KELAS))\n"
    "fig, ax = plt.subplots(figsize=(8, 3))\n"
    "ax.bar(CFG.KELAS, n_kelas, color='C0')\n"
    "ax.set(title=f'citra berlabel per kelas (rasio maks/min {n_kelas.max() / n_kelas.min():.2f})', ylabel='jumlah')\n"
    "ax.tick_params(axis='x', labelrotation=30)\n"
    "plt.tight_layout()\n"
    "fig.savefig(CFG.FIG_DIR / 'sebaran_label.png', dpi=100)\n"
    "plt.show()"
))
cells.append(md(
    "#### Insights\n\n"
    "> Kelas cukup seimbang, 189 sampai 261 citra per kelas (rasio 1,38). Bobot kelas akar-invers frekuensi dipakai "
    "sebagai koreksi ringan di loss. Electronic paling banyak kehilangan citra karena beberapa cluster electronic "
    "saling bertukar anggota antar bootstrap.\n\n"
    "> Grup near-duplicate menyatukan salinan hasil flip, rotasi, atau unggah ulang. Validasi memakai grup ini "
    "supaya salinan sebuah citra tidak muncul di train dan validasi sekaligus."
))

cells.append(md(
    "## Preprocessing\n\n"
    "Setiap citra train dibuka sekali: orientasi EXIF diterapkan, mode dikonversi ke RGB (ada PNG mode P dan RGBA), "
    "pita hitam murni dari sumber dipotong, lalu di-stretch bicubic ke 512 x 512 dan disimpan di RAM (sekitar 1,5 "
    "GB). Fungsinya identik dengan notebook v2 dan `temp/prep_302_verifikasi_run1/pipa.py`. Letterbox ditolak karena "
    "pita hitamnya terkumpul di kelas electronic dan menjadi jalan pintas. Denoise juga tidak dipakai karena "
    "tekstur adalah isyarat material."
))
cells.append(code(
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
    "t0 = time.time()\n"
    "with ThreadPoolExecutor(8) as ex:\n"
    "    X_train = np.stack(list(ex.map(muat_citra, f_train)))\n"
    "print(f'X_train {X_train.shape}, {X_train.nbytes / 2**30:.2f} GiB, {time.time() - t0:.0f} detik')"
))

cells.append(md(
    "## Montage Seluruh Train per Kelas\n\n"
    "Montage memuat semua citra train setelah preprocessing, dikelompokkan menurut label final, ditambah satu montage "
    "untuk 170 citra tak berlabel. Montage ini adalah bukti visual label dan alat pemeriksaan preprocessing. Montage "
    "tidak dipakai untuk mengubah label. Salinannya disimpan di `gambar/montage_<kelas>.jpg`."
))
cells.append(code(
    "def montage(idx, judul, berkas, kol=24, uk=56):\n"
    "    baris = math.ceil(len(idx) / kol)\n"
    "    kanvas = Image.new('RGB', (kol * uk, baris * uk), (255, 255, 255))\n"
    "    for j, i in enumerate(idx):\n"
    "        kanvas.paste(Image.fromarray(X_train[i]).resize((uk, uk), Image.BILINEAR), (j % kol * uk, j // kol * uk))\n"
    "    kanvas.save(CFG.FIG_DIR / berkas, quality=85)\n"
    "    plt.figure(figsize=(kol * 0.55, baris * 0.55 + 0.5))\n"
    "    plt.imshow(kanvas)\n"
    "    plt.axis('off')\n"
    "    plt.title(judul, fontsize=10)\n"
    "    plt.show()\n"
    "\n"
    "\n"
    "for k, kls in enumerate(CFG.KELAS):\n"
    "    montage(L[Y == k], f'{kls}: {int((Y == k).sum())} citra berlabel', f'montage_{kls}.jpg')\n"
    "montage(U, f'tak berlabel: {len(U)} citra (campur dan dibuang aturan)', 'montage_tak_berlabel.jpg')"
))
cells.append(md(
    "#### Insights\n\n"
    "> Pemeriksaan lokal atas seluruh 1.920 citra (montage per kelas dan grid augmentasi di `temp/nb_7b_qlora/output`, "
    "dibuat dengan fungsi preprocessing yang sama) hanya menilai preprocessing dan dilakukan setelah nama cluster "
    "diputuskan tim. Orientasi EXIF benar, 123 citra yang pita hitamnya dipotong tidak menyisakan pita, dan objek "
    "pada citra beraspek ekstrem memanjang tetapi tetap utuh. Tidak ada citra yang dilabeli, dipindah, atau dibuang "
    "berdasarkan pemeriksaan ini, sesuai QnA #16 dan #20.\n\n"
    "> Watermark situs stok foto, bingkai bawaan sumber, dan sudut gelap melingkar dari lensa fisheye pada foto nampan "
    "makanan adalah sifat sumber, bukan hasil preprocessing. Trim hanya memotong baris atau kolom tepi yang hitam "
    "seragam, jadi sudut gelap fisheye tidak ikut terpotong. Sebanyak 105 citra bersisi pendek di bawah 200 px tampak "
    "buram setelah upscale, tetapi isinya tetap terbaca."
))

cells.append(md(
    "## Augmentasi\n\n"
    "Parameter augmentasi dipilih lewat iterasi lokal `prep_301` dan `prep_302` dengan kriteria yang ditetapkan "
    "sebelum pengukuran: RandomResizedCrop dengan luas minimum 0,5 (scale 0,4 membuang objek kecil di latar polos pada "
    "10,05% crop, di atas batas 10%), flip horizontal dan vertikal, rotasi kelipatan 90 derajat yang meniru salinan "
    "dihedral di data, serta jitter brightness 0,2 dan contrast 0,1 tanpa saturation dan hue supaya warna cardboard "
    "tidak bergeser ke paper. Rotasi kecil ditolak karena sudut isiannya mencolok. `transform` dibuat sebagai fungsi "
    "karena resolusi bisa turun lewat fallback di bagian 5. Grid menampilkan satu citra per kelas: asli, setelah trim "
    "dan stretch, lalu tiga hasil augmentasi train."
))
cells.append(code(
    "def transform(ukuran):\n"
    "    aug = [T.RandomResizedCrop(ukuran, scale=CFG.RRC_SCALE, ratio=(3 / 4, 4 / 3), antialias=True),\n"
    "           T.RandomHorizontalFlip(),\n"
    "           T.RandomVerticalFlip(),\n"
    "           T.RandomChoice([T.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),\n"
    "           T.ColorJitter(brightness=CFG.JITTER[0], contrast=CFG.JITTER[1])]\n"
    "    normal = [T.ToDtype(torch.float32, scale=True), T.Normalize(CFG.MEAN, CFG.STD)]\n"
    "    return (T.Compose([T.ToImage(), *aug, *normal]),\n"
    "            T.Compose([T.ToImage(), T.Resize((ukuran, ukuran), antialias=True), *normal]),\n"
    "            T.Compose([T.ToImage(), *aug]))\n"
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
    "        return x if self.y is None else (x, int(self.y[i]))\n"
    "\n"
    "\n"
    "UKURAN = CFG.UKURAN\n"
    "tf_train, tf_eval, tf_tampil = transform(UKURAN)\n"
    "rng = np.random.default_rng(CFG.SEED)\n"
    "fig, ax = plt.subplots(len(CFG.KELAS), 5, figsize=(10, 2.1 * len(CFG.KELAS)))\n"
    "for r, k in enumerate(range(len(CFG.KELAS))):\n"
    "    i = rng.choice(L[Y == k])\n"
    "    with Image.open(f_train[i]) as im:\n"
    "        ax[r, 0].imshow(ImageOps.exif_transpose(im).convert('RGB'))\n"
    "    ax[r, 0].set_title(f'{id_train[i]} asli ({CFG.KELAS[k]})', fontsize=7)\n"
    "    ax[r, 1].imshow(X_train[i])\n"
    "    ax[r, 1].set_title('trim + stretch 512', fontsize=7)\n"
    "    for c in range(2, 5):\n"
    "        ax[r, c].imshow(tf_tampil(X_train[i]).permute(1, 2, 0).numpy())\n"
    "        ax[r, c].set_title('augmentasi train', fontsize=7)\n"
    "for a in ax.ravel():\n"
    "    a.axis('off')\n"
    "plt.tight_layout()\n"
    "fig.savefig(CFG.FIG_DIR / 'augmentasi.jpg', dpi=80)\n"
    "plt.show()"
))
cells.append(md(
    "#### Insights\n\n"
    "> Augmentasi menjaga isyarat material: crop tidak lebih kecil dari 50% luas, warna kardus tidak dipudarkan lewat "
    "saturation, tekstur tidak dihaluskan, dan tidak ada sudut isian yang bisa menjadi artefak. Augmentasi kompresi "
    "JPEG dan penurunan resolusi tidak dipakai, karena sumber berkas yang hanya ada di train tidak muncul di test "
    "(EDA lokal 212). Evaluasi dan test memakai resize langsung ke 384 x 384 tanpa crop."
))

# ---------------------------------------------------------------------------
# 5. Model
# ---------------------------------------------------------------------------
cells.append(section("Model DINOv3 ViT-7B dengan LoRA", "5"))
cells.append(md(
    "Model dibangun dalam empat langkah: unduh bobot resmi timm dengan revisi terkunci, bangun arsitektur kosong lalu "
    "isi bobotnya tensor demi tensor (dikuantisasi NF4 atau dikonversi ke bf16 langsung di GPU), pasang LoRA di semua "
    "blok, lalu uji satu langkah training untuk memilih resolusi dan batch mikro yang muat. Struktur timm Eva dan "
    "pemuatan safetensors ini sama dengan yang sudah berjalan di Run 1 notebook v1 (606 tensor, 0 kunci asing)."
))
cells.append(md(
    "## Bobot Pretrained\n\n"
    "Bobot diunduh sekali ke cache Hugging Face dengan revisi dikunci. Bila sudah ada di cache, unduhan dilewati. Ruang "
    "disk diperiksa lebih dulu karena berkas fp32 berukuran 26,9 GB."
))
cells.append(code(
    "def berkas_bobot():\n"
    "    repo, rev, _ = CFG.M7B\n"
    "    try:\n"
    "        return Path(hf_hub_download(repo, 'model.safetensors', revision=rev, local_files_only=True))\n"
    "    except Exception:\n"
    "        Path(HF_HUB_CACHE).mkdir(parents=True, exist_ok=True)\n"
    "        bebas = shutil.disk_usage(HF_HUB_CACHE).free / 2**30\n"
    "        assert bebas >= CFG.DISK_MIN_GIB, f'ruang disk {bebas:.0f} GiB, butuh minimal {CFG.DISK_MIN_GIB} GiB untuk bobot fp32 26,9 GB'\n"
    "        return Path(hf_hub_download(repo, 'model.safetensors', revision=rev))\n"
    "\n"
    "\n"
    "t0 = time.time()\n"
    "BERKAS_7B = berkas_bobot()\n"
    "print(BERKAS_7B, f'{BERKAS_7B.stat().st_size / 1e9:.1f} GB, {time.time() - t0:.0f} detik')"
))

cells.append(md(
    "## Muat Bobot Terkuantisasi\n\n"
    "Arsitektur dibuat kosong dengan `init_empty_weights`. Pada mode NF4, kelima lapis linear di setiap blok (qkv, "
    "proj, fc1_g, fc1_x, fc2, total 200 lapis atau 99,9% dari 6,72 miliar parameter) diganti `bnb.nn.Linear4bit`, lalu setiap bobot "
    "dibaca dari safetensors, dikirim ke GPU, dan dikuantisasi NF4 dengan double quantization. Pada mode bf16, bobot "
    "yang sama hanya dikonversi ke bf16. Parameter kecil (LayerNorm, LayerScale, bias, token, patch embedding) "
    "disimpan dalam fp32 demi kestabilan. RAM tidak pernah menampung seluruh bobot fp32. Semua parameter base model "
    "dibekukan."
))
cells.append(code(
    "def muat_7b(berkas, mode):\n"
    "    with init_empty_weights():\n"
    "        m = timm.create_model(CFG.M7B[2], pretrained=False, num_classes=0)\n"
    "    linear = [n for n, mod in m.named_modules() if isinstance(mod, nn.Linear) and n.startswith('blocks.')]\n"
    "    if mode == 'nf4':\n"
    "        for n in linear:\n"
    "            lama = m.get_submodule(n)\n"
    "            induk, anak = n.rsplit('.', 1)\n"
    "            with init_empty_weights():\n"
    "                baru = bnb.nn.Linear4bit(lama.in_features, lama.out_features, bias=lama.bias is not None,\n"
    "                                         compute_dtype=torch.bfloat16, compress_statistics=True, quant_type='nf4')\n"
    "            setattr(m.get_submodule(induk), anak, baru)\n"
    "    besar = {f'{n}.weight' for n in linear}\n"
    "    param = dict(m.named_parameters())\n"
    "    terisi = set()\n"
    "    with safe_open(str(berkas), 'pt', device='cpu') as sf:\n"
    "        for k in sf.keys():\n"
    "            assert k in param, f'kunci asing di bobot: {k}'\n"
    "            t = sf.get_tensor(k).to(DEV)\n"
    "            if mode == 'nf4' and k in besar:\n"
    "                mod = m.get_submodule(k.rsplit('.', 1)[0])\n"
    "                mod.weight = bnb.nn.Params4bit(t.to(torch.bfloat16), requires_grad=False, compress_statistics=True,\n"
    "                                               quant_type='nf4', module=mod).to(DEV)\n"
    "            else:\n"
    "                set_module_tensor_to_device(m, k, DEV, value=t, dtype=torch.bfloat16 if k in besar else torch.float32)\n"
    "            terisi.add(k)\n"
    "            del t\n"
    "    kurang = set(param) - terisi\n"
    "    assert not kurang, f'parameter tanpa bobot: {sorted(kurang)[:5]}'\n"
    "    for n, b in m.named_buffers():\n"
    "        set_module_tensor_to_device(m, n, DEV, value=b)\n"
    "    m.requires_grad_(False)\n"
    "    return m, len(linear), len(terisi)\n"
    "\n"
    "\n"
    "torch.cuda.reset_peak_memory_stats(DEV)\n"
    "t0 = time.time()\n"
    "m, n_linear, n_tensor = muat_7b(BERKAS_7B, MUAT)\n"
    "DETIK_MUAT = time.time() - t0\n"
    "print(f'{CFG.M7B[2]} mode {MUAT}: {n_tensor} tensor, {n_linear} lapis linear, {len(m.blocks)} blok, '\n"
    "      f'{DETIK_MUAT:.0f} detik, VRAM {torch.cuda.memory_allocated(DEV) / 2**30:.1f} GiB')"
))

cells.append(md(
    "## Adapter LoRA dan Head\n\n"
    "LoRA rank 16 (alpha 32, dropout 0,05) dipasang lewat PEFT pada kelima lapis linear di setiap blok. Pola nama "
    "dibatasi ke `blocks.*` supaya konvolusi patch embedding tidak ikut. Parameter LoRA disimpan dalam fp32 sebagai "
    "master weight, sedangkan perhitungan berjalan dalam bf16 lewat autocast. Fitur adalah token CLS dan rata-rata "
    "patch token setelah LayerNorm akhir (4 token register tidak dipakai). Setiap blok dibungkus gradient "
    "checkpointing non-reentrant, sehingga yang disimpan hanya masukan blok dan aktivasi di dalam blok dihitung ulang "
    "saat backward. Salinan LoRA awal disimpan supaya setiap training (model final, fold, pseudo-label) mulai dari "
    "titik yang sama tanpa memuat ulang 7B."
))
cells.append(code(
    "torch.manual_seed(CFG.SEED)\n"
    "m = inject_adapter_in_model(LoraConfig(r=CFG.LORA_R, lora_alpha=CFG.LORA_ALPHA, lora_dropout=CFG.LORA_DROPOUT,\n"
    "                                       bias='none', target_modules=CFG.LORA_TARGET), m)\n"
    "for n, p in m.named_parameters():\n"
    "    p.requires_grad_('lora_' in n)\n"
    "    if 'lora_' in n:\n"
    "        p.data = p.data.float()\n"
    "LORA = {n: p for n, p in m.named_parameters() if p.requires_grad}\n"
    "LORA_AWAL = {n: p.detach().clone() for n, p in LORA.items()}\n"
    "kepala = nn.Linear(2 * m.embed_dim, len(CFG.KELAS)).to(DEV)\n"
    "n_modul = sum('lora_A' in n for n in LORA)\n"
    "assert n_modul == 5 * len(m.blocks), f'LoRA terpasang di {n_modul} lapis, seharusnya {5 * len(m.blocks)}'\n"
    "\n"
    "\n"
    "def fitur(m, x):\n"
    "    h = m.patch_embed(x)\n"
    "    h, rope = m._pos_embed(h)\n"
    "    h = m.norm_pre(h)\n"
    "    for blk in m.blocks:\n"
    "        h = checkpoint(blk, h, rope=rope, use_reentrant=False) if torch.is_grad_enabled() else blk(h, rope=rope)\n"
    "    h = m.norm(h)\n"
    "    return torch.cat([h[:, 0], h[:, m.num_prefix_tokens:].mean(1)], 1)\n"
    "\n"
    "\n"
    "n_latih = sum(p.numel() for p in LORA.values()) + sum(p.numel() for p in kepala.parameters())\n"
    "print(f'LoRA di {n_modul} lapis linear | parameter latih {n_latih / 1e6:.1f} juta | '\n"
    "      f'VRAM {torch.cuda.memory_allocated(DEV) / 2**30:.1f} GiB')"
))

cells.append(md(
    "## Uji Satu Langkah dan Fallback\n\n"
    "Satu langkah training (forward, backward, tanpa update) dijalankan pada input acak untuk mengukur waktu dan VRAM "
    "puncak. Urutannya: 384 px dengan batch mikro 16, 8, 4, lalu 2. Bila semua OOM atau perkiraan waktu model final "
    "melewati `BATAS_FINAL_JAM`, resolusi turun ke 320, 256, lalu 224. Konfigurasi dianggap muat bila VRAM puncak "
    "paling banyak 85% kapasitas, supaya fragmentasi selama training tidak memicu OOM. Batch efektif tetap 16 lewat "
    "gradient accumulation, sehingga resep training tidak berubah."
))
cells.append(code(
    "def uji_langkah(ukuran, bs):\n"
    "    x = torch.randn(bs, 3, ukuran, ukuran, device=DEV)\n"
    "    y = torch.zeros(bs, dtype=torch.long, device=DEV)\n"
    "    m.train()\n"
    "    torch.cuda.reset_peak_memory_stats(DEV)\n"
    "    waktu = []\n"
    "    for _ in range(3):\n"
    "        torch.cuda.synchronize()\n"
    "        t0 = time.time()\n"
    "        with torch.autocast('cuda', dtype=torch.bfloat16):\n"
    "            loss = F.cross_entropy(kepala(fitur(m, x)).float(), y)\n"
    "        loss.backward()\n"
    "        torch.cuda.synchronize()\n"
    "        waktu.append(time.time() - t0)\n"
    "    return min(waktu[1:]), torch.cuda.max_memory_allocated(DEV) / 2**30\n"
    "\n"
    "\n"
    "def coba(ukuran, bs):\n"
    "    try:\n"
    "        return uji_langkah(ukuran, bs)\n"
    "    except torch.cuda.OutOfMemoryError:\n"
    "        return None\n"
    "    finally:\n"
    "        for p in [*LORA.values(), *kepala.parameters()]:\n"
    "            p.grad = None\n"
    "        gc.collect()\n"
    "        torch.cuda.empty_cache()\n"
    "\n"
    "\n"
    "def pilih_konfigurasi():\n"
    "    for ukuran in (CFG.UKURAN, *CFG.UKURAN_CADANGAN):\n"
    "        for bs in CFG.BS_MIKRO:\n"
    "            h = coba(ukuran, bs)\n"
    "            if h is None:\n"
    "                print(f'{ukuran} px, batch mikro {bs}: OOM')\n"
    "                continue\n"
    "            detik, vram = h\n"
    "            per_citra = detik / bs\n"
    "            # 1,15: cadangan untuk DataLoader, sinkronisasi, dan update optimizer\n"
    "            jam_final = per_citra * len(L) * CFG.EPOCH * 1.15 / 3600\n"
    "            print(f'{ukuran} px, batch mikro {bs}: {detik:.2f} detik/langkah, VRAM puncak {vram:.1f} GiB, '\n"
    "                  f'perkiraan model final {jam_final:.2f} jam')\n"
    "            if vram > CFG.VRAM_MAKS * VRAM_GIB:\n"
    "                continue\n"
    "            if jam_final > CFG.BATAS_FINAL_JAM:\n"
    "                break\n"
    "            return dict(ukuran=ukuran, bs_mikro=bs, detik_per_citra=per_citra, vram_gib=round(vram, 1),\n"
    "                        jam_final=round(jam_final, 2))\n"
    "    raise RuntimeError('tidak ada konfigurasi yang muat di VRAM dan BATAS_FINAL_JAM; turunkan EPOCH atau naikkan batas')\n"
    "\n"
    "\n"
    "PILIHAN = pilih_konfigurasi()\n"
    "UKURAN, BS_MIKRO = PILIHAN['ukuran'], PILIHAN['bs_mikro']\n"
    "AKUM = CFG.BS // BS_MIKRO\n"
    "tf_train, tf_eval, tf_tampil = transform(UKURAN)\n"
    "print(f'dipakai: {UKURAN} px, batch mikro {BS_MIKRO} x akumulasi {AKUM} = batch efektif {BS_MIKRO * AKUM}')\n"
    "print(f'perkiraan: model final {PILIHAN[\"jam_final\"]:.2f} jam, satu fold CV {PILIHAN[\"jam_final\"] * 0.8 * 1.1:.2f} jam')"
))
cells.append(md(
    "#### Insights\n\n"
    "> Isi setelah run: resolusi dan batch mikro terpilih, VRAM puncak, dan perkiraan waktu. Bila resolusi turun dari "
    "384, catat alasannya (OOM atau waktu) di laporan, karena resolusi ikut menentukan detail tekstur yang terlihat "
    "model."
))

# ---------------------------------------------------------------------------
# 6. Training
# ---------------------------------------------------------------------------
cells.append(section("Training", "6"))
cells.append(md(
    "Training terdiri dari dua tahap. Tahap pertama mengekstrak fitur beku seluruh train sekali (LoRA masih nol, "
    "jadi identik dengan base model) dan memakainya untuk dua hal: linear probe sebagai pemeriksaan kewarasan sebelum "
    "training panjang, dan inisialisasi head. Tahap kedua melatih LoRA dan head bersama. Fold StratifiedGroupKFold "
    "ditetapkan di sini dan dipakai ulang di bagian 9."
))
cells.append(md(
    "## Fitur Beku dan Linear Probe\n\n"
    "Regresi logistik (C = 0,1) dilatih pada fitur beku yang dipusatkan dan diskalakan dengan satu konstanta, lalu "
    "koefisiennya dilipat menjadi bobot head pada fitur mentah. Skala per dimensi tidak dipakai karena dimensi yang "
    "hampir konstan akan menghasilkan bobot sangat besar. Linear probe dinilai dengan StratifiedGroupKFold 5 fold yang "
    "sama dengan cross-validation. Bila accuracy rata-ratanya di bawah 0,8, notebook berhenti karena kemungkinan besar "
    "ada masalah pemuatan bobot atau preprocessing. Angka ini bukan perkiraan skor test: aturan pembersihan notebook v2 "
    "memakai tetangga di ruang 7B, jadi label bersih memang selaras dengan ruang ini."
))
cells.append(code(
    "# Python 3.14 memakai forkserver secara default; worker forkserver tidak bisa memuat kelas yang didefinisikan di notebook\n"
    "MP_KONTEKS = 'fork' if CFG.NUM_WORKERS else None\n"
    "\n"
    "\n"
    "@torch.no_grad()\n"
    "def prediksi(X, idx=None, fitur_saja=False):\n"
    "    m.eval()\n"
    "    dl = DataLoader(Citra(X, tf_eval, idx), batch_size=2 * BS_MIKRO, num_workers=CFG.NUM_WORKERS,\n"
    "                    multiprocessing_context=MP_KONTEKS)\n"
    "    out = []\n"
    "    for x in dl:\n"
    "        with torch.autocast('cuda', dtype=torch.bfloat16):\n"
    "            f = fitur(m, x.to(DEV, non_blocking=True))\n"
    "            out.append((f if fitur_saja else torch.softmax(kepala(f).float(), 1)).float().cpu())\n"
    "    return torch.cat(out).numpy()\n"
    "\n"
    "\n"
    "def kepala_lp(F_, y):\n"
    "    mu, s = F_.mean(0), F_.std()\n"
    "    lr = LogisticRegression(C=CFG.LP_C, max_iter=3000).fit((F_ - mu) / s, y)\n"
    "    assert len(lr.classes_) == len(CFG.KELAS), 'ada kelas tanpa contoh di data latih'\n"
    "    W = lr.coef_ / s\n"
    "    return W.astype(np.float32), (lr.intercept_ - W @ mu).astype(np.float32)\n"
    "\n"
    "\n"
    "t0 = time.time()\n"
    "F_TRAIN = prediksi(X_train, fitur_saja=True)\n"
    "DETIK_FITUR = time.time() - t0\n"
    "assert np.isfinite(F_TRAIN).all(), 'fitur beku non-finite'\n"
    "\n"
    "sgkf = StratifiedGroupKFold(n_splits=CFG.N_FOLD, shuffle=True, random_state=CFG.SEED)\n"
    "FOLD = np.full(len(L), -1)\n"
    "for f, (_, b) in enumerate(sgkf.split(L, Y, GRUP[L])):\n"
    "    FOLD[b] = f\n"
    "assert (FOLD >= 0).all() and (pd.Series(FOLD).groupby(GRUP[L]).nunique() == 1).all(), 'grup terbelah antar fold'\n"
    "\n"
    "ACC_LP = []\n"
    "for f in range(CFG.N_FOLD):\n"
    "    a, b = FOLD != f, FOLD == f\n"
    "    W, c = kepala_lp(F_TRAIN[L[a]], Y[a])\n"
    "    ACC_LP.append(float(((F_TRAIN[L[b]] @ W.T + c).argmax(1) == Y[b]).mean()))\n"
    "print(f'fitur beku {F_TRAIN.shape} dalam {DETIK_FITUR:.0f} detik')\n"
    "print('ukuran fold:', np.bincount(FOLD).tolist(), '| accuracy linear probe per fold:', np.round(ACC_LP, 4).tolist())\n"
    "print(f'linear probe: {np.mean(ACC_LP):.4f} +- {np.std(ACC_LP, ddof=1):.4f}')\n"
    "assert np.mean(ACC_LP) >= CFG.LP_MIN, 'linear probe terlalu rendah: periksa bobot dan preprocessing sebelum training'"
))
cells.append(md(
    "#### Insights\n\n"
    "> Isi setelah run: accuracy linear probe per fold dan rata-ratanya. Angka ini menjadi pembanding bagi "
    "cross-validation LoRA di bagian 9. Bila LoRA tidak lebih baik dari linear probe, fine-tune tidak menambah "
    "informasi di atas fitur beku untuk label ini."
))

cells.append(md(
    "## Fungsi Training\n\n"
    "`latih` mengembalikan LoRA ke titik awal, menginisialisasi head dari linear probe pada data latihnya sendiri, "
    "lalu melatih LoRA (LR 1e-4) dan head (LR 1e-3) dengan AdamW (weight decay 0,05), warmup linear 10% langkah lalu "
    "cosine, cross-entropy dengan label smoothing 0,1 dan bobot kelas akar-invers, gradient clipping 1,0, serta "
    "autocast bf16. Bila diberi nama, checkpoint LoRA, head, dan optimizer disimpan tiap epoch dan training lanjut "
    "dari sana saat sel dijalankan ulang, asalkan data dan konfigurasinya sama. `latih_aman` mengulang dengan batch "
    "mikro setengahnya bila terjadi OOM, dengan batch efektif tetap 16."
))
cells.append(code(
    "def bobot_kelas(y):\n"
    "    w = 1 / np.sqrt(np.bincount(y, minlength=len(CFG.KELAS)).clip(1))\n"
    "    return torch.tensor(w / w.mean(), dtype=torch.float32, device=DEV)\n"
    "\n"
    "\n"
    "def jadwal(langkah, total):\n"
    "    warm = max(1, int(CFG.WARMUP * total))\n"
    "    if langkah < warm:\n"
    "        return (langkah + 1) / warm\n"
    "    return 0.5 * (1 + math.cos(math.pi * (langkah - warm) / max(1, total - warm)))\n"
    "\n"
    "\n"
    "def latih(idx, y, idx_va=None, y_va=None, nama=None):\n"
    "    seed_everything(CFG.SEED)\n"
    "    W, c = kepala_lp(F_TRAIN[idx], y)\n"
    "    with torch.no_grad():\n"
    "        for n, p in LORA.items():\n"
    "            p.copy_(LORA_AWAL[n])\n"
    "        kepala.weight.copy_(torch.from_numpy(W))\n"
    "        kepala.bias.copy_(torch.from_numpy(c))\n"
    "    dl = DataLoader(Citra(X_train, tf_train, idx, y), batch_size=BS_MIKRO, shuffle=True, drop_last=True,\n"
    "                    num_workers=CFG.NUM_WORKERS, worker_init_fn=seed_worker, pin_memory=True,\n"
    "                    multiprocessing_context=MP_KONTEKS, generator=torch.Generator().manual_seed(CFG.SEED),\n"
    "                    persistent_workers=CFG.NUM_WORKERS > 0)\n"
    "    per_epoch = len(dl) // AKUM\n"
    "    total = per_epoch * CFG.EPOCH\n"
    "    opt = torch.optim.AdamW([{'params': list(LORA.values()), 'lr': CFG.LR_LORA},\n"
    "                             {'params': list(kepala.parameters()), 'lr': CFG.LR_KEPALA}], weight_decay=CFG.WD)\n"
    "    lr_dasar = [g['lr'] for g in opt.param_groups]\n"
    "    kriteria = nn.CrossEntropyLoss(weight=bobot_kelas(y), label_smoothing=CFG.SMOOTH)\n"
    "    resep = (UKURAN, CFG.BS, CFG.EPOCH, CFG.LR_LORA, CFG.LR_KEPALA, CFG.WD, CFG.LORA_R, CFG.LORA_ALPHA, MUAT)\n"
    "    kunci = hashlib.sha256(np.asarray(idx).tobytes() + np.asarray(y).tobytes() + repr(resep).encode()).hexdigest()[:16]\n"
    "    jalur = CFG.OUTPUT_DIR / f'ckpt_{nama}.pt' if nama else None\n"
    "    mulai, langkah, riwayat, pv = 0, 0, [], None\n"
    "    if jalur and jalur.exists():\n"
    "        s = torch.load(jalur, map_location='cpu', weights_only=False)\n"
    "        if s['kunci'] == kunci:\n"
    "            with torch.no_grad():\n"
    "                for n, p in LORA.items():\n"
    "                    p.copy_(s['lora'][n])\n"
    "            kepala.load_state_dict(s['kepala'])\n"
    "            opt.load_state_dict(s['opt'])\n"
    "            mulai, langkah, riwayat = s['epoch'], s['langkah'], s['riwayat']\n"
    "            print(f'{jalur.name}: lanjut setelah epoch {mulai} dari {CFG.EPOCH}')\n"
    "    for ep in range(mulai, CFG.EPOCH):\n"
    "        m.train()\n"
    "        rugi, t0 = [], time.time()\n"
    "        opt.zero_grad(set_to_none=True)\n"
    "        for i, (x, yb) in enumerate(dl):\n"
    "            if i >= per_epoch * AKUM:\n"
    "                break\n"
    "            with torch.autocast('cuda', dtype=torch.bfloat16):\n"
    "                loss = kriteria(kepala(fitur(m, x.to(DEV, non_blocking=True))).float(), yb.to(DEV, non_blocking=True))\n"
    "            if not torch.isfinite(loss):\n"
    "                raise FloatingPointError(f'loss non-finite di epoch {ep + 1}')\n"
    "            (loss / AKUM).backward()\n"
    "            rugi.append(loss.item())\n"
    "            if (i + 1) % AKUM == 0:\n"
    "                skala = jadwal(langkah, total)\n"
    "                for g, lr0 in zip(opt.param_groups, lr_dasar):\n"
    "                    g['lr'] = lr0 * skala\n"
    "                nn.utils.clip_grad_norm_([*LORA.values(), *kepala.parameters()], 1.0)\n"
    "                opt.step()\n"
    "                opt.zero_grad(set_to_none=True)\n"
    "                langkah += 1\n"
    "        r = dict(epoch=ep + 1, loss=round(float(np.mean(rugi)), 4), detik=round(time.time() - t0))\n"
    "        if idx_va is not None:\n"
    "            pv = prediksi(X_train, idx_va)\n"
    "            r['val_acc'] = round(float((pv.argmax(1) == y_va).mean()), 4)\n"
    "        riwayat.append(r)\n"
    "        print(r, flush=True)\n"
    "        if jalur:\n"
    "            sementara = jalur.with_suffix('.tmp')\n"
    "            torch.save(dict(kunci=kunci, epoch=ep + 1, langkah=langkah, riwayat=riwayat, opt=opt.state_dict(),\n"
    "                            lora={n: p.detach().cpu() for n, p in LORA.items()}, kepala=kepala.state_dict()), sementara)\n"
    "            os.replace(sementara, jalur)\n"
    "    if idx_va is not None and pv is None:\n"
    "        pv = prediksi(X_train, idx_va)\n"
    "    return riwayat, pv\n"
    "\n"
    "\n"
    "def latih_aman(*args, **kw):\n"
    "    global BS_MIKRO, AKUM\n"
    "    while True:\n"
    "        try:\n"
    "            return latih(*args, **kw)\n"
    "        except torch.cuda.OutOfMemoryError:\n"
    "            if BS_MIKRO <= 1:\n"
    "                raise\n"
    "        BS_MIKRO //= 2\n"
    "        AKUM = CFG.BS // BS_MIKRO\n"
    "        print(f'OOM saat training, batch mikro diturunkan ke {BS_MIKRO} x akumulasi {AKUM}')\n"
    "        for p in [*LORA.values(), *kepala.parameters()]:\n"
    "            p.grad = None\n"
    "        gc.collect()\n"
    "        torch.cuda.empty_cache()\n"
    "\n"
    "\n"
    "def simpan_adapter(berkas, info):\n"
    "    t = {f'lora.{n}': p.detach().cpu().contiguous() for n, p in LORA.items()}\n"
    "    t.update({f'kepala.{k}': v.detach().cpu().contiguous() for k, v in kepala.state_dict().items()})\n"
    "    save_file(t, str(berkas), metadata={k: str(v) for k, v in info.items()})\n"
    "    print(f'adapter disimpan: {berkas.name} ({berkas.stat().st_size / 2**20:.0f} MiB)')"
))

# ---------------------------------------------------------------------------
# 7. Model final dan inference
# ---------------------------------------------------------------------------
cells.append(section("Model Final dan Inference Test", "7"))
cells.append(md(
    "Model final dilatih pada seluruh 1.750 citra berlabel dengan resep yang sama dengan cross-validation. Urutannya "
    "sengaja dibuat model final dulu, lalu CV: semua hyperparameter sudah ditetapkan dan tidak bergantung pada hasil "
    "CV, sehingga submission aman meskipun CV terpotong waktu (pelajaran run versi 4 notebook v2 yang mati di tengah "
    "CV). Yang disimpan hanya adapter LoRA dan head (sekitar 150 MB) serta checkpoint resume berisi state optimizer "
    "(sekitar 450 MB), bukan model 7B utuh. Test baru dimuat di sini "
    "dengan preprocessing yang identik dengan evaluasi train (trim pita hitam, stretch, resize langsung), lalu "
    "diprediksi satu kali tanpa TTA. Probabilitas 170 citra train tak berlabel juga disimpan untuk pseudo-label "
    "opsional di bagian 10."
))
cells.append(code(
    "t0 = time.time()\n"
    "RIW_FINAL, _ = latih_aman(L, Y, nama='final')\n"
    "JAM_FINAL = (time.time() - t0) / 3600\n"
    "INFO_ADAPTER = dict(repo=CFG.M7B[0], revisi=CFG.M7B[1], mode=MUAT, ukuran=UKURAN, lora_r=CFG.LORA_R,\n"
    "                    lora_alpha=CFG.LORA_ALPHA, target=CFG.LORA_TARGET, epoch=CFG.EPOCH, kelas=','.join(CFG.KELAS))\n"
    "simpan_adapter(CFG.OUTPUT_DIR / 'adapter_lora_final.safetensors', INFO_ADAPTER)\n"
    "\n"
    "with ThreadPoolExecutor(8) as ex:\n"
    "    X_test = np.stack(list(ex.map(muat_citra, f_test)))\n"
    "P_TEST = prediksi(X_test)\n"
    "P_U = prediksi(X_train, U)\n"
    "np.save(CFG.OUTPUT_DIR / 'prob_test.npy', P_TEST)\n"
    "np.save(CFG.OUTPUT_DIR / 'prob_train_tak_berlabel.npy', P_U)\n"
    "assert np.isfinite(P_TEST).all() and P_TEST.shape == (480, len(CFG.KELAS))\n"
    "print(f'model final {JAM_FINAL:.2f} jam | riwayat: {RIW_FINAL}')\n"
    "print('sebaran prediksi test:', pd.Series(np.array(CFG.KELAS)[P_TEST.argmax(1)]).value_counts().to_dict())\n"
    "print('sebaran label train:', pd.Series(np.array(CFG.KELAS)[Y]).value_counts(normalize=True).round(3).to_dict())"
))

# ---------------------------------------------------------------------------
# 8. Submission
# ---------------------------------------------------------------------------
cells.append(section("Submission", "8"))
cells.append(md(
    "Submission ditulis dengan format `image_id,waste_category`. `image_id` diambil apa adanya dari sample "
    "submission (string tiga digit, bukan dari nama berkas) dan urutan barisnya disamakan. Berkas dibaca ulang dan "
    "dicocokkan dengan sample sebelum dinyatakan siap: header, jumlah baris, urutan ID, dan nama kelas. Yang "
    "mengunggah submission adalah tim."
))
cells.append(code(
    "def tulis_submission(P, berkas):\n"
    "    contoh = pd.read_csv(CFG.SUB_PATH, dtype=str, encoding='utf-8-sig')\n"
    "    pred = dict(zip(id_test.tolist(), np.array(CFG.KELAS)[P.argmax(1)]))\n"
    "    assert set(contoh.image_id.astype(int)) == set(pred), 'image_id test tidak cocok dengan sample'\n"
    "    sub = pd.DataFrame({'image_id': contoh.image_id, 'waste_category': [pred[int(i)] for i in contoh.image_id]})\n"
    "    sub.to_csv(berkas, index=False)\n"
    "    cek = pd.read_csv(berkas, dtype=str)\n"
    "    assert list(cek.columns) == ['image_id', 'waste_category'] and len(cek) == 480\n"
    "    assert cek.image_id.tolist() == contoh.image_id.tolist() and cek.image_id.str.fullmatch(r'\\d{3}').all()\n"
    "    assert cek.waste_category.isin(CFG.KELAS).all()\n"
    "    return cek\n"
    "\n"
    "\n"
    "SUB = tulis_submission(P_TEST, CFG.OUTPUT_DIR / 'submission.csv')\n"
    "print('submission.csv siap:', CFG.OUTPUT_DIR / 'submission.csv')\n"
    "print(SUB.waste_category.value_counts().to_dict())\n"
    "display(SUB.head())"
))

# ---------------------------------------------------------------------------
# 9. Validasi lokal
# ---------------------------------------------------------------------------
cells.append(section("Validasi Lokal", "9"))
cells.append(md(
    "Accuracy di sini adalah kesepakatan model dengan label cluster yang dinamai tim dan sudah dibersihkan, bukan "
    "accuracy terhadap ground truth. Citra ragu sudah dibuang dari label, jadi angka ini cenderung lebih tinggi "
    "daripada accuracy test. Simpangan baku antar fold menunjukkan seberapa jauh skor bisa bergeser hanya karena "
    "sampel yang berbeda."
))
cells.append(md(
    "## Cross-validation\n\n"
    "StratifiedGroupKFold 5 fold memakai grup near-duplicate, sehingga salinan sebuah citra tidak muncul di train dan "
    "validasi sekaligus. EDA lokal 211 mengukur alasannya: dengan split acak, 33% citra validasi punya tetangga "
    "terdekat dari grup duplikatnya sendiri, dan accuracy kNN naik 3,1 sampai 7,0 poin hanya karena salinan. Tiap fold "
    "memakai resep yang persis sama dengan model final, termasuk inisialisasi head dari linear probe pada data latih "
    "fold itu saja. Accuracy fold diambil dari epoch terakhir, tanpa memilih epoch terbaik. Minimal 2 fold selalu "
    "dijalankan (`MIN_FOLD`) supaya simpangan baku bisa dihitung. Setelah itu fold berikutnya hanya dimulai bila "
    "perkiraan waktunya masih di dalam `BATAS_CV_JAM`. Submission sudah ditulis sebelum bagian ini, jadi CV boleh "
    "dihentikan kapan saja. Hasil tiap fold disimpan ke disk dan dimuat ulang saat sel dijalankan lagi."
))
cells.append(code(
    "TAG = f'{MUAT}_{UKURAN}_e{CFG.EPOCH}_r{CFG.LORA_R}'\n"
    "OOF = np.zeros((len(L), len(CFG.KELAS)))\n"
    "RIW_CV, SELESAI = [], []\n"
    "if CFG.JALANKAN_CV:\n"
    "    t_mulai = time.time()\n"
    "    t_fold = PILIHAN['jam_final'] * 0.8 * 1.1 * 3600\n"
    "    for f in range(CFG.N_FOLD):\n"
    "        a, b = np.flatnonzero(FOLD != f), np.flatnonzero(FOLD == f)\n"
    "        berkas = CFG.OUTPUT_DIR / f'cv_fold{f}_{TAG}.npz'\n"
    "        if berkas.exists():\n"
    "            with np.load(berkas) as s:\n"
    "                OOF[b] = s['pv']\n"
    "                RIW_CV += json.loads(str(s['riwayat']))\n"
    "            SELESAI.append(f)\n"
    "            print(f'fold {f}: dimuat dari {berkas.name}')\n"
    "            continue\n"
    "        if len(SELESAI) >= CFG.MIN_FOLD and (time.time() - t_mulai + t_fold) / 3600 > CFG.BATAS_CV_JAM:\n"
    "            print(f'fold {f} dan seterusnya dilewati: perkiraan melewati BATAS_CV_JAM = {CFG.BATAS_CV_JAM} jam')\n"
    "            break\n"
    "        t0 = time.time()\n"
    "        print(f'fold {f}: latih {len(a)}, validasi {len(b)}')\n"
    "        riw, pv = latih_aman(L[a], Y[a], L[b], Y[b])\n"
    "        OOF[b] = pv\n"
    "        riw = [dict(fold=f, **r) for r in riw]\n"
    "        RIW_CV += riw\n"
    "        SELESAI.append(f)\n"
    "        sementara = berkas.with_suffix('.tmp')\n"
    "        with open(sementara, 'wb') as fh:\n"
    "            np.savez(fh, pv=pv, riwayat=json.dumps(riw))\n"
    "        os.replace(sementara, berkas)\n"
    "        t_fold = time.time() - t0\n"
    "else:\n"
    "    print('cross-validation dilewati (JALANKAN_CV = False)')\n"
    "print('fold selesai:', SELESAI)"
))
cells.append(md(
    "## Metrik\n\n"
    "Metrik kompetisi adalah accuracy, jadi accuracy per fold beserta mean dan simpangan bakunya menjadi angka utama. "
    "Karena masalahnya multikelas, precision, recall, dan F1 per kelas serta rata-rata macro dan weighted ikut "
    "dilaporkan, bersama confusion matrix dan kurva accuracy validasi per epoch. Linear probe pada fold yang sama "
    "menjadi pembanding. Interval binomial 95% untuk 145 citra (public) dan 335 citra (private) dicetak sebagai patokan "
    "selisih public dan private yang wajar."
))
cells.append(code(
    "ACC_CV = np.array([(OOF[FOLD == f].argmax(1) == Y[FOLD == f]).mean() for f in SELESAI])\n"
    "if len(ACC_CV):\n"
    "    sd = ACC_CV.std(ddof=1) if len(ACC_CV) > 1 else float('nan')\n"
    "    lp = np.array(ACC_LP)[SELESAI]\n"
    "    print('accuracy per fold:', ACC_CV.round(4).tolist())\n"
    "    print(f'accuracy CV ({len(ACC_CV)} dari {CFG.N_FOLD} fold): {ACC_CV.mean():.4f} +- {sd:.4f}')\n"
    "    print(f'linear probe pada fold yang sama: {lp.mean():.4f} | selisih LoRA - linear probe per fold: {(ACC_CV - lp).round(4).tolist()}')\n"
    "    for n in (145, 335):\n"
    "        se = math.sqrt(ACC_CV.mean() * (1 - ACC_CV.mean()) / n)\n"
    "        print(f'n = {n}: galat baku {se:.4f}, interval 95% kira-kira {ACC_CV.mean() - 1.96 * se:.3f} sampai {min(1, ACC_CV.mean() + 1.96 * se):.3f}')\n"
    "\n"
    "    ada = np.isin(FOLD, SELESAI)\n"
    "    yt, yp = Y[ada], OOF[ada].argmax(1)\n"
    "    p, r, f1, n = precision_recall_fscore_support(yt, yp, labels=range(len(CFG.KELAS)), zero_division=0)\n"
    "    tabel = pd.DataFrame({'precision': p, 'recall': r, 'f1': f1, 'n': n}, index=CFG.KELAS)\n"
    "    for rata in ('macro', 'weighted'):\n"
    "        tabel.loc[rata] = [*precision_recall_fscore_support(yt, yp, average=rata, zero_division=0)[:3], len(yt)]\n"
    "    display(tabel.round(4))\n"
    "\n"
    "    cm = confusion_matrix(yt, yp, labels=range(len(CFG.KELAS)))\n"
    "    fig, ax = plt.subplots(1, 3, figsize=(19, 5))\n"
    "    ax[0].imshow(cm, cmap='Blues')\n"
    "    ax[0].set_xticks(range(len(CFG.KELAS)), CFG.KELAS, rotation=45, fontsize=8)\n"
    "    ax[0].set_yticks(range(len(CFG.KELAS)), CFG.KELAS, fontsize=8)\n"
    "    for i in range(len(CFG.KELAS)):\n"
    "        for j in range(len(CFG.KELAS)):\n"
    "            ax[0].text(j, i, cm[i, j], ha='center', va='center', fontsize=8, color='w' if cm[i, j] > cm.max() / 2 else 'k')\n"
    "    ax[0].set(xlabel='prediksi OOF', ylabel='label cluster bersih', title='confusion matrix OOF')\n"
    "    x = np.arange(len(CFG.KELAS))\n"
    "    ax[1].bar(x - 0.2, r, 0.4, label='recall')\n"
    "    ax[1].bar(x + 0.2, p, 0.4, label='precision')\n"
    "    ax[1].set_xticks(x, CFG.KELAS, rotation=45, fontsize=8)\n"
    "    ax[1].set(ylim=(max(0, min(r.min(), p.min()) - 0.05), 1.0), title='precision dan recall per kelas (OOF)')\n"
    "    ax[1].legend()\n"
    "    for f, d in pd.DataFrame(RIW_CV).groupby('fold'):\n"
    "        ax[2].plot(d.epoch, d.val_acc, marker='o', label=f'fold {f}')\n"
    "    ax[2].set(xlabel='epoch', ylabel='accuracy validasi', title='accuracy validasi per epoch')\n"
    "    ax[2].legend()\n"
    "    plt.tight_layout()\n"
    "    fig.savefig(CFG.FIG_DIR / 'validasi.png', dpi=100)\n"
    "    plt.show()\n"
    "else:\n"
    "    print('belum ada fold selesai, metrik CV tidak tersedia')"
))
cells.append(md(
    "#### Insights\n\n"
    "> Isi setelah run: mean dan simpangan baku accuracy per fold, selisihnya terhadap linear probe, kelas dengan "
    "recall terendah, dan pasangan kelas yang paling sering tertukar di confusion matrix. Bila simpangan baku antar "
    "fold lebih dari sekitar 1,5 poin persen, selisih skor public dan private yang besar wajar terjadi. Bila kurva "
    "validasi masih naik di epoch terakhir, jumlah epoch mungkin terlalu sedikit, tetapi resep tidak diubah setelah "
    "melihat angka ini."
))

# ---------------------------------------------------------------------------
# 10. Pseudo-label opsional
# ---------------------------------------------------------------------------
cells.append(section("Pseudo-label Train (Opsional)", "10"))
cells.append(md(
    "Bagian ini hanya berjalan bila `JALANKAN_PSEUDO = True`, setelah `submission.csv` aman. Model final memberi "
    "label pada 170 citra train tak berlabel (51 `campur` dan 119 yang dibuang aturan), dan hanya prediksi dengan "
    "probabilitas maksimum minimal `AMBANG_PSEUDO = 0,85` yang dipakai. Ambang ditetapkan sebelum run. Dengan label "
    "smoothing 0,1, probabilitas prediksi yang yakin berada di sekitar 0,91, jadi 0,85 hanya menerima prediksi yang "
    "jelas dominan. Pseudo-labeling hanya memakai train, sesuai QnA #22.3, dan label yang sudah ada tidak diganti. "
    "Satu model baru dilatih dengan resep yang sama, lalu prediksinya ditulis ke `submission_pseudo.csv` tanpa "
    "menimpa `submission.csv`.\n\n"
    "Efek pseudo-label tidak divalidasi dengan CV karena butuh training tambahan untuk setiap fold. Citra tak berlabel "
    "adalah citra yang paling ragu, sehingga label semunya bisa menambah noise. Karena itu `submission.csv` tetap "
    "submission utama, dan keputusan memakai `submission_pseudo.csv` ada pada tim."
))
cells.append(code(
    "INFO_PSEUDO = None\n"
    "if CFG.JALANKAN_PSEUDO:\n"
    "    yakin = P_U.max(1) >= CFG.AMBANG_PSEUDO\n"
    "    U_PL, Y_PL = U[yakin], P_U[yakin].argmax(1)\n"
    "    print(f'pseudo-label: {int(yakin.sum())} dari {len(U)} citra train tak berlabel lolos ambang {CFG.AMBANG_PSEUDO}')\n"
    "    print(pd.Series(np.array(CFG.KELAS)[Y_PL], dtype=object).value_counts().to_dict())\n"
    "    t0 = time.time()\n"
    "    RIW_PSEUDO, _ = latih_aman(np.r_[L, U_PL], np.r_[Y, Y_PL], nama='pseudo')\n"
    "    simpan_adapter(CFG.OUTPUT_DIR / 'adapter_lora_pseudo.safetensors', {**INFO_ADAPTER, 'pseudo_label': int(yakin.sum())})\n"
    "    P_TEST_PSEUDO = prediksi(X_test)\n"
    "    np.save(CFG.OUTPUT_DIR / 'prob_test_pseudo.npy', P_TEST_PSEUDO)\n"
    "    tulis_submission(P_TEST_PSEUDO, CFG.OUTPUT_DIR / 'submission_pseudo.csv')\n"
    "    sama = float((P_TEST_PSEUDO.argmax(1) == P_TEST.argmax(1)).mean())\n"
    "    INFO_PSEUDO = dict(n=int(yakin.sum()), jam=round((time.time() - t0) / 3600, 2), riwayat=RIW_PSEUDO, sama_dengan_utama=sama)\n"
    "    print(f'submission_pseudo.csv siap | prediksi test sama dengan submission utama: {sama:.4f}')\n"
    "else:\n"
    "    print('pseudo-label dilewati (JALANKAN_PSEUDO = False)')"
))

# ---------------------------------------------------------------------------
# 11. Ringkasan
# ---------------------------------------------------------------------------
cells.append(section("Ringkasan Run", "11"))
cells.append(md(
    "Ringkasan run disimpan ke `ringkasan_run.json` untuk laporan: versi pustaka dan GPU, mode bobot, resolusi dan "
    "batch terpilih, waktu tiap tahap, riwayat training, linear probe, cross-validation, dan sebaran prediksi test. "
    "Berkas yang diunggah tim adalah `keluaran_7b/submission.csv`."
))
cells.append(code(
    "RINGKASAN = dict(\n"
    "    gpu=INFO_GPU, mode=MUAT, konfigurasi=PILIHAN, ukuran=UKURAN, bs_mikro=BS_MIKRO, akumulasi=AKUM,\n"
    "    lora=dict(r=CFG.LORA_R, alpha=CFG.LORA_ALPHA, dropout=CFG.LORA_DROPOUT, lapis=n_modul, parameter_latih=n_latih),\n"
    "    detik_muat=round(DETIK_MUAT), detik_fitur=round(DETIK_FITUR), jam_final=round(JAM_FINAL, 2), riwayat_final=RIW_FINAL,\n"
    "    linear_probe=dict(per_fold=ACC_LP, mean=float(np.mean(ACC_LP)), std=float(np.std(ACC_LP, ddof=1))),\n"
    "    cv=dict(fold=SELESAI, per_fold=ACC_CV.tolist(), mean=float(ACC_CV.mean()) if len(ACC_CV) else None,\n"
    "            std=float(ACC_CV.std(ddof=1)) if len(ACC_CV) > 1 else None),\n"
    "    submission=SUB.waste_category.value_counts().to_dict(), pseudo=INFO_PSEUDO)\n"
    "with open(CFG.OUTPUT_DIR / 'ringkasan_run.json', 'w', encoding='utf-8') as fh:\n"
    "    json.dump(RINGKASAN, fh, indent=1, default=str)\n"
    "print(json.dumps(RINGKASAN, indent=1, default=str))\n"
    "print('berkas keluaran:', sorted(p.name for p in CFG.OUTPUT_DIR.iterdir()))"
))
cells.append(md(
    "#### Insights\n\n"
    "> Isi setelah run: resolusi dan mode bobot yang benar-benar dipakai, waktu model final, accuracy CV dengan "
    "simpangan bakunya, dan perbandingannya dengan linear probe. Semua angka di laporan diambil dari "
    "`ringkasan_run.json`, bukan dari perkiraan di sel markdown.\n\n"
    "> Submission berasal dari satu model DINOv3 ViT-7B dengan adapter LoRA yang dilatih pada seluruh label bersih. "
    "Tidak ada ensemble, rata-rata fold, TTA, atau pseudo-label dari test."
))

# ---------------------------------------------------------------------------
# Notebook writer
# ---------------------------------------------------------------------------
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "cells": cells,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(cells)} cells -> {OUT}")
