"""Builder notebook end-to-end DSC Gelar Rasa 2026. Jangan mengedit .ipynb langsung.

    python temp/nb_final/build_final.py
"""
import hashlib
import json
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "dsc_gelar_rasa_2026_dinov3_convnext_normal.ipynb"
HR = "---"
SUMBER = ["temp/exp_005_imagenet_map/imagenet_map.py", "temp/exp_006_cluster_dinov3/ekstrak_embedding.py", "temp/exp_008_in22k_map/infer22k.py",
          "temp/exp_008_in22k_map/infer_dms.py", "temp/eda_104_duplikat/hash_dup.py", "temp/eda_110_fusi/label_final.py",
          "temp/eda_112_label_v2/peta.py", "temp/eda_112_label_v2/sumber_v2.py", "temp/eda_112_label_v2/label_v2.py",
          "temp/research_uji_crossfit/deteksi.py", "temp/research_uji_crossfit/aturan_label.py", "temp/research_uji_crossfit/uji8_isi_coteach.py",
          "temp/research_uji_crossfit/uji10_label_v3.py", "temp/exp_011_label_v4/gabung_label.py", "temp/eda_113_elektronik/label_v5.py",
          "temp/eda_114_paper/label_v6.py", "temp/exp_010_dua_backbone/train_dua.py"]


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": uuid4().hex[:8], "metadata": {}, "source": source}


def code(source: str, tags=None) -> dict:
    meta = {"tags": tags} if tags else {}
    return {"cell_type": "code", "execution_count": None, "id": uuid4().hex[:8], "metadata": meta, "outputs": [], "source": source.strip("\n")}


def section(title: str, anchor: str) -> dict:
    return md(f"{HR}\n\n# {title} <a name=\"{anchor}\"></a>\n\n{HR}")


cells = []

# ---------------------------------------------------------------------------
# Banner (3 cells) + TOC
# ---------------------------------------------------------------------------
cells.append(md(
    f"{HR}\n\n"
    "# DSC Gelar Rasa 2026: Klasifikasi Sampah Tanpa Label\n\n"
    "*Pipeline end-to-end yang membangun label train secara otomatis dari model citra, lalu melatih DINOv3 ViT-L dan ConvNeXt-L.*"
))
cells.append(md(
    f"{HR}\n\n"
    "## Nama Tim\n\n"
    "- Isi nama anggota 1\n"
    "- Isi nama anggota 2\n"
    "- Isi nama anggota 3"
))
cells.append(md(
    f"{HR}\n\n"
    "## Table of Contents\n\n"
    "1. [**Introduction**](#1)\n"
    "2. [**Initialization**](#2)\n"
    "3. [**Sumber Sinyal**](#3)\n"
    "4. [**Label Dasar**](#4)\n"
    "5. [**Pemeriksaan Independen**](#5)\n"
    "6. [**Deteksi Label Salah**](#6)\n"
    "7. [**Aturan Baterai**](#7)\n"
    "8. [**Training Tahap Pertama**](#8)\n"
    "9. [**Aturan Foil dan Pengisian**](#9)\n"
    "10. [**Training Final**](#10)\n"
    "11. [**Submission**](#11)\n"
))

# ---------------------------------------------------------------------------
# Section 1: Introduction
# ---------------------------------------------------------------------------
cells.append(section("Introduction", "1"))
cells.append(md(
    "## Overview\n\n"
    "*Pemilahan sampah manual lambat dan rawan keliru, sehingga Computer Vision dipakai untuk mengenali kategori material dari citra. "
    "Kompetisi ini memberi 1.920 citra train dan 480 citra test untuk delapan kategori, tetapi berkas label train tidak tersedia.*"
))
cells.append(md(
    "## Aim\n\n"
    "*Notebook ini membangun label train secara otomatis tanpa pelabelan manual, lalu melatih dua backbone berbeda arsitektur "
    "untuk memprediksi `waste_category` pada 480 citra test.*"
))
cells.append(md(
    "## Metric\n\n"
    "***Accuracy** mengukur proporsi prediksi benar terhadap seluruh data yang dievaluasi. "
    "Metric ini dipakai panitia pada leaderboard.*\n\n"
    "$$\\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}$$"
))
cells.append(md(
    "## Dataset\n\n"
    "*Data berasal dari Kaggle, kompetisi DSC Gelar Rasa 2026 di: https://www.kaggle.com/competitions/dsc-gelar-rasa-2026*\n\n"
    "*Dataset berisi 1.920 citra train tanpa label dan 480 citra test, dengan ekstensi campur dan ukuran beragam. "
    "Delapan kelasnya adalah glass, paper, electronic, metal, plastic, cardboard, textile, dan foodwaste.*\n\n"
    "*Dataset dipakai untuk membentuk label train otomatis dan melatih model klasifikasi.*\n\n"
    "```\n"
    "data/\n"
    "├── Dataset DSC GelarRasa 2026/\n"
    "|    ├── train/\n"
    "|    └── test/\n"
    "└── SampleSubmission_DSC.csv\n"
    "```"
))
cells.append(md(
    "## Approach: Label Otomatis dan Dua Backbone\n\n"
    "*Tidak ada model bahasa-visi di notebook ini. Semua sinyal berasal dari model yang hanya menerima citra, sesuai rule 8 dan rule 9.*\n\n"
    "*Label dasar dibentuk dari 36 run KMeans pada embedding DINOv3. Nama cluster keluar dari rumus, yaitu massa kategori ImageNet-1k setelah "
    "normalisasi prior. Label lalu diperiksa oleh sumber independen, co-teaching dua head, dan aturan kosakata untuk baterai serta foil.*\n\n"
    "```\n"
    "Citra train\n"
    "    |\n"
    "    +-- Embedding DINOv3 ViT-L dan ConvNeXt-L\n"
    "    +-- Softmax ImageNet-1k, ImageNet-22k, material DMS46\n"
    "         |\n"
    "    Bagging KMeans -> nama otomatis -> label dasar\n"
    "         |\n"
    "    Dawid-Skene -> co-teaching -> aturan baterai -> aturan foil\n"
    "         |\n"
    "    Fine-tune DINOv3 ViT-L + ConvNeXt-L -> rata-rata probabilitas\n"
    "```\n\n"
    "*Model prediksi akhir hanya dua backbone. Classifier ImageNet dan DMS46 hanya dipakai sebagai sumber label di dalam train.*"
))

# ---------------------------------------------------------------------------
# Section 2: Initialization
# ---------------------------------------------------------------------------
cells.append(section("Initialization", "2"))
cells.append(md(
    "## Environment Setup\n\n"
    "Notebook berjalan di satu GPU RTX 5060 Ti 16 GB. Perkiraan waktu penuh sekitar empat jam, karena ada dua putaran training dua backbone. "
    "Keluaran antara disimpan di `OUTPUT_DIR/cache`, sehingga run bisa dilanjutkan setelah terhenti."
))
cells.append(code("!nvidia-smi"))
cells.append(md("The following cell installs all libraries used in this notebook."))
cells.append(code("%pip install -q numpy==2.4.6 pandas==2.2.3 scikit-learn==1.6.1 scipy timm==1.0.27 transformers==5.15.1 pillow"))
cells.append(md("## Import Libraries"))
cells.append(code(r"""
import json
import math
import os
import random
import re
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for nama_env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[nama_env] = '4'

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from scipy.fft import dctn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
from timm.data import ImageNetInfo
from timm.optim import create_optimizer_v2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from transformers import AutoImageProcessor, AutoModelForImageClassification
"""))
cells.append(md("## Seed Everything"))
cells.append(code(r"""
def seed_everything(seed: int = 42, warn_only: bool = True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    return seed


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


SEED = seed_everything(42)
torch.set_num_threads(4)
""", tags=["definitions"]))
cells.append(md(
    "## Settings\n\n"
    "Semua path, ambang, dan hyperparameter dipusatkan di sini. Saat pindah ke Colab atau Kaggle, cukup ubah path di kelas ini. "
    "Konstanta aturan label dibekukan sebelum aturannya diterapkan, jadi nilainya tidak boleh disetel ulang terhadap skor."
))
cells.append(code(r"""
class Settings:
    SEED = 42
    SMOKE_TEST = False
    SMOKE_TRAIN = 640
    SMOKE_TEST_N = 32

    PROJECT_ROOT = Path.cwd()
    while not (PROJECT_ROOT / 'data' / 'train').is_dir() and PROJECT_ROOT.parent != PROJECT_ROOT:
        PROJECT_ROOT = PROJECT_ROOT.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    SAMPLE_SUB = DATA_DIR / 'SampleSubmission_DSC.csv'
    CACHE_DIR = Path(os.environ.get('NGEDATA_CACHE', str(Path(r'D:\cache_moved') if Path(r'D:\cache_moved').exists() else Path.home() / '.cache' / 'dsc-gelar-rasa')))
    HUB_DIR = CACHE_DIR / 'huggingface' / 'hub'
    DMS_DIR = CACHE_DIR / 'dms46'
    OUTPUT_DIR = PROJECT_ROOT / 'temp' / 'nb_final' / 'output'

    NAMES = ['glass', 'paper', 'electronic', 'metal', 'plastic', 'cardboard', 'textile', 'foodwaste']
    N_KELAS = 8
    VRAM_MAKS_GIB = 13.5
    NUM_WORKERS = 0

    UKURAN = 384
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    BS_EMBED = 16
    BS_IN1K = 8
    BS_IN22K = 12

    KS = (48, 64, 96)
    KMEANS_SEEDS = 12
    PCA_DIM = 128
    SUARA_MIN = 0.75
    SIM_GROUP = 0.9
    HAMMING_GROUP = 4

    Q_MIN = 0.90
    DUKUNG_MIN = 2
    ITER_DS = 5
    BOBOT_DS = {'c_in1k': 0.5, 'c_in22k': 0.5, 'c_dms': 0.5, 'b_in22k': 0.5, 'b_dms': 0.5, 'e_lr': 1.0}

    LUPA = 0.2
    R_COTEACH = 5
    S_COTEACH = 3
    N_FOLD = 5
    TOLAK = 0.9
    YAKIN = 0.9
    SEED_DASAR = 2026
    SEED_RONDE2 = 3030
    SEED_ISI = (11, 22, 33, 44, 55)
    AMBANG_ISI = 0.7
    EPOCH_COTEACH = 60

    AMBANG_LISTRIK = 0.20
    AMBANG_FOIL = 0.20
    P_MIN = 0.80
    AMBANG_VOC = 0.20
    AMBANG_DMS = 0.30

    EPOCHS = 8
    LR = 1e-4
    HEAD_MULT = 10.0
    LAYER_DECAY = 0.75
    WD = 0.05
    WARMUP_EPOCH = 1
    LABEL_SMOOTHING = 0.1
    DROP_PATH = 0.1
    BS = 8
    ACCUM = 2
    BS_EVAL = 16
    K_ENSEMBLE = 3
    P_OOF = 0.80


CFG = Settings()
""", tags=["settings", "definitions"]))
cells.append(md(
    "Batas VRAM dipasang di tingkat proses supaya pelanggaran menjadi OOM, bukan bocor ke RAM. "
    "Mode smoke test memakai subset dan run KMeans yang lebih sedikit."
))
cells.append(code(r"""
if CFG.SMOKE_TEST:
    CFG.KS = (6, 8)
    CFG.KMEANS_SEEDS = 2
    CFG.R_COTEACH = 1
    CFG.S_COTEACH = 1
    CFG.N_FOLD = 2
    CFG.SEED_ISI = (11,)
    CFG.EPOCH_COTEACH = 2
    CFG.K_ENSEMBLE = 2

CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE = CFG.OUTPUT_DIR / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)
NAMES = CFG.NAMES
ID = {c: i for i, c in enumerate(NAMES)}
EL = ID['electronic']

if torch.cuda.is_available():
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(CFG.VRAM_MAKS_GIB / total_gib, 0)


def cache_npy(nama, fn):
    p = CACHE / f'{nama}.npy'
    if p.exists():
        return np.load(p)
    x = fn()
    np.save(p, x)
    return x
"""))
cells.append(md("## Load Dataset\n\nPath citra dikumpulkan dan diurutkan. ID diambil dari nama berkas karena ekstensinya campur."))
cells.append(code(r"""
train_files = sorted((CFG.DATA_DIR / 'train').iterdir())
test_files = sorted((CFG.DATA_DIR / 'test').iterdir())
if CFG.SMOKE_TEST:
    train_files, test_files = train_files[:CFG.SMOKE_TRAIN], test_files[:CFG.SMOKE_TEST_N]

ids = np.array([p.stem for p in train_files])
ids_te = np.array([p.stem for p in test_files])
nt, N = len(ids), len(ids) + len(ids_te)
peta_file = dict(zip(ids, train_files))
print(f'Train: {nt}  Test: {len(ids_te)}')
"""))

# ---------------------------------------------------------------------------
# Section 3: Sumber Sinyal
# ---------------------------------------------------------------------------
cells.append(section("Sumber Sinyal", "3"))
cells.append(md(
    "Semua sinyal dihitung dari citra saja. Embedding DINOv3 ViT-L dan ConvNeXt-L dipakai untuk clustering dan deteksi label salah. "
    "Softmax ImageNet dan material DMS46 dipakai untuk menamai cluster dan memeriksa label. Bobot dimuat dari cache lokal, dengan fallback ke Hugging Face Hub."
))
cells.append(code(r"""
ARCH = {'vit': ('timm/vit_large_patch16_dinov3.lvd1689m', 'vit_large_patch16_dinov3.lvd1689m', {'img_size': CFG.UKURAN}),
        'cnn': ('timm/convnext_large.dinov3_lvd1689m', 'convnext_large.dinov3_lvd1689m', {})}
MODEL_22K = {'convnext_l': ('timm/convnext_large.fb_in22k', 'convnext_large.fb_in22k', 288),
             'swinv2_l': ('timm/swinv2_large_window12_192.ms_in22k', 'swinv2_large_window12_192.ms_in22k', 192),
             'beitv2_l': ('timm/beitv2_large_patch16_224.in1k_ft_in22k', 'beitv2_large_patch16_224.in1k_ft_in22k', 224)}


def resolve_bobot(repo):
    lokal = CFG.CACHE_DIR / repo.replace('/', '--') / 'model.safetensors'
    if lokal.exists():
        return str(lokal)
    for s in sorted((CFG.HUB_DIR / ('models--' + repo.replace('/', '--')) / 'snapshots').glob('*')):
        if (s / 'model.safetensors').exists():
            return str(s / 'model.safetensors')
    return None


def buat_timm(repo, arch, **kw):
    berkas = resolve_bobot(repo)
    overlay = {'file': berkas} if berkas else None
    return timm.create_model(arch, pretrained=True, pretrained_cfg_overlay=overlay, **kw)


class Citra(Dataset):
    def __init__(self, files, labels=None, tf=None):
        self.files, self.labels, self.tf = files, labels, tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as im:
            x = self.tf(im.convert('RGB'))
        return (x, -1) if self.labels is None else (x, int(self.labels[i]))


def tf_eval():
    return T.Compose([T.Resize((CFG.UKURAN, CFG.UKURAN), interpolation=T.InterpolationMode.BICUBIC),
                      T.ToTensor(), T.Normalize(CFG.MEAN, CFG.STD)])


def tf_train():
    return T.Compose([T.RandomResizedCrop(CFG.UKURAN, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                      T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2, 0.02),
                      T.ToTensor(), T.Normalize(CFG.MEAN, CFG.STD)])


def loader(ds, bs, shuffle=False, seed=42):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=CFG.NUM_WORKERS, pin_memory=True,
                      worker_init_fn=seed_worker, generator=make_generator(seed), drop_last=shuffle)
""", tags=["definitions"]))
cells.append(md("## Embedding DINOv3\n\nEmbedding ViT-L dan ConvNeXt-L dihitung di GPU pada 384 px dengan batch 16 dan bf16."))
cells.append(code(r"""
@torch.no_grad()
def ekstrak_vit():
    model = buat_timm(*ARCH['vit'][:2], num_classes=0, drop_path_rate=0.0, **ARCH['vit'][2]).cuda().eval()
    hasil = {}
    for nama, files in (('train', train_files), ('test', test_files)):
        out = []
        for x, _ in loader(Citra(files, tf=tf_eval()), CFG.BS_EMBED):
            x = x.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out.append(model(x).float().cpu())
        e = torch.cat(out).numpy()
        hasil[nama] = (e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float32)
    del model
    torch.cuda.empty_cache()
    return np.vstack([hasil['train'], hasil['test']])


@torch.no_grad()
def ekstrak_cnn():
    model = buat_timm(*ARCH['cnn'][:2], num_classes=0).cuda().eval()
    hasil = {}
    for nama, files in (('train', train_files), ('test', test_files)):
        out = []
        for x, _ in loader(Citra(files, tf=tf_eval()), CFG.BS_EMBED):
            x = x.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out.append(model(x).float().cpu())
        e = torch.cat(out).numpy()
        hasil[nama] = (e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float32)
    del model
    torch.cuda.empty_cache()
    return np.vstack([hasil['train'], hasil['test']])


E_vit = cache_npy('emb_vit', ekstrak_vit)
E_cnn = cache_npy('emb_cnn', ekstrak_cnn)
E_tr, E_te = E_vit[:nt], E_vit[nt:]
Ec_tr, Ec_te = E_cnn[:nt], E_cnn[nt:]
np.save(CACHE / 'ids_train.npy', ids)
np.save(CACHE / 'ids_test.npy', ids_te)
print('embedding', E_vit.shape, E_cnn.shape)
"""))
cells.append(md("## Softmax ImageNet-1k\n\nConvNeXtV2-Huge ImageNet-1k dijalankan dengan flip TTA. Keluarannya dipakai sebagai sumber nama cluster utama."))
cells.append(code(r"""
class CitraIn1k(Dataset):
    def __init__(self, files, proc):
        self.files, self.proc = files, proc

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as im:
            rgb = im.convert('RGB')
        return self.proc(images=[rgb, rgb.transpose(Image.FLIP_LEFT_RIGHT)], return_tensors='pt')['pixel_values']


@torch.no_grad()
def infer_in1k():
    src = str(CFG.CACHE_DIR / 'convnextv2_huge_clean')
    proc = AutoImageProcessor.from_pretrained(src, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(src, local_files_only=True, dtype=torch.bfloat16).cuda().eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    nama_kelas = [id2label[i].split(',')[0].strip() for i in range(len(id2label))]
    out = []
    for files in (train_files, test_files):
        for px in loader(CitraIn1k(files, proc), CFG.BS_IN1K):
            b, v = px.shape[:2]
            px = px.flatten(0, 1).cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logit = model(pixel_values=px).logits
            out.append(torch.softmax(logit.float(), 1).view(b, v, -1).mean(1).cpu())
    del model
    torch.cuda.empty_cache()
    (CACHE / 'nama_in1k.json').write_text(json.dumps(nama_kelas), encoding='utf-8')
    return torch.cat(out).numpy().astype(np.float32)


p_in1k = cache_npy('p_in1k', infer_in1k)
nama_in1k = json.loads((CACHE / 'nama_in1k.json').read_text(encoding='utf-8'))
print('ImageNet-1k', p_in1k.shape)
"""))
cells.append(md("## Softmax ImageNet-22k\n\nTiga classifier ImageNet-22k berbeda arsitektur menyediakan kosakata yang tidak ada di ImageNet-1k, seperti baterai, kaleng, dan foil. Tidak ada model berbasis CLIP di daftar ini."))
cells.append(code(r"""
@torch.no_grad()
def infer_22k(nama):
    repo, arch, ukuran = MODEL_22K[nama]
    model = buat_timm(repo, arch).cuda().eval()
    data_cfg = timm.data.resolve_model_data_config(model)
    tf = T.Compose([T.Resize((ukuran, ukuran), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(),
                    T.Normalize(data_cfg['mean'], data_cfg['std'])])
    out = []
    for files in (train_files, test_files):
        for x, _ in loader(Citra(files, tf=tf), CFG.BS_IN22K):
            x = x.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p = (torch.softmax(model(x).float(), 1) + torch.softmax(model(x.flip(3)).float(), 1)) / 2
            out.append(p.cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(out).numpy().astype(np.float32)


P22 = {m: cache_npy(f'p22k_{m}', lambda m=m: infer_22k(m)) for m in MODEL_22K}
print({m: v.shape for m, v in P22.items()})
"""))
cells.append(md("## Material DMS46\n\nDMS46 dari Apple memberi porsi piksel material per citra. Porsi dihitung pada seluruh citra dan pada area tengah, lalu dirata-rata."))
cells.append(code(r"""
DMS46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33, 34, 35,
         36, 37, 38, 39, 41, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 56]
taksonomi = json.loads((CFG.DMS_DIR / 'taxonomy.json').read_text(encoding='utf-8'))
nama_dms = [taksonomi['names'][i] for i in DMS46]


def siapkan_dms(path):
    with Image.open(path) as im:
        img = im.convert('RGB')
    w, h = img.size
    s = min(512 / h, 512 / w)
    img = img.resize((math.ceil(s * w), math.ceil(s * h)), Image.LANCZOS)
    x = torch.from_numpy(np.array(img).transpose((2, 0, 1))).float()
    return T.Normalize([v * 255 for v in CFG.MEAN], [v * 255 for v in CFG.STD])(x).unsqueeze(0)


@torch.no_grad()
def infer_dms():
    model = torch.jit.load(str(CFG.DMS_DIR / 'extracted' / 'DMS46_v1.pt')).cuda().eval()
    files = train_files + test_files
    semua, tengah = np.zeros((len(files), 46), np.float32), np.zeros((len(files), 46), np.float32)
    for i, p in enumerate(files):
        m = model(siapkan_dms(p).cuda())[0].data.cpu()[0, 0].numpy().astype(int)
        h, w = m.shape
        semua[i] = np.bincount(m.ravel(), minlength=46)[:46] / m.size
        c = m[h // 4:h - h // 4, w // 4:w - w // 4]
        tengah[i] = np.bincount(c.ravel(), minlength=46)[:46] / c.size
    del model
    torch.cuda.empty_cache()
    return 0.5 * semua + 0.5 * tengah


D_dms = cache_npy('dms', infer_dms)
print('DMS46', D_dms.shape)
"""))
cells.append(md(
    "#### Insights\n\n"
    "> ImageNet-1k tidak punya kelas baterai, kaleng, dan foil, sehingga benda seperti baterai jatuh ke `metal`. Karena itu ImageNet-22k diperlukan sebagai kosakata tambahan.\n\n"
    "> DMS46 memberi nama material secara langsung, tetapi layar elektronik dilabeli glass, sehingga DMS tidak dipakai sebagai bukti `electronic`."
))

# ---------------------------------------------------------------------------
# Section 4: Label Dasar
# ---------------------------------------------------------------------------
cells.append(md(
    "## Pakai Cache Label\n\n"
    "Bila `label_final.csv` sudah ada, section 4 sampai 9 dilewati dan notebook langsung ke training final. "
    "Softmax, embedding, dan prediksi model juga dimuat dari `OUTPUT_DIR/cache`, jadi run ulang tidak menghitung apa pun yang sudah selesai. Hapus berkasnya untuk menghitung ulang."
))
cells.append(code(r"""
LEWATI_LABEL = (CFG.OUTPUT_DIR / 'label_final.csv').exists()
print('rantai label dilewati:', LEWATI_LABEL)
"""))
i_mulai = len(cells)
cells.append(section("Label Dasar", "4"))
cells.append(md(
    "Label dasar dibentuk dari pemetaan kelas ImageNet-1k ke delapan kategori berdasarkan nama kelas. Massa softmax dijumlahkan per kategori, "
    "lalu dinormalkan dengan prior train. Tanpa normalisasi, `plastic` memegang 0,319 massa dan menyerap kelas lain."
))
cells.append(code(r"""
PETA_IN1K = {
    'glass': ['beer bottle', 'beer glass', 'wine bottle', 'goblet', 'vase', 'whiskey jug', 'pitcher', 'beaker', 'measuring cup', 'perfume', 'Petri dish'],
    'paper': ['envelope', 'binder', 'paper towel', 'toilet tissue', 'comic book', 'crossword puzzle', 'book jacket', 'menu', 'web site'],
    'electronic': ['cellular telephone', 'computer keyboard', 'desktop computer', 'dial telephone', 'hand-held computer', 'hard disc', 'iPod', 'laptop',
                   'loudspeaker', 'modem', 'monitor', 'mouse', 'notebook', 'pay-phone', 'photocopier', 'printer', 'projector', 'radio', 'remote control',
                   'screen', 'space bar', 'television', 'typewriter keyboard', 'cassette player', 'CD player', 'tape player', 'cassette', 'digital clock',
                   'digital watch', 'joystick', 'microphone', 'oscilloscope', 'Polaroid camera', 'reflex camera', 'switch', 'home theater',
                   'entertainment center', 'microwave', 'toaster', 'vacuum', 'electric fan', 'hand blower', 'refrigerator', 'washer', 'dishwasher',
                   'espresso maker', 'waffle iron', 'iron', 'space heater', 'sewing machine', 'cash machine'],
    'metal': ['bottlecap', 'cocktail shaker', 'milk can', 'bucket', 'caldron', 'Dutch oven', 'frying pan', 'wok', 'chain', 'nail', 'screw', 'screwdriver',
              'hammer', 'hatchet', 'cleaver', 'letter opener', 'safety pin', 'buckle', 'padlock', 'combination lock', 'hook', 'steel drum', 'strainer',
              'ladle', 'spatula', 'can opener', 'corkscrew', 'oil filter', 'hair spray', 'lighter', 'disk brake', 'car wheel', 'manhole cover', 'safe',
              'coffeepot', 'brass', 'shovel', 'radiator', 'rain barrel'],
    'plastic': ['pop bottle', 'water bottle', 'water jug', 'plastic bag', 'packet', 'pill bottle', 'lotion', 'soap dispenser', 'sunscreen', 'tub', 'syringe', 'tray'],
    'cardboard': ['carton', 'crate', 'pencil box'],
    'textile': ['abaya', 'academic gown', 'apron', 'bath towel', 'bib', 'bikini', 'bolo tie', 'bonnet', 'bow tie', 'brassiere', 'cardigan', 'cloak', 'dishrag',
                'doormat', 'feather boa', 'fur coat', 'gown', 'handkerchief', 'hoopskirt', 'jean', 'jersey', 'kimono', 'lab coat', 'maillot', 'military uniform',
                'miniskirt', 'mitten', 'overskirt', 'pajama', 'pillow', 'poncho', 'prayer rug', 'quilt', 'sarong', 'sock', 'stole', 'suit', 'sweatshirt',
                'swimming trunks', 'trench coat', 'velvet', 'vestment', 'Windsor tie', 'wool', 'backpack', 'purse', 'knot', 'sleeping bag', 'cowboy hat',
                'sombrero', 'clog', 'cowboy boot', 'Loafer', 'running shoe', 'sandal', 'bathing cap', 'Christmas stocking', 'teddy'],
    'foodwaste': ['guacamole', 'consomme', 'hot pot', 'trifle', 'ice cream', 'ice lolly', 'French loaf', 'bagel', 'pretzel', 'cheeseburger', 'hotdog',
                  'mashed potato', 'head cabbage', 'broccoli', 'cauliflower', 'zucchini', 'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber',
                  'artichoke', 'bell pepper', 'cardoon', 'mushroom', 'Granny Smith', 'strawberry', 'orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit',
                  'custard apple', 'pomegranate', 'carbonara', 'chocolate sauce', 'dough', 'meat loaf', 'pizza', 'potpie', 'burrito', 'corn', 'acorn', 'hip',
                  'ear', 'plate', 'soup bowl'],
}


def skor_in1k(p1k):
    M = np.zeros((len(NAMES), len(nama_in1k)), dtype=np.float32)
    for c, daftar in enumerate(NAMES):
        for n in PETA_IN1K[daftar]:
            idx = [i for i, x in enumerate(nama_in1k) if x == n]
            assert idx, f'nama kelas tidak ditemukan: {n}'
            M[c, idx] = 1
    massa = (M @ p1k.T).T
    cakupan = massa.sum(1)
    relatif = massa / np.clip(cakupan[:, None], 1e-9, None)
    return relatif.astype(np.float64).round(5) * cakupan[:, None].astype(np.float64).round(5)


S_in1k = skor_in1k(p_in1k)
print('porsi massa train:', dict(zip(NAMES, (S_in1k[:nt].mean(0) / S_in1k[:nt].mean(0).sum()).round(3).tolist())))
"""))
cells.append(md(
    "## Bagging KMeans\n\n"
    "Satu run KMeans tidak stabil, dengan ARI antar seed sekitar 0,65. Karena itu dipakai 36 run: K 48, 64, dan 96, masing-masing 12 seed, "
    "pada PCA 128 dimensi yang dinormalkan. KMeans hanya di-fit pada train. Nama cluster keluar dari argmax log-odds setelah normalisasi prior."
))
cells.append(code(r"""
pca = PCA(CFG.PCA_DIM, random_state=42).fit(E_tr)


def proyeksi(x):
    v = pca.transform(x)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


Ztr, Zte = proyeksi(E_tr), proyeksi(E_te)
runs = []
for k in CFG.KS:
    for s in range(CFG.KMEANS_SEEDS):
        km = KMeans(k, n_init=1, random_state=1000 * k + s).fit(Ztr)
        runs.append((k, km.labels_, km.predict(Zte)))
print('run', len(runs))


def relatif(v):
    pri = v[:nt].mean(0)
    r = np.divide(v, pri, out=np.zeros_like(v, dtype=np.float64), where=pri > 0)
    return r, pri > 0


def bag(skor_daftar):
    V = np.zeros((N, CFG.N_KELAS))
    for k, lab, lab_te in runs:
        L = np.zeros((k, CFG.N_KELAS))
        for skor in skor_daftar:
            r, ada = relatif(skor)
            v = r[:nt]
            rata = np.stack([v[lab == c].mean(0) if (lab == c).any() else np.ones(CFG.N_KELAS) for c in range(k)])
            rata = rata / np.maximum(rata.sum(1, keepdims=True), 1e-9)
            lr = np.log(rata + 1e-3)
            lr[:, ~ada] = np.log(1 / CFG.N_KELAS)
            L += lr
        V[np.arange(N), L.argmax(1)[np.r_[lab, lab_te]]] += 1
    return V / len(runs)


V_in1k = bag([S_in1k])
print('suara train >= 0,75:', int((V_in1k[:nt].max(1) >= CFG.SUARA_MIN).sum()))
"""))
cells.append(md(
    "## Group Citra Mirip\n\n"
    "Banyak citra train adalah keluarga augmentasi. Group dibentuk dari kemiripan cos DINOv3 minimal 0,9 atau jarak Hamming pHash dihedral maksimal 4, "
    "hanya pada pasangan train. Group mencegah citra mirip terpisah antara data latih dan hold-out."
))
cells.append(code(r"""
def phash_dari(a):
    d = dctn(a, norm='ortho')[:8, :8].ravel()
    d = d[1:]
    return (d > np.median(d)).astype(np.uint8)


def hash_semua(p):
    with Image.open(p) as im:
        g = np.asarray(im.convert('L').resize((32, 32), Image.LANCZOS), dtype=np.float32)
    out = []
    for fl in (False, True):
        b = np.fliplr(g) if fl else g
        for r in range(4):
            out.append(phash_dari(np.rot90(b, r)))
    return np.array(out)


H = cache_npy('phash', lambda: np.stack([hash_semua(p) for p in train_files]).astype(bool))
basis = H[:, 0, :].astype(np.float32) * 2 - 1
sel = np.stack([(63 - (H[:, t, :].astype(np.float32) * 2 - 1) @ basis.T) / 2 for t in range(8)])
terbaik = sel.min(0)
np.fill_diagonal(terbaik, 99)
pa_h, pb_h = np.where(np.triu(np.minimum(terbaik, terbaik.T) <= CFG.HAMMING_GROUP, 1))

Sim = E_tr @ E_tr.T
np.fill_diagonal(Sim, 0)
a_s, b_s = np.where(Sim >= CFG.SIM_GROUP)
ii, jj = np.r_[a_s, pa_h], np.r_[b_s, pb_h]
n_group, grp = connected_components(csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(nt, nt)), directed=False)
ukuran_group = pd.Series(grp).value_counts()
print('group', n_group, 'terbesar', int(ukuran_group.max()), 'citra dalam group > 1:', int(ukuran_group[ukuran_group > 1].sum()))
"""))
cells.append(md("## Label t1\n\nLabel dipakai bila porsi suara minimal 0,75. Hold-out memakai StratifiedGroupKFold 5 fold, fold 0, seed 42."))
cells.append(code(r"""
y_t1 = V_in1k.argmax(1)
suara = V_in1k.max(1)
df_t1 = pd.DataFrame({'image_id': ids, 'label': np.array(NAMES)[y_t1[:nt]], 'suara': suara[:nt].round(4), 'group': grp})
t1 = df_t1[suara[:nt] >= CFG.SUARA_MIN].reset_index(drop=True)
_, va_t1 = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(t1, t1.label, t1.group))
t1['fold'] = 1
t1.loc[va_t1, 'fold'] = 0
assert not (t1.groupby('group').fold.nunique() > 1).any()
print('label t1:', len(t1), t1.label.value_counts().reindex(NAMES).fillna(0).astype(int).to_dict())
"""))
cells.append(md(
    "#### Insights\n\n"
    "> Normalisasi prior adalah perubahan terbesar dalam pipeline. Tanpa normalisasi, `plastic` menyerap sebagian besar `paper` dan `glass`.\n\n"
    "> Citra yang suaranya di bawah 0,75 sengaja dibiarkan kosong pada tahap ini, karena clusternya tidak konsisten antar run."
))

# ---------------------------------------------------------------------------
# Section 5: Pemeriksaan Independen
# ---------------------------------------------------------------------------
cells.append(section("Pemeriksaan Independen", "5"))
cells.append(md(
    "ImageNet-22k dan DMS46 tidak ikut membentuk label t1, jadi keduanya bisa memeriksa label dari luar. Peta kategori dibuat dari nama resmi kelas "
    "lewat regex dan rentang synset WordNet, lalu dibekukan sebelum hasil inferensi dilihat. Tidak ada keputusan per cluster."
))
cells.append(code(r"""
DMS_PETA = {'Cardboard': 'cardboard', 'Carpet/rug': 'textile', 'Fabric/cloth': 'textile', 'Food': 'foodwaste', 'Glass': 'glass',
            'Metal': 'metal', 'Paper': 'paper', 'Plastic, clear': 'plastic', 'Plastic, non-clear': 'plastic'}
LEMMA = {
    'cardboard': r'\b(cardboard|carton|corrugated|paperboard|pasteboard|boxboard|cardboard box)\b',
    'paper': r'\b(paper|newspaper|newsprint|magazine|book|paperback|hardback|notebook|notepad|envelope|tissue|napkin|postcard|'
             r'brochure|pamphlet|leaflet|flyer|handbill|receipt|stationery|poster|comic|ticket|manuscript|letter paper|paper towel|document)\b',
    'plastic': r'\b(plastic|plastics|polyethylene|polystyrene|styrofoam|cellophane|plastic bag|garbage bag|shopping bag|water bottle|'
               r'pop bottle|soda bottle|squeeze bottle|tupperware|bubble wrap|shrink wrap|plastic wrap)\b',
    'glass': r'\b(glass|glassware|goblet|wineglass|tumbler|beaker|carafe|decanter|flagon|snifter|vial|phial|ampule|jar|wine bottle|beer bottle|'
             r'vase|champagne flute|demijohn|carboy)\b',
    'metal': r'\b(can|tin can|tin|beer can|soda can|canister|aluminum|aluminium|foil|tinfoil|steel|ironware|copper|brass|bronze|metal|metalware|'
             r'scrap metal|cutlery|tableware fork|nail|screw|bolt|nut and bolt|washer|pliers|wrench|spanner|hammer|saucepan|frying pan|skillet|kettle)\b',
    'textile': r'\b(textile|fabric|cloth|clothing|garment|apparel|shirt|t-shirt|tee shirt|sweater|jumper|jeans|trousers|pants|dress|skirt|blouse|'
               r'jacket|coat|sock|towel|blanket|curtain|rug|carpet|scarf|shawl|hoodie|sweatshirt|underwear|pajama|denim|wool|cotton|linen|silk|tweed)\b',
    'electronic': r'\b(computer|laptop|notebook computer|desktop computer|telephone|phone|smartphone|cellular telephone|television|tv|monitor|radio|printer|'
                  r'keyboard|loudspeaker|speaker|headphone|earphone|charger|battery|circuit|circuit board|transistor|remote control|calculator|router|modem|'
                  r'camera|camcorder|video recorder|stereo|amplifier|tuner|microwave|toaster|hair dryer|vacuum cleaner|electric fan|blender|electric)\b',
}
GLOSS = {
    'cardboard': r'\b(cardboard|paperboard|pasteboard|corrugated)\b',
    'paper': r'\b(made of paper|of paper|paper container|sheet of paper)\b',
    'plastic': r'\b(made of plastic|of plastic|plastic container|plastic)\b',
    'glass': r'\b(made of glass|of glass|glass container|glass bottle|glass vessel|blown glass)\b',
    'metal': r'\b(metal|metallic|aluminum|aluminium|steel|cast iron|tin)\b',
    'textile': r'\b(garment|clothing|fabric|textile|woven|knitted|cloth|worn (by|on|over|under))\b',
    'electronic': r'\b(electronic|electrical device|electric device|computer|circuit|powered by electricity|electrically)\b',
}
BUANG_LEMMA = r'\b(sunglass|sunglasses|spyglass|hourglass|eyeglass|eyeglasses|magnifying glass|looking glass|field glass|opera glass|glasses|stained glass|flypaper|sandpaper|wallpaper|' \
              r'paper clip|paperweight|can opener|tin opener|fiberglass|fibreglass|bookcase|bookshelf|bookend|cannon|canopy|nut|washer|electric chair|electric guitar|electric organ)\b'
info22 = ImageNetInfo('imagenet-22k')


def boleh_synset(s):
    n = int(s[1:])
    return 2666000 <= n < 4990000 or 6000000 <= n < 7000000 or 7555000 <= n < 8000000 or 14000000 <= n < 16000000


def makanan_synset(s):
    n = int(s[1:])
    return 7555000 <= n < 7881800 or 7933000 <= n < 8000000


def peta22k():
    n = info22.num_classes()
    M = np.zeros((8, n), dtype=np.float32)
    for i in range(n):
        s = info22.index_to_label_name(i)
        lem, _, gl = info22.index_to_description(i, detailed=True).partition(':')
        lem, gl = lem.lower(), gl.lower()
        if not boleh_synset(s) or re.search(BUANG_LEMMA, lem):
            continue
        if makanan_synset(s):
            M[7, i] = 1
            continue
        kena = [c for c in LEMMA if re.search(LEMMA[c], lem)]
        if not kena:
            kena = [c for c in GLOSS if re.search(GLOSS[c], gl)]
        for c in kena:
            M[ID[c], i] = 1 / len(kena)
    return M


def peta_dms():
    M = np.zeros((8, 46), dtype=np.float32)
    for k, c in DMS_PETA.items():
        M[ID[c], nama_dms.index(k)] = 1
    return M


M22, Md = peta22k(), peta_dms()
skor_22k = {m: v @ M22.T for m, v in P22.items()}
S_in22k = np.mean(list(skor_22k.values()), axis=0)
S_dms = D_dms @ Md.T
print('kelas 22k terpetakan:', int((M22.sum(0) > 0).sum()))
"""))
cells.append(md(
    "## Dawid-Skene\n\n"
    "Enam annotator dipakai: argmax per citra dari ImageNet-1k, ImageNet-22k, dan DMS, nama cluster dari 22k dan DMS, serta LR cross-fit pada embedding. "
    "Annotator satu kanal diberi bobot 0,5. Nama cluster in1k tidak ikut sebagai annotator, supaya label lama tidak mengunci diri. DMS netral untuk `electronic`."
))
cells.append(code(r"""
def lengkap(p, kelas):
    q = np.zeros((len(p), CFG.N_KELAS))
    q[:, kelas] = p
    return q


d2 = pd.DataFrame({'image_id': ids}).merge(t1[['image_id', 'label', 'suara', 'fold']], on='image_id', how='left')
d2['group'] = grp
d2['c_in1k'] = np.array(NAMES)[relatif(S_in1k)[0].argmax(1)[:nt]]
d2['c_in22k'] = np.array(NAMES)[relatif(S_in22k)[0].argmax(1)[:nt]]
d2['c_dms'] = np.array(NAMES)[relatif(S_dms)[0].argmax(1)[:nt]]
d2['b_in1k'] = np.array(NAMES)[V_in1k[:nt].argmax(1)]
d2['b_in22k'] = np.array(NAMES)[bag([S_in22k])[:nt].argmax(1)]
d2['b_dms'] = np.array(NAMES)[bag([S_dms])[:nt].argmax(1)]

ada_label = d2.label.notna().values
idx2 = np.where(ada_label)[0]
y2 = d2.label[idx2].map(ID).values
pe = np.zeros((nt, CFG.N_KELAS))
for tr_, va_ in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(idx2, y2, grp[idx2]):
    m = LogisticRegression(C=3, max_iter=3000).fit(E_tr[idx2[tr_]], y2[tr_])
    pe[idx2[va_]] = lengkap(m.predict_proba(E_tr[idx2[va_]]), m.classes_)
    pe[~ada_label] += lengkap(m.predict_proba(E_tr[~ada_label]), m.classes_) / 5
d2['e_lr'] = np.array(NAMES)[pe.argmax(1)]

ANN = list(CFG.BOBOT_DS)
R_ds = np.stack([d2[k].map(ID).values for k in ANN], 1)
W_ds = np.array([CFG.BOBOT_DS[k] for k in ANN])
D_IDX = [i for i, k in enumerate(ANN) if k.endswith('_dms')]


def ds_post(R, pi, C):
    L = np.tile(np.log(pi), (len(R), 1))
    for j, w in enumerate(W_ds):
        L += w * np.log(C[j, :, R[:, j]])
    q = np.exp(L - L.max(1, keepdims=True))
    return q / q.sum(1, keepdims=True)


def ds_fit(R):
    q = np.zeros((nt, 8))
    for j, w in enumerate(W_ds):
        q[np.arange(nt), R[:, j]] += w
    q = q / q.sum(1, keepdims=True)
    for it in range(CFG.ITER_DS):
        pi = (q.sum(0) + 1) / (q.sum() + 8)
        C = np.zeros((len(ANN), 8, 8))
        for j in range(len(ANN)):
            for r in range(8):
                C[j, :, r] = q[R[:, j] == r].sum(0)
        C = (C + 1) / (C + 1).sum(2, keepdims=True)
        for j in D_IDX:
            nonel = [c for c in range(8) if c != EL]
            C[j, EL, :] = (C[j, nonel, :] * pi[nonel, None]).sum(0) / pi[nonel].sum()
        q = ds_post(R, pi, C)
    return q


q_ds = ds_fit(R_ds)
d2['ds'] = np.array(NAMES)[q_ds.argmax(1)]
print('prediksi DS:', d2.ds.value_counts().reindex(NAMES).fillna(0).astype(int).to_dict())
"""))
cells.append(md(
    "## Aturan Koreksi v2\n\n"
    "Koreksi butuh posterior minimal 0,90, dukungan minimal dua kanal, dan dukungan lebih banyak dari label lama. Penambahan label juga butuh LR setuju. "
    "Bila `electronic` terlibat, dukungan dihitung dari in1k dan in22k saja, dan kelas baru wajib didukung keduanya."
))
cells.append(code(r"""
def dukungan(r, c, tanpa_dms):
    n = int(r['c_in1k'] == c or r['b_in1k'] == c) + int(r['c_in22k'] == c or r['b_in22k'] == c)
    if not tanpa_dms:
        n += int(r['c_dms'] == c or r['b_dms'] == c)
    return n


keputusan, akhir = [], []
for i, r in enumerate(d2.to_dict('records')):
    L = r['label'] if isinstance(r['label'], str) else None
    c = r['ds']
    qc = q_ds[i, ID[c]]
    el = c == 'electronic' or L == 'electronic'
    syarat = dukungan(r, c, el)
    cukup = (syarat == 2) if el else (syarat >= CFG.DUKUNG_MIN)
    if L is None:
        if qc >= CFG.Q_MIN and cukup and r['e_lr'] == c:
            keputusan.append('tambah'); akhir.append(c)
        else:
            keputusan.append('kosong'); akhir.append(None)
    elif c != L and qc >= CFG.Q_MIN and cukup and syarat > dukungan(r, L, el):
        keputusan.append('koreksi'); akhir.append(c)
    else:
        keputusan.append('tetap'); akhir.append(L)
d2['keputusan'], d2['akhir'] = keputusan, akhir

gf = t1.groupby('group').fold.first()
v2 = d2[d2.akhir.notna()][['image_id', 'akhir', 'suara', 'group', 'fold']].rename(columns={'akhir': 'label'}).copy()
v2['fold'] = [f if f == f else gf.get(g, 1) for f, g in zip(v2.fold, v2.group)]
v2['fold'] = v2.fold.astype(int)
v2 = v2.sort_values('image_id').reset_index(drop=True)
print('keputusan:', d2.keputusan.value_counts().to_dict(), '| label v2:', len(v2))
"""))
cells.append(md(
    "#### Insights\n\n"
    "> Bukti independen hanya mengoreksi sedikit label. Tambahan terbesar justru datang dari citra sulit yang sebelumnya kosong, terutama pada keluarga citra 416 x 416.\n\n"
    "> Bias tiap sumber kuat, misalnya DMS membaca banyak perangkat elektronik sebagai plastik, sehingga satu sumber saja tidak boleh membalik label."
))

# ---------------------------------------------------------------------------
# Section 6: Deteksi Label Salah
# ---------------------------------------------------------------------------
cells.append(section("Deteksi Label Salah", "6"))
cells.append(md(
    "Co-teaching melatih dua head linear di atas embedding ViT dan CNN. Tiap head memilih sampel loss kecil untuk melatih head lainnya. Pada noise sintetis, "
    "metode ini menemukan 63% noise cluster 5%, sedangkan LR hanya 3%, dengan presisi 0,96. Konstanta dipilih dari noise sintetis sebelum diterapkan pada label asli."
))
cells.append(code(r"""
def lipat(y_lab, grp_lab, n, seed):
    return list(StratifiedGroupKFold(n, shuffle=True, random_state=seed).split(np.zeros(len(y_lab)), y_lab, grp_lab))


def co_teaching(Ea, Eb, y, Qa, Qb, lupa, seed, bs=128, lr=.05, wd=1e-4, kira_k=10):
    epoch = CFG.EPOCH_COTEACH
    g = torch.Generator().manual_seed(seed)
    xa, xb = torch.as_tensor(Ea, dtype=torch.float32, device='cuda'), torch.as_tensor(Eb, dtype=torch.float32, device='cuda')
    t = torch.as_tensor(y, device='cuda')
    m_a = torch.nn.Linear(xa.shape[1], CFG.N_KELAS).cuda()
    m_b = torch.nn.Linear(xb.shape[1], CFG.N_KELAS).cuda()
    opt = torch.optim.AdamW(list(m_a.parameters()) + list(m_b.parameters()), lr=lr, weight_decay=wd)
    n = len(y)
    langkah = max(1, n // bs) * epoch
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, langkah)
    for ep in range(epoch):
        buang = lupa * min(1.0, (ep + 1) / kira_k)
        urut = torch.randperm(n, generator=g).cuda()
        for i in range(0, n - bs + 1, bs):
            b = urut[i:i + bs]
            la, lb = m_a(xa[b]), m_b(xb[b])
            ra = torch.nn.functional.cross_entropy(la, t[b], reduction='none')
            rb = torch.nn.functional.cross_entropy(lb, t[b], reduction='none')
            k = int((1 - buang) * len(b))
            pa = torch.argsort(ra.detach())[:k]
            pb = torch.argsort(rb.detach())[:k]
            loss = ra[pb].mean() + rb[pa].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
    with torch.no_grad():
        pa = torch.softmax(m_a(torch.as_tensor(Qa, dtype=torch.float32, device='cuda')), 1).cpu().numpy()
        pb = torch.softmax(m_b(torch.as_tensor(Qb, dtype=torch.float32, device='cuda')), 1).cpu().numpy()
    return pa, pb


def oof_coteach(Ev, Ec, y, grp_, seed):
    R, S = CFG.R_COTEACH, CFG.S_COTEACH
    n = len(y)
    a, b = np.zeros((R * S, n, 8), np.float32), np.zeros((R * S, n, 8), np.float32)
    for r in range(R):
        split = lipat(y, grp_, CFG.N_FOLD, seed + r)
        for s in range(S):
            for f, (tr, va) in enumerate(split):
                pa, pb = co_teaching(Ev[tr], Ec[tr], y[tr], Ev[va], Ec[va], CFG.LUPA, seed + 1000 * r + 100 * s + f)
                a[r * S + s, va], b[r * S + s, va] = pa, pb
    return a, b


def putuskan(a, b, y):
    n = len(y)
    i = np.arange(n)
    tv, tc = (a.argmax(2) != y).mean(0), (b.argmax(2) != y).mean(0)
    pa, pb = a.mean(0), b.mean(0)
    m = ((pa + pb) / 2).copy()
    m[i, y] = -1
    alt = m.argmax(1)
    dua = (tv >= CFG.TOLAK) & (tc >= CFG.TOLAK)
    kor = dua & (pa.argmax(1) == alt) & (pb.argmax(1) == alt) & (pa[i, alt] >= CFG.YAKIN) & (pb[i, alt] >= CFG.YAKIN)
    return kor, dua & ~kor, alt
""", tags=["definitions"]))
cells.append(md("## Pembersihan Label t1\n\nCitra yang ditolak kedua head dikoreksi bila keduanya sepakat pada kelas lain dengan keyakinan minimal 0,9, dan selain itu dibuang."))
cells.append(code(r"""
y_lab = np.full(nt, -1)
y_lab[pd.Series(ids).isin(t1.image_id).values] = t1.label.map(ID).values
il = np.where(y_lab >= 0)[0]
torch.manual_seed(67280421310721)
a_oof, b_oof = oof_coteach(E_tr[il], Ec_tr[il], y_lab[il], grp[il], CFG.SEED_DASAR)
kor, bua, alt = putuskan(a_oof, b_oof, y_lab[il])
id_buang = ids[il][bua]
peta_koreksi = dict(zip(ids[il][kor], np.array(NAMES)[alt[kor]]))
bersih = t1.copy()
bersih.loc[bersih.image_id.isin(peta_koreksi), 'label'] = bersih.image_id.map(peta_koreksi)
bersih = bersih[~bersih.image_id.isin(id_buang)].reset_index(drop=True)
print('koreksi', int(kor.sum()), '| buang', int(bua.sum()), '| bersih', len(bersih))
"""))
cells.append(md("## Pengisian Citra Kosong\n\nHead co-teaching yang dilatih pada semua label t1 mengisi citra kosong bila kedua head sepakat dengan p minimal 0,7. Isian lalu disaring sekali lagi oleh aturan yang sama."))
cells.append(code(r"""
isisa = np.where(y_lab < 0)[0]
torch.manual_seed(67280421310721)
A, B = [], []
for s in CFG.SEED_ISI:
    a_, b_ = co_teaching(E_tr[il], Ec_tr[il], y_lab[il], E_tr[isisa], Ec_tr[isisa], CFG.LUPA, s)
    A.append(a_)
    B.append(b_)
pa_isi, pb_isi = np.mean(A, 0), np.mean(B, 0)
ua = pa_isi.argmax(1)
isi = (ua == pb_isi.argmax(1)) & (pa_isi.max(1) >= CFG.AMBANG_ISI) & (pb_isi.max(1) >= CFG.AMBANG_ISI)

pos = {k: i for i, k in enumerate(ids)}
Y_gab = np.full(nt, -1)
Y_gab[[pos[k] for k in bersih.image_id]] = bersih.label.map(ID).values
Y_gab[isisa[isi]] = ua[isi]
ii_gab = np.where(Y_gab >= 0)[0]
a2, b2 = oof_coteach(E_tr[ii_gab], Ec_tr[ii_gab], Y_gab[ii_gab], grp[ii_gab], CFG.SEED_RONDE2)
kor2, bua2, _ = putuskan(a2, b2, Y_gab[ii_gab])
adalah_isian = y_lab[ii_gab] < 0
terima = ii_gab[adalah_isian & ~(kor2 | bua2)]
fold_grup = bersih.groupby('group').fold.first().to_dict()
tambah_v3 = pd.DataFrame({'image_id': ids[terima], 'label': np.array(NAMES)[Y_gab[terima]], 'suara': V_in1k[:nt][terima].max(1).round(4),
                          'group': grp[terima], 'fold': [fold_grup.get(g, 1) for g in grp[terima]]})
v3 = pd.concat([bersih, tambah_v3], ignore_index=True).sort_values('image_id').reset_index(drop=True)
print('isian kandidat', int(isi.sum()), '| diterima', len(tambah_v3), '| label v3:', len(v3))
"""))
cells.append(md("## Label v4\n\nLabel v2 dan v3 disatukan. Isian yang bertentangan dibuang, citra diragukan dihapus, dan koreksi co-teaching diterapkan."))
cells.append(code(r"""
v2i, v3i = v2.set_index('image_id'), v3.set_index('image_id')
tambah_ids = v3i.index.difference(v2i.index)
gab = pd.concat([v2i, v3i.loc[tambah_ids].reindex(columns=v2i.columns)])
konflik = v2i.index.intersection(v3i.index)
beda = konflik[v2i.loc[konflik, 'label'].values != v3i.loc[konflik, 'label'].values]
gab = gab.drop(index=beda)
gab = gab.drop(index=[k for k in id_buang if k in gab.index])
for k, c in peta_koreksi.items():
    gab.loc[k, 'label'] = c
v4 = gab.sort_index().reset_index()
v4['fold'] = v4.fold.fillna(1).astype(int)
v4['group'] = v4.image_id.map(dict(zip(ids, grp)))
print('label v4:', len(v4), v4.label.value_counts().reindex(NAMES).fillna(0).astype(int).to_dict())
"""))
cells.append(md(
    "#### Insights\n\n"
    "> Co-teaching lebih berguna sebagai detektor daripada sebagai model akhir. Pengisian citra kosong berdampak lebih besar pada prediksi test dibanding pembersihan.\n\n"
    "> Menurut uji pada noise sintetis, soft label, label spreading, dan optimal transport tidak memberi perbaikan terukur, jadi ketiganya tidak dipakai."
))

# ---------------------------------------------------------------------------
# Section 7: Aturan Baterai
# ---------------------------------------------------------------------------
cells.append(section("Aturan Baterai", "7"))
cells.append(md(
    "ImageNet-1k tidak punya kelas baterai, sehingga benda silinder logam ini berlabel `metal`. Kosakata ImageNet-22k dipakai untuk mengukur massa baterai, lampu, dan papan sirkuit "
    "sebagai median tiga model. Bila massa terbesar minimal 0,20, label menjadi `electronic`. Ambang berasal dari celah sebaran train, tanpa ID citra dan tanpa nama per cluster."
))
cells.append(code(r"""
GRUP_LISTRIK = {
    'sel': r'\b(battery|batteries|dry cell|dry battery|storage cell|storage battery|voltaic cell|voltaic battery|galvanic cell|primary cell|secondary cell|'
           r'solar cell|fuel cell|nicad|nickel-cadmium|alkaline cell|lead-acid|energizer|weston cell|daniell cell|clark cell|cadmium cell|mercury cell)\b',
    'lampu': r'\b(light bulb|lightbulb|incandescent lamp|fluorescent lamp|fluorescent|fluorescent fixture|neon lamp|neon tube|electric lamp|bulb|floodlight|spotlight|halogen lamp|mercury-vapor lamp|sodium-vapor lamp|led)\b',
    'papan': r'\b(circuit board|printed circuit|cpu board|motherboard|mother board|circuitry|circuit|integrated circuit|chip|microchip|transistor|resistor|capacitor|diode|semiconductor device|computer circuit|logic chip|memory chip)\b',
}
BUANG_LISTRIK = r"\b(thermometer|battery park|artillery|gun battery|poker chip|potato chip|wood chip|chip shot|bulb onion|tulip|onion|barbed wire|chicken wire|wire cloth|jackfruit|jack-o'-lantern|jackstraw|jackknife|lumberjack|bootjack|car jack)\b"


def bangun_listrik():
    n = info22.num_classes()
    M = {k: np.zeros(n, dtype=bool) for k in GRUP_LISTRIK}
    for i in range(n):
        nomor = int(info22.index_to_label_name(i)[1:])
        lem = info22.index_to_description(i).lower()
        artefak = 2666000 <= nomor < 4990000 or 6000000 <= nomor < 7000000
        if re.search(BUANG_LISTRIK, lem):
            continue
        for k, r in GRUP_LISTRIK.items():
            if re.search(r, lem) and (artefak or k == 'sel'):
                M[k][i] = True
    return M


M_listrik = bangun_listrik()
massa_listrik = {k: np.sort(np.stack([p[:nt][:, v].sum(1) for p in P22.values()], 1), 1)[:, 1].round(4) for k, v in M_listrik.items()}
D_listrik = np.max(np.stack(list(massa_listrik.values()), 1), 1)
kena_listrik = pd.DataFrame({'image_id': ids[D_listrik >= CFG.AMBANG_LISTRIK], 'D': D_listrik[D_listrik >= CFG.AMBANG_LISTRIK].round(4)})

v5 = v4.copy()
ada = kena_listrik.image_id.isin(v5.image_id)
lama = v5.set_index('image_id').label
ubah = kena_listrik[ada & (kena_listrik.image_id.map(lama) != 'electronic')]
baru = kena_listrik[~ada]
m_ubah = v5.image_id.isin(ubah.image_id)
v5.loc[m_ubah, 'label'] = 'electronic'
v5.loc[m_ubah, 'suara'] = v5.image_id[m_ubah].map(dict(zip(ubah.image_id, ubah.D)))
gf5 = v5.groupby('group').fold.first()
grp_map = dict(zip(ids, grp))
tb = pd.DataFrame({'image_id': baru.image_id, 'label': 'electronic', 'suara': baru.D, 'group': baru.image_id.map(grp_map)})
tb['fold'] = [int(gf5.get(g, 1)) for g in tb.group]
v5 = pd.concat([v5, tb], ignore_index=True).sort_values('image_id').reset_index(drop=True)
print('kena', len(kena_listrik), '| diubah', len(ubah), '| baru', len(baru), '| label v5:', len(v5))
"""))
cells.append(md(
    "#### Insights\n\n"
    "> Pencemaran label `metal` terkonsentrasi pada satu keluarga sumber, yaitu citra web: 10 dari 22 label metal di keluarga itu ternyata baterai.\n\n"
    "> Aturan ini mengasumsikan baterai masuk kategori `electronic`. Asumsi ini berasal dari definisi kelas, yaitu komponen elektronik, bukan dari label."
))

# ---------------------------------------------------------------------------
# Section 8: Training Tahap Pertama
# ---------------------------------------------------------------------------
cells.append(section("Training Tahap Pertama", "8"))
cells.append(md(
    "DINOv3 ViT-L dan ConvNeXt-L di-fine-tune pada 384 px selama 8 epoch dengan layer decay 0,75, label smoothing 0,1, dan bobot kelas invers akar frekuensi. "
    "Batch 8 dengan accumulation 2 menjaga VRAM di bawah batas. Resolusi 448 sudah diuji dan lebih buruk, jadi 384 dipakai. Model tahap ini mengisi citra kosong pada section berikutnya."
))
cells.append(code(r"""
def bobot_kelas(y):
    n = np.bincount(y, minlength=8).astype(float)
    w = 1 / np.sqrt(np.maximum(n, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def prediksi(model, files):
    if not files:
        return np.zeros((0, 8))
    model.eval()
    hasil = []
    for x, _ in loader(Citra(files, tf=tf_eval()), CFG.BS_EVAL):
        x = x.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p = torch.softmax(model(x).float(), 1) + torch.softmax(model(x.flip(3)).float(), 1)
        hasil.append((p / 2).cpu())
    return torch.cat(hasil).numpy()


def latih(arch, label_df, tag):
    p_te, p_si = CACHE / f'{tag}_test.npy', CACHE / f'{tag}_sisa.npy'
    df = label_df.sort_values('image_id').reset_index(drop=True)
    sisa_ids = sorted(k for k in ids if k not in set(df.image_id))
    if p_te.exists() and p_si.exists():
        return np.load(p_te), pd.DataFrame(np.load(p_si), index=sisa_ids, columns=NAMES)
    seed_everything(CFG.SEED)
    y = df.label.map(ID).values
    files = [peta_file[i] for i in df.image_id]
    repo, nama, kw = ARCH[arch]
    model = buat_timm(repo, nama, num_classes=8, drop_path_rate=CFG.DROP_PATH, **kw).cuda()
    opt = create_optimizer_v2(model, opt='adamw', lr=CFG.LR, weight_decay=CFG.WD, layer_decay=CFG.LAYER_DECAY)
    puncak = max(g.get('lr_scale', 1.0) for g in opt.param_groups)
    for g in opt.param_groups:
        g['lr_scale'] = g.get('lr_scale', 1.0) * (CFG.HEAD_MULT if g.get('lr_scale', 1.0) >= puncak else 1.0)
    dl = loader(Citra(files, y, tf_train()), CFG.BS, shuffle=True, seed=CFG.SEED)
    per_epoch = len(dl) // CFG.ACCUM
    total, warm = per_epoch * CFG.EPOCHS, per_epoch * CFG.WARMUP_EPOCH
    w = bobot_kelas(y).cuda()
    step, t0 = 0, time.time()
    for ep in range(CFG.EPOCHS):
        model.train()
        opt.zero_grad(set_to_none=True)
        rugi = []
        for i, (x, t) in enumerate(dl):
            x, t = x.cuda(non_blocking=True), t.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(x).float(), t, weight=w, label_smoothing=CFG.LABEL_SMOOTHING)
            (loss / CFG.ACCUM).backward()
            rugi.append(loss.item())
            if (i + 1) % CFG.ACCUM == 0:
                faktor = min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * step / total)))
                for g in opt.param_groups:
                    g['lr'] = CFG.LR * g['lr_scale'] * faktor
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
        print(f'{tag} epoch {ep + 1}/{CFG.EPOCHS} loss {np.mean(rugi):.4f} t {time.time() - t0:.0f}s vram {torch.cuda.max_memory_reserved() / 2**30:.1f}GiB', flush=True)
    pt = prediksi(model, list(test_files))
    ps = prediksi(model, [peta_file[k] for k in sisa_ids])
    np.save(p_te, pt)
    np.save(p_si, ps)
    del model, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return pt, pd.DataFrame(ps, index=sisa_ids, columns=NAMES)
""", tags=["definitions"]))
cells.append(code(r"""
pt_vit5, ps_vit5 = latih('vit', v5, 'vit_v5')
pt_cnn5, ps_cnn5 = latih('cnn', v5, 'cnn_v5')
print('tahap pertama selesai', pt_vit5.shape, pt_cnn5.shape)
"""))

# ---------------------------------------------------------------------------
# Section 9: Aturan Foil dan Pengisian
# ---------------------------------------------------------------------------
cells.append(section("Aturan Foil dan Pengisian", "9"))
cells.append(md(
    "Foil berlabel `paper` karena ImageNet-1k tidak punya kelas foil. Massa `foil` ImageNet-22k bimodal, dengan celah kosong antara 0,05 dan 0,2, jadi citra dengan massa minimal 0,20 "
    "menjadi `metal`. Citra kosong lain hanya diisi bila ViT dan CNN sepakat, rata-rata p minimal 0,80, LR cross-fit setuju, dan ada bukti di luar embedding."
))
cells.append(code(r"""
GRUP_MATERIAL = {
    'foil': r'\b(tinfoil|tin foil|aluminum foil|aluminium foil|foil|silver paper|gold foil|gold leaf|metal foil)\b',
    'kain': r'\b(diaper|nappy|napkin|serviette|handkerchief|hankie|towel|washcloth|dishcloth|dishrag|rag|tatter|sheet|bedsheet|gauze|cheesecloth|muslin|cotton|tissue paper|facial tissue|tippet|cloth|fabric)\b',
    'kertas': r'\b(paper|newspaper|tabloid|gazette|daily|envelope|file folder|folder|notebook|exercise book|magazine|pamphlet|brochure|leaflet|flyer|printout|document|wrapping paper|crepe paper|tissue paper|paper towel)\b',
    'kardus': r'\b(carton|cardboard|packing box|packing case|bandbox|chipboard|corrugated|mailer|shoebox|box|pasteboard|paperboard)\b',
    'kemasan': r'\b(packet|sachet|packaging|package|parcel|wrapper|wrapping|blister pack|pouch|cellophane|plastic wrap|shrink-wrap|bin liner|plastic bag|grocery bag|shopping bag|garbage bag)\b',
    'botol_plastik': r'\b(bottled water|mineral water|branch water|seltzer|water bottle|drinking water|soda water|pop bottle)\b',
    'kaca': r'\b(phial|vial|jar|jampot|mason jar|glass|tumbler|flute|liqueur glass|wineglass|highball glass|shot glass|flagon|carboy|demijohn|carafe|decanter|goblet|champagne cup|wine bottle|beer bottle)\b',
    'kaleng': r'\b(can|tin|beer can|soda can|coffee can|canister|tinderbox|aerosol|canned food)\b',
}
BUANG_MATERIAL = r'\b(hydrofoil|thermometer|wallpaper|sandpaper|flypaper|paper clip|paperweight|box wrench|box spring|sentry box|music box|penalty box|sound box|box kite|box camera|jury box|glasses|sunglasses|spyglass|hourglass|field glass|opera glass|magnifying glass|looking glass|bird|fish|tree|plant)\b'


def bangun_material():
    n = info22.num_classes()
    M = {k: np.zeros(n, dtype=bool) for k in GRUP_MATERIAL}
    for i in range(n):
        nomor = int(info22.index_to_label_name(i)[1:])
        lem = info22.index_to_description(i).lower()
        boleh = 2666000 <= nomor < 4990000 or 6000000 <= nomor < 7000000 or 7555000 <= nomor < 8000000 or 14000000 <= nomor < 16000000
        if not boleh or re.search(BUANG_MATERIAL, lem):
            continue
        for k, r in GRUP_MATERIAL.items():
            if re.search(r, lem):
                M[k][i] = True
    return M


M_material = bangun_material()
massa_v = {k: np.median(np.stack([p[:nt][:, v].sum(1) for p in P22.values()], 1), 1).round(4) for k, v in M_material.items()}
kolom_dms = {'Paper': ['Paper'], 'Cardboard': ['Cardboard'], 'Metal': ['Metal'], 'Glass': ['Glass'], 'Plastic': ['Plastic, clear', 'Plastic, non-clear'], 'Fabric': ['Fabric/cloth']}
massa_d = {k: D_dms[:nt][:, [nama_dms.index(x) for x in v]].sum(1).round(4) for k, v in kolom_dms.items()}
d_food = D_dms[:nt][:, nama_dms.index('Food')]
VOC = {'paper': ['kertas'], 'cardboard': ['kardus'], 'metal': ['kaleng', 'foil'], 'glass': ['kaca'], 'plastic': ['botol_plastik', 'kemasan'], 'textile': ['kain']}
DMS_KELAS = {'paper': 'Paper', 'cardboard': 'Cardboard', 'metal': 'Metal', 'glass': 'Glass', 'plastic': 'Plastic', 'textile': 'Fabric'}
Sm = np.median(np.stack([v[:nt] for v in skor_22k.values()], 2), 2)

X_lr = np.hstack([normalize(E_tr), normalize(Ec_tr)])
ada4 = pd.Series(ids).isin(v4.image_id).values
idx4 = np.where(ada4)[0]
y4 = pd.Series(ids[idx4]).map(v4.set_index('image_id').label).map(ID).values
pl = np.zeros((nt, 8))
for tr_, va_ in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(idx4, y4, grp[idx4]):
    mdl = LogisticRegression(C=3, max_iter=2000).fit(X_lr[idx4[tr_]], y4[tr_])
    pl[idx4[va_]] = lengkap(mdl.predict_proba(X_lr[idx4[va_]]), mdl.classes_)
    pl[~ada4] += lengkap(mdl.predict_proba(X_lr[~ada4]), mdl.classes_) / 5
pl_kelas = np.array(NAMES)[pl.argmax(1)]

kena_foil = massa_v['foil'] >= CFG.AMBANG_FOIL
id_v5 = set(v5.image_id)
kos = np.array([k not in set(v4.image_id) for k in ids]) & ~kena_foil
baris = []
for i in np.where(kos)[0]:
    im = ids[i]
    if im not in ps_vit5.index:
        continue
    cv, cc = ps_vit5.loc[im].idxmax(), ps_cnn5.loc[im].idxmax()
    p = (ps_vit5.loc[im, cv] + ps_cnn5.loc[im, cv]) / 2
    if cv == cc:
        c = cv
        if c == 'electronic':
            voc, mat = Sm[i, ID[c]], 0.0
        elif c == 'foodwaste':
            voc, mat = Sm[i, ID[c]], d_food[i]
        else:
            voc, mat = max(massa_v[k][i] for k in VOC[c]), massa_d[DMS_KELAS[c]][i]
        luar = voc >= CFG.AMBANG_VOC or mat >= CFG.AMBANG_DMS
        if p >= CFG.P_MIN and pl_kelas[i] == c and luar:
            baris.append({'image_id': im, 'label': c, 'suara': round(float(p), 4)})
r2 = pd.DataFrame(baris, columns=['image_id', 'label', 'suara'])

v6 = v5.copy()
id_foil = ids[kena_foil]
m_foil = v6.image_id.isin(id_foil)
nilai_foil = dict(zip(id_foil, massa_v['foil'][kena_foil]))
v6.loc[m_foil, 'label'] = 'metal'
v6.loc[m_foil, 'suara'] = v6.image_id[m_foil].map(nilai_foil)
gf6 = v6.groupby('group').fold.first()
foil_baru = pd.DataFrame({'image_id': [k for k in id_foil if k not in id_v5]})
foil_baru['label'] = 'metal'
foil_baru['suara'] = foil_baru.image_id.map(nilai_foil)
br = pd.concat([foil_baru, r2], ignore_index=True)
br['group'] = br.image_id.map(grp_map)
br['fold'] = [int(gf6.get(g, 1)) for g in br.group]
v6 = pd.concat([v6, br], ignore_index=True).sort_values('image_id').reset_index(drop=True)
print('foil', int(kena_foil.sum()), '| isian R2', len(r2), '| label v6:', len(v6), v6.label.value_counts().reindex(NAMES).fillna(0).astype(int).to_dict())
v6.to_csv(CFG.OUTPUT_DIR / 'label_final.csv', index=False)
"""))
cells.append(md(
    "#### Insights\n\n"
    "> Hanya foil yang punya sinyal bimodal. Pasangan cardboard lawan paper dan glass lawan plastic menurun bertahap, jadi tidak ada aturan yang dipaksakan untuk keduanya.\n\n"
    "> Aturan foil mengasumsikan foil masuk kategori `metal`. DMS tidak mengonfirmasi ini, jadi buktinya hanya kosakata ImageNet-22k."
))

# ---------------------------------------------------------------------------
# Section 10: Training Final
# ---------------------------------------------------------------------------
i_akhir = len(cells)
cells.append(section("Training Final", "10"))
cells.append(md("Kedua backbone dilatih ulang pada label final dengan konfigurasi yang sama. Pelatihan memakai semua label, karena tidak ada ground truth train yang bisa dijadikan validasi bermakna."))
cells.append(code(r"""
if LEWATI_LABEL:
    v6 = pd.read_csv(CFG.OUTPUT_DIR / 'label_final.csv', dtype={'image_id': str})
    print('label dimuat dari cache:', len(v6))
pt_vit, _ = latih('vit', v6, 'vit_v6')
pt_cnn, _ = latih('cnn', v6, 'cnn_v6')
print('training final selesai', pt_vit.shape, pt_cnn.shape)
"""))

cells.append(md(
    "## Ensemble K-Fold\n\n"
    "Variansi antar run sekitar 5 citra, hampir sebesar sisa jarak ke target. Tiap backbone dilatih tiga kali lagi pada fold berkelompok, sehingga rata-rata probabilitas mengurangi variansi "
    "tanpa menambah arsitektur. Citra di luar fold latih ikut diprediksi dan tersimpan sebagai prediksi out-of-fold di cache."
))
cells.append(code(r"""
k_ens = np.zeros(len(v6), dtype=int)
v6_urut = v6.sort_values('image_id').reset_index(drop=True)
for k, (_, va) in enumerate(StratifiedGroupKFold(CFG.K_ENSEMBLE, shuffle=True, random_state=42).split(v6_urut, v6_urut.label, v6_urut.group)):
    k_ens[va] = k
v6_urut['k_ens'] = k_ens
assert not (v6_urut.groupby('group').k_ens.nunique() > 1).any()
pt_semua = {'vit': [pt_vit], 'cnn': [pt_cnn]}
ps_semua = {'vit': [], 'cnn': []}
for k in range(CFG.K_ENSEMBLE):
    for arch in ('vit', 'cnn'):
        p, ps = latih(arch, v6_urut[v6_urut.k_ens != k], f'{arch}_k{k}')
        pt_semua[arch].append(p)
        ps_semua[arch].append(ps)
print('anggota per backbone:', {a: len(v) for a, v in pt_semua.items()})
"""))
cells.append(md(
    "## Koreksi dari Out-of-Fold\n\n"
    "Prediksi out-of-fold fine-tune jauh lebih kuat daripada head linear. Label dikoreksi bila ViT dan CNN sepakat pada kelas lain dengan rata-rata p minimal 0,8, "
    "dan citra kosong diisi dengan syarat yang sama. Ambang 0,8 dibekukan sebelum skor dilihat. Kedua backbone lalu dilatih sekali lagi pada label ini dan masuk ke ensemble."
))
cells.append(code(r"""
def tabel_oof(arch):
    bagian = [ps_semua[arch][k].loc[v6_urut.image_id[v6_urut.k_ens == k]] for k in range(CFG.K_ENSEMBLE)]
    return pd.concat(bagian).reindex(v6_urut.image_id)


def sepakat_oof(pa, pb):
    ca, cb = pa.values.argmax(1), pb.values.argmax(1)
    i = np.arange(len(pa))
    return ca, ca == cb, (pa.values[i, ca] + pb.values[i, ca]) / 2


pa_oof, pb_oof = tabel_oof('vit'), tabel_oof('cnn')
ca_oof, sama_oof, p_oof = sepakat_oof(pa_oof, pb_oof)
kelas_oof = np.array(NAMES)[ca_oof]
ubah_oof = sama_oof & (kelas_oof != v6_urut.label.values) & (p_oof >= CFG.P_OOF)
v7 = v6_urut.drop(columns=['k_ens']).copy()
v7.loc[ubah_oof, 'label'] = kelas_oof[ubah_oof]

kos_ids = sorted(set(ids) - set(v6_urut.image_id))
ra = sum(ps_semua['vit'][k].loc[kos_ids] for k in range(CFG.K_ENSEMBLE)) / CFG.K_ENSEMBLE
rb = sum(ps_semua['cnn'][k].loc[kos_ids] for k in range(CFG.K_ENSEMBLE)) / CFG.K_ENSEMBLE
ca_kos, sama_kos, p_kos = sepakat_oof(ra, rb)
isi_oof = sama_kos & (p_kos >= CFG.P_OOF)
isian = pd.DataFrame({'image_id': np.array(kos_ids)[isi_oof], 'label': np.array(NAMES)[ca_kos[isi_oof]], 'suara': p_kos[isi_oof].round(4)})
v7 = pd.concat([v7, isian], ignore_index=True).sort_values('image_id').reset_index(drop=True)
v7.to_csv(CFG.OUTPUT_DIR / 'label_final_v7.csv', index=False)
print('koreksi', int(ubah_oof.sum()), '| isian', len(isian), '| label v7:', len(v7), v7.label.value_counts().reindex(NAMES).fillna(0).astype(int).to_dict())

for arch in ('vit', 'cnn'):
    p, _ = latih(arch, v7, f'{arch}_v7')
    pt_semua[arch].append(p)
print('anggota per backbone:', {a: len(v) for a, v in pt_semua.items()})
"""))

# ---------------------------------------------------------------------------
# Section 11: Submission
# ---------------------------------------------------------------------------
cells.append(section("Submission", "11"))
cells.append(md("Prediksi akhir adalah rata-rata probabilitas semua anggota kedua backbone dengan flip TTA, dengan bobot sama per backbone. `image_id` ditulis sebagai string tiga digit sesuai format submission."))
cells.append(code(r"""
test_pred = (np.mean(pt_semua['vit'], 0) + np.mean(pt_semua['cnn'], 0)) / 2
sub = pd.DataFrame({'image_id': ids_te, 'waste_category': np.array(NAMES)[test_pred.argmax(1)]})
if not CFG.SMOKE_TEST:
    ss = pd.read_csv(CFG.SAMPLE_SUB, dtype=str)
    assert list(sub.columns) == list(ss.columns) and (sub.image_id.values == ss.image_id.values).all()
assert sub.waste_category.isin(NAMES).all()
sub.to_csv(CFG.OUTPUT_DIR / 'submission.csv', index=False)
print(sub.waste_category.value_counts().reindex(NAMES).fillna(0).astype(int).to_dict())
print('setuju ViT lawan CNN:', round(float((pt_vit.argmax(1) == pt_cnn.argmax(1)).mean()), 4))
"""))
cells.append(md(
    "#### Insights\n\n"
    "> Model prediksi akhir hanya dua backbone berbeda arsitektur, sesuai batas yang ditetapkan. Sumber label tambahan tidak ikut memprediksi test.\n\n"
    "> Skor hold-out pada label otomatis bersifat melingkar, jadi tidak dilaporkan di notebook ini sebagai akurasi."
))

# ---------------------------------------------------------------------------
# Notebook writer
# ---------------------------------------------------------------------------
for i, c in enumerate(cells):
    teks = c["source"]
    assert "\u2014" not in teks, f"em dash di cell: {teks[:60]}"
    assert "# ---" not in teks, f"pembatas di cell: {teks[:60]}"
    if c["cell_type"] == "code":
        if i_mulai <= i < i_akhir and "definitions" not in c["metadata"].get("tags", []):
            teks = "if not LEWATI_LABEL:\n" + "".join("    " + l if l.strip() else l for l in teks.splitlines(keepends=True))
        c["source"] = teks.splitlines(keepends=True)
hash_sumber = {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in SUMBER}
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.15"},
        "ngedata": {"eksperimen": ["EXP-005", "EXP-006", "EXP-009", "EXP-010"], "sumber_sha256": hash_sumber},
    },
    "cells": cells,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(cells)} cells -> {OUT}")
