"""Builder notebook utilitas: konversi bobot DINOv3 ViT-7B timm (fp32, 26,9 GB) ke shard fp16 (~13,4 GB).

    .venv/bin/python temp/exp_014_dinov3_7b/build_konversi.py

Jalankan notebook hasilnya SEKALI di Kaggle (CPU cukup, Internet ON), lalu simpan keluarannya sebagai dataset privat.
Hanya konversi dtype tensor, tanpa forward model.
"""
import json
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).resolve().parents[2] / "notebook" / "konversi_dinov3_7b_fp16.ipynb"
HR = "---"


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
    "# DSC Gelar Rasa 2026: Konversi Bobot DINOv3 ViT-7B ke fp16\n\n"
    "*Notebook utilitas sekali jalan untuk menyiapkan dataset bobot fp16 yang dipakai notebook utama.*"
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
    "3. [**Konversi**](#3)\n"
    "4. [**Verifikasi**](#4)\n"
))

# ---------------------------------------------------------------------------
# 1. Pendahuluan
# ---------------------------------------------------------------------------
cells.append(section("Pendahuluan", "1"))
cells.append(md(
    "Bobot resmi `timm/vit_7b_patch16_dinov3.lvd1689m` disimpan dalam fp32 (26,9 GB). Notebook utama memakai bobot "
    "fp16 yang dibagi ke dua T4, jadi konversi sekali di sini menghemat unduhan dan ruang disk di setiap run. "
    "Notebook ini hanya mengubah dtype tensor dan tidak menjalankan model.\n\n"
    "Cara pakai:\n\n"
    "1. Jalankan dengan Internet ON (CPU sudah cukup).\n"
    "2. Setelah selesai, buat dataset Kaggle **privat** dari folder `/kaggle/working/dinov3_7b_fp16`, sertakan teks "
    "DINOv3 License.\n"
    "3. Lampirkan dataset itu ke notebook utama. Notebook utama mencari pola `dinov3_7b_fp16*.safetensors`."
))

# ---------------------------------------------------------------------------
# 2. Inisialisasi
# ---------------------------------------------------------------------------
cells.append(section("Inisialisasi", "2"))
cells.append(md(
    "## Environment Setup\n\n"
    "Unduhan fp32 butuh sekitar 27 GB di `/tmp`, dan keluaran fp16 sekitar 13,4 GB di `/kaggle/working` (batas 20 GB). "
    "Sel berikut menampilkan ruang disk yang tersedia."
))
cells.append(code("!df -h /tmp /kaggle/working"))
cells.append(md("Sel berikut memastikan `safetensors` dan `huggingface_hub` terpasang."))
cells.append(code("%pip install -q safetensors huggingface_hub"))
cells.append(md("## Import Libraries"))
cells.append(code(
    "import hashlib\n"
    "import json\n"
    "import shutil\n"
    "from pathlib import Path\n"
    "\n"
    "import torch\n"
    "from huggingface_hub import hf_hub_download\n"
    "from safetensors import safe_open\n"
    "from safetensors.torch import save_file"
))
cells.append(md(
    "## Settings\n\n"
    "Revisi bobot dikunci supaya hasil konversi selalu sama dengan revisi yang dipakai notebook utama."
))
cells.append(code(
    "class Settings:\n"
    "    REPO = 'timm/vit_7b_patch16_dinov3.lvd1689m'\n"
    "    REV = 'b40f6c827cb02b13c7a23a76cf598489bb0f94e5'\n"
    "    HF_CACHE = Path('/tmp/hf')\n"
    "    OUT_DIR = Path('/kaggle/working/dinov3_7b_fp16')\n"
    "    SHARD_BYTES = 4 * 2**30\n"
    "    FP16_MAX = 65504.0\n"
    "    DISK_MIN_GIB = 30\n"
    "\n"
    "\n"
    "CFG = Settings()\n"
    "CFG.HF_CACHE.mkdir(parents=True, exist_ok=True)\n"
    "CFG.OUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    "bebas = shutil.disk_usage(CFG.HF_CACHE).free / 2**30\n"
    "assert bebas >= CFG.DISK_MIN_GIB, f'/tmp hanya {bebas:.0f} GiB, butuh minimal {CFG.DISK_MIN_GIB} GiB untuk bobot fp32'\n"
    "print(f'/tmp bebas {bebas:.0f} GiB')"
))

# ---------------------------------------------------------------------------
# 3. Konversi
# ---------------------------------------------------------------------------
cells.append(section("Konversi", "3"))
cells.append(md(
    "Tensor dibaca satu per satu dengan `safe_open`, jadi RAM tidak pernah memuat seluruh bobot. Tensor float "
    "dikonversi ke fp16 dan dicek finite, lalu ditulis per shard 4 GiB. Buffer RoPE tidak ada di checkpoint "
    "(non-persisten), jadi tidak ikut terkonversi."
))
cells.append(code(
    "src = hf_hub_download(CFG.REPO, 'model.safetensors', revision=CFG.REV, cache_dir=CFG.HF_CACHE)\n"
    "manifest = {'repo': CFG.REPO, 'revision': CFG.REV, 'shard': {}}\n"
    "shard, ukuran, n, maks = {}, 0, 0, 0.0\n"
    "\n"
    "\n"
    "def simpan(shard, n):\n"
    "    f = CFG.OUT_DIR / f'dinov3_7b_fp16-{n:05d}.safetensors'\n"
    "    save_file(shard, str(f), metadata={'format': 'pt', 'sumber': f'{CFG.REPO}@{CFG.REV}'})\n"
    "    manifest['shard'][f.name] = hashlib.sha256(f.read_bytes()).hexdigest()\n"
    "    print(f.name, len(shard), 'tensor', flush=True)\n"
    "\n"
    "\n"
    "with safe_open(src, 'pt', device='cpu') as sf:\n"
    "    for k in sf.keys():\n"
    "        t = sf.get_tensor(k)\n"
    "        if t.is_floating_point():\n"
    "            maks = max(maks, t.abs().max().item())\n"
    "            t = t.to(torch.float16)\n"
    "            assert torch.isfinite(t).all(), f'{k} overflow saat konversi ke fp16'\n"
    "        shard[k] = t.contiguous()\n"
    "        ukuran += t.numel() * t.element_size()\n"
    "        if ukuran >= CFG.SHARD_BYTES:\n"
    "            n += 1\n"
    "            simpan(shard, n)\n"
    "            shard, ukuran = {}, 0\n"
    "if shard:\n"
    "    n += 1\n"
    "    simpan(shard, n)\n"
    "manifest['maks_abs_bobot_fp32'] = maks\n"
    "(CFG.OUT_DIR / 'manifest_dinov3_7b_fp16.json').write_text(json.dumps(manifest, indent=1))\n"
    "print('selesai, |bobot| maks', maks)"
))

# ---------------------------------------------------------------------------
# 4. Verifikasi
# ---------------------------------------------------------------------------
cells.append(section("Verifikasi", "4"))
cells.append(md(
    "Hasil dicek terhadap sumber: jumlah tensor dan nama kunci harus sama persis, nilai bobot terbesar harus di "
    "bawah batas fp16, dan total ukuran sekitar 13,4 GB. Berkas fp32 di `/tmp` dihapus setelah lolos."
))
cells.append(code(
    "kunci_fp16 = set()\n"
    "for f in sorted(CFG.OUT_DIR.glob('dinov3_7b_fp16-*.safetensors')):\n"
    "    with safe_open(str(f), 'pt', device='cpu') as sf:\n"
    "        kunci_fp16 |= set(sf.keys())\n"
    "with safe_open(src, 'pt', device='cpu') as sf:\n"
    "    kunci_src = set(sf.keys())\n"
    "assert kunci_fp16 == kunci_src, (len(kunci_fp16), len(kunci_src))\n"
    "assert maks < CFG.FP16_MAX\n"
    "total = sum(f.stat().st_size for f in CFG.OUT_DIR.glob('*.safetensors')) / 1e9\n"
    "print(f'{len(kunci_fp16)} tensor cocok | total {total:.2f} GB | |bobot| maks {maks:.2f}')\n"
    "shutil.rmtree(CFG.HF_CACHE, ignore_errors=True)"
))
cells.append(md(
    "#### Insights\n\n"
    "> Isi setelah run: jumlah shard, total ukuran, dan nilai bobot terbesar. Bila assert overflow gagal, notebook "
    "utama tetap bisa memakai jalur unduhan fp32 langsung."
))

# ---------------------------------------------------------------------------
# Notebook writer
# ---------------------------------------------------------------------------
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
        "kaggle": {"accelerator": "none", "isInternetEnabled": True, "isGpuEnabled": False},
    },
    "cells": cells,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(cells)} cells -> {OUT}")
