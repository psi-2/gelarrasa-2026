"""Pembungkus satu job GPU untuk infer22k.py lalu infer_dms.py tanpa mengubah skrip aslinya.

Batas VRAM 5 GiB (skrip asli 8 GiB), DataLoader 3 worker, thread math 1, agar total maksimal 4 thread CPU.
Keluaran tetap di temp/exp_008_in22k_map/output/.

    python temp/eda_112_label_v2/jalankan_gpu.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "temp" / "exp_008_in22k_map"))
sys.argv = [sys.argv[0]]

import torch
from torch.utils.data import DataLoader as DL

import infer22k
import infer_dms


def loader(*a, **k):
    k["num_workers"] = 3
    return DL(*a, **k)


if __name__ == "__main__":
    torch.set_num_threads(1)
    infer22k.VRAM_MAKS_GIB = 5.0
    infer22k.DataLoader = loader
    infer_dms.VRAM_MAKS_GIB = 5.0
    infer22k.main()
    torch.cuda.empty_cache()
    infer_dms.main()
    print("GPU_SELESAI", flush=True)
