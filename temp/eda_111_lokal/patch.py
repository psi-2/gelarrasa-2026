"""Hipotesis D: patch token DINOv3 ViT-L 384, diproyeksikan PCA 64 dan disimpan float16.

PCA di-fit dari 200 citra train. Satu job GPU, batas 5 GiB. Citra tidak ditampilkan.

Keluaran: output/tokens.npy (2400 x 576 x 64, float16, memmap), output/pca.npz

    set HF_HUB_OFFLINE=1
    python temp/eda_111_lokal/patch.py
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from scripts.utils.model_loader import resolve_model

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
DATA = ROOT / "data"
UK = 384
MEAN = np.array([.485, .456, .406], dtype=np.float32)
STD = np.array([.229, .224, .225], dtype=np.float32)


class Citra(Dataset):
    def __init__(self, f):
        self.f = f

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        with Image.open(self.f[i]) as im:
            a = im.convert("RGB").resize((UK, UK), Image.BICUBIC)
        return torch.from_numpy((np.asarray(a, dtype=np.float32) / 255 - MEAN) / STD).permute(2, 0, 1)


def main():
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(5.0 / total, 0)
    src, _ = resolve_model("timm/vit_large_patch16_dinov3.lvd1689m", verbose=False)
    ov = {"file": str(Path(src) / "model.safetensors")} if Path(src).is_dir() else None
    m = timm.create_model("vit_large_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, pretrained_cfg_overlay=ov, img_size=UK).cuda().eval()
    pre = m.num_prefix_tokens
    files = sorted((DATA / "train").iterdir()) + sorted((DATA / "test").iterdir())
    rng = np.random.default_rng(42)
    sampel = rng.choice(1920, 200, replace=False)

    @torch.no_grad()
    def tok(px):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = m.forward_features(px)[:, pre:]
        t = t.float()
        return torch.nn.functional.normalize(t, dim=-1)

    kumpul = []
    for px in DataLoader(Citra([files[i] for i in sampel]), batch_size=8, num_workers=2):
        kumpul.append(tok(px.cuda()).flatten(0, 1)[::4].cpu())
    X = torch.cat(kumpul).numpy()
    mu = X.mean(0)
    _, s, vt = np.linalg.svd(X[:60000] - mu, full_matrices=False)
    P = vt[:64].astype(np.float32)
    print("PCA 64 variansi", round(float((s[:64] ** 2).sum() / (s ** 2).sum()), 3), flush=True)
    np.savez(OUT / "pca.npz", mu=mu, P=P)
    muP = torch.from_numpy(mu).cuda()
    PP = torch.from_numpy(P).cuda()
    mm = np.lib.format.open_memmap(OUT / "tokens.npy", mode="w+", dtype=np.float16, shape=(len(files), 576, 64))
    i, t0 = 0, time.time()
    for px in DataLoader(Citra(files), batch_size=8, num_workers=2):
        z = ((tok(px.cuda()) - muP) @ PP.T).cpu().numpy()
        mm[i:i + len(z)] = z.astype(np.float16)
        i += len(z)
        if i % 400 == 0:
            mm.flush()
            print(i, round(time.time() - t0), "detik, puncak", round(torch.cuda.max_memory_reserved() / 2**30, 2), "GiB", flush=True)
    mm.flush()
    print("selesai", i, round(time.time() - t0))


if __name__ == "__main__":
    main()
