"""Vektor area per kelas dari EoMT DINOv3 (COCO-panoptic base dan ADE20K large).

Non-VLM: model hanya menerima citra, tanpa teks, tidak dilatih pada dataset sampah.
Keluaran numerik saja: output/{coco,ade}_{split}.npy (N x C x 2), kanal 0 seluruh citra, kanal 1 area tengah 50%.

    set HF_HUB_OFFLINE=1
    python temp/eda_106_eomt/eomt_area.py --model coco
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForUniversalSegmentation

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
DATA = ROOT / "data"
VRAM_GIB = 5.0
REPO = {"coco": ("tue-mps/eomt-dinov3-coco-panoptic-base-640", 640),
        "ade": ("tue-mps/eomt-dinov3-ade-semantic-large-512", 512)}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Citra(Dataset):
    def __init__(self, files, size):
        self.files, self.size = files, size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as im:
            a = im.convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        x = (np.asarray(a, dtype=np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(x).permute(2, 0, 1)


@torch.no_grad()
def jalan(model, files, size, bs):
    tengah = None
    keluar = []
    for px in DataLoader(Citra(files, size), batch_size=bs, num_workers=2, shuffle=False):
        px = px.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(pixel_values=px)
        cl = o.class_queries_logits.float().softmax(-1)[..., :-1]
        m = o.masks_queries_logits.float().sigmoid()
        sem = torch.einsum("bqc,bqhw->bchw", cl, m)
        sem = sem / sem.sum(1, keepdim=True).clamp_min(1e-6)
        if tengah is None:
            h, w = sem.shape[-2:]
            tengah = (h // 4, 3 * h // 4, w // 4, 3 * w // 4)
        a = sem.mean((2, 3))
        t = sem[:, :, tengah[0]:tengah[1], tengah[2]:tengah[3]].mean((2, 3))
        keluar.append(torch.stack([a, t], 2).cpu())
    return torch.cat(keluar).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(REPO), required=True)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(VRAM_GIB / total, 0)
    repo, size = REPO[a.model]
    model = AutoModelForUniversalSegmentation.from_pretrained(repo, local_files_only=True, dtype=torch.bfloat16).cuda().eval()
    id2 = {int(k): v for k, v in model.config.id2label.items()}
    (OUT / f"{a.model}_kelas.json").write_text(json.dumps([id2[i] for i in range(len(id2))]), encoding="utf-8")
    t0 = time.time()
    for split in ("test", "train"):
        files = sorted((DATA / split).iterdir())[:a.limit]
        x = jalan(model, files, size, a.bs)
        if a.limit is None:
            np.save(OUT / f"{a.model}_{split}.npy", x.astype(np.float32))
        print(split, x.shape, round(time.time() - t0), "detik, puncak", round(torch.cuda.max_memory_reserved() / 2**30, 2), "GiB", flush=True)


if __name__ == "__main__":
    main()
