"""Porsi piksel material per citra dari DMS46 (Apple, ECCV 2022). Model citra saja, tanpa teks.

Preprocessing mengikuti inference.py resmi. Hasil hanya angka histogram, tanpa peta atau citra.

    python temp/exp_008_in22k_map/infer_dms.py [--limit 32]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.utils.seed_utils import seed_everything  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms as TTR  # noqa: E402
from PIL import Image  # noqa: E402

DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "output"
DMS = Path(r"D:\cache_moved\dms46")
VRAM_MAKS_GIB = 8.0
DMS46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33, 34, 35,
         36, 37, 38, 39, 41, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 56]
MEAN = [x * 255 for x in (0.485, 0.456, 0.406)]
STD = [x * 255 for x in (0.229, 0.224, 0.225)]


def siapkan(path):
    with Image.open(path) as im:
        img = im.convert("RGB")
    w, h = img.size
    s = min(512 / h, 512 / w)
    img = img.resize((math.ceil(s * w), math.ceil(s * h)), Image.LANCZOS)
    x = torch.from_numpy(np.array(img).transpose((2, 0, 1))).float()
    return TTR.Normalize(MEAN, STD)(x).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    seed_everything(42, warn_only=True)
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(VRAM_MAKS_GIB / total_gib, 0)
    t = json.loads((DMS / "taxonomy.json").read_text(encoding="utf-8"))
    print("kunci taxonomy:", list(t)[:8])
    nama = [t["names"][i] for i in DMS46]
    (OUT / "dms_kelas.json").write_text(json.dumps(nama, indent=1), encoding="utf-8")
    print("kelas:", nama)
    model = torch.jit.load(str(DMS / "extracted" / "DMS46_v1.pt")).cuda().eval()
    t0 = time.time()
    for split in ("test", "train"):
        files = sorted((DATA / split).iterdir())[:args.limit]
        semua, tengah = np.zeros((len(files), 46), np.float32), np.zeros((len(files), 46), np.float32)
        for i, p in enumerate(files):
            with torch.no_grad():
                m = model(siapkan(p).cuda())[0].data.cpu()[0, 0].numpy().astype(int)
            h, w = m.shape
            semua[i] = np.bincount(m.ravel(), minlength=46)[:46] / m.size
            c = m[h // 4:h - h // 4, w // 4:w - w // 4]
            tengah[i] = np.bincount(c.ravel(), minlength=46)[:46] / c.size
            if i % 500 == 0:
                print(split, i, round(time.time() - t0), flush=True)
        np.save(OUT / f"dms_{split}.npy", semua)
        np.save(OUT / f"dms_tengah_{split}.npy", tengah)
        print(split, "rata porsi teratas:", sorted(zip(semua.mean(0).round(3).tolist(), nama), reverse=True)[:8], flush=True)
    print("vram puncak", round(torch.cuda.max_memory_reserved() / 2**30, 2), "detik", round(time.time() - t0))


if __name__ == "__main__":
    main()
