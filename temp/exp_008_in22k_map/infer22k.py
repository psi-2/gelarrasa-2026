"""Inferensi classifier ImageNet-22k lokal untuk train dan test, simpan softmax penuh.

Hanya model citra, tanpa teks. Batas VRAM 8 GiB karena GPU dibagi dengan sub agent EDA.

    python temp/exp_008_in22k_map/infer22k.py [--limit 32]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.utils.seed_utils import seed_everything  # noqa: E402

import numpy as np  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import transforms as T  # noqa: E402

DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "output"
HUB = Path(r"D:\cache_moved\huggingface\hub")
MODEL = {"convnext_l": ("timm/convnext_large.fb_in22k", "convnext_large.fb_in22k", 288),
         "swinv2_l": ("timm/swinv2_large_window12_192.ms_in22k", "swinv2_large_window12_192.ms_in22k", 192),
         "beitv2_l": ("timm/beitv2_large_patch16_224.in1k_ft_in22k", "beitv2_large_patch16_224.in1k_ft_in22k", 224)}
VRAM_MAKS_GIB = 8.0


def snapshot(repo):
    for s in sorted((HUB / ("models--" + repo.replace("/", "--")) / "snapshots").glob("*")):
        if (s / "model.safetensors").exists():
            return s / "model.safetensors"
    raise SystemExit(f"bobot tidak ada: {repo}")


class Citra(Dataset):
    def __init__(self, files, tf):
        self.files, self.tf = files, tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as im:
            return self.tf(im.convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bs", type=int, default=12)
    args = ap.parse_args()
    seed_everything(42, warn_only=True)
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(VRAM_MAKS_GIB / total_gib, 0)
    OUT.mkdir(exist_ok=True)
    laporan = {}
    for nama, (repo, arch, ukuran) in MODEL.items():
        t0 = time.time()
        model = timm.create_model(arch, pretrained=True, pretrained_cfg_overlay={"file": str(snapshot(repo))}).cuda().eval()
        cfg = timm.data.resolve_model_data_config(model)
        tf = T.Compose([T.Resize((ukuran, ukuran), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(),
                        T.Normalize(cfg["mean"], cfg["std"])])
        print(nama, "kelas", model.num_classes, "ukuran", ukuran, "mean", cfg["mean"], flush=True)
        for split in ("test", "train"):
            files = sorted((DATA / split).iterdir())[:args.limit]
            hasil = []
            for x in DataLoader(Citra(files, tf), batch_size=args.bs, num_workers=4):
                x = x.cuda(non_blocking=True)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    p = (torch.softmax(model(x).float(), 1) + torch.softmax(model(x.flip(3)).float(), 1)) / 2
                hasil.append(p.cpu())
            p = torch.cat(hasil).numpy().astype(np.float32)
            np.save(OUT / f"p22k_{split}_{nama}.npy", p)
            print(nama, split, p.shape, "maks rata", round(float(p.max(1).mean()), 4), flush=True)
        laporan[nama] = {"detik": round(time.time() - t0, 1), "vram_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print(json.dumps(laporan))
    (OUT / "infer22k.json").write_text(json.dumps(laporan, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
