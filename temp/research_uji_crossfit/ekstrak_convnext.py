"""Ekstrak embedding ConvNeXt-L DINOv3 di CPU, 4 thread, dari bobot cache lokal, tanpa unduhan.

Preprocessing sama dengan tf_eval pada temp/exp_010_dua_backbone/train_dua.py: resize 384 bicubic, normalisasi ImageNet.
Skrip hanya menyimpan vektor. Tidak ada citra yang ditampilkan atau disimpan.
Potongan disimpan tiap 64 citra agar bisa dilanjutkan bila proses terhenti. Test hanya diekstrak, tidak dipakai fit.

    python temp/research_uji_crossfit/ekstrak_convnext.py [--limit 8]
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_HUB_OFFLINE"] = "1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms as T  # noqa: E402

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
POT = OUT / "potongan_cnn"
POT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"
BOBOT = Path(r"D:\cache_moved\timm--convnext_large.dinov3_lvd1689m\model.safetensors")
UKURAN, CHUNK, BS = 384, 64, 4
tf = T.Compose([T.Resize((UKURAN, UKURAN), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def muat_tensor(p):
    with Image.open(p) as im:
        return tf(im.convert("RGB"))


@torch.no_grad()
def jalan(model, split, limit):
    files = sorted((DATA / split).iterdir())
    if limit:
        files = files[:limit]
    ids = np.array([p.stem for p in files])
    t0 = time.time()
    for awal in range(0, len(files), CHUNK):
        f_out = POT / f"{split}_{awal:05d}.npy"
        if f_out.exists() and not limit:
            continue
        hasil = []
        bagian = files[awal:awal + CHUNK]
        for i in range(0, len(bagian), BS):
            x = torch.stack([muat_tensor(p) for p in bagian[i:i + BS]])
            hasil.append(model(x).float().numpy())
        np.save(f_out, np.concatenate(hasil))
        print(f"{split} {awal + len(bagian)}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    if limit:
        return
    emb = np.concatenate([np.load(POT / f"{split}_{a:05d}.npy") for a in range(0, len(files), CHUNK)])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(OUT / f"emb_cnn_{split}.npy", emb.astype(np.float32))
    np.save(OUT / f"ids_cnn_{split}.npy", ids)
    print(split, "selesai", emb.shape, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    assert BOBOT.exists()
    model = timm.create_model("convnext_large.dinov3_lvd1689m", pretrained=True, num_classes=0,
                              pretrained_cfg_overlay={"file": str(BOBOT)}).eval()
    jalan(model, "train", args.limit)
    jalan(model, "test", args.limit)


if __name__ == "__main__":
    main()
