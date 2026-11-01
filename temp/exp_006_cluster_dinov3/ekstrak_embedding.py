"""Ekstrak embedding DINOv3 ViT-L untuk train dan test, tanpa label dan tanpa teks.

    python temp/exp_006_cluster_dinov3/ekstrak_embedding.py
"""
import numpy as np
import torch

from train_dino import DATA, HERE, SEED, VRAM_MAKS_GIB, buat_model, ekstrak, seed_everything


def main():
    seed_everything(SEED)
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(VRAM_MAKS_GIB / total_gib, 0)
    out = HERE / "output"
    model = buat_model(0).cuda()
    for split in ("train", "test"):
        files = sorted((DATA / split).iterdir())
        emb = ekstrak(model, files)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        np.save(out / f"emb_{split}.npy", emb.astype(np.float32))
        np.save(out / f"ids_{split}.npy", np.array([p.stem for p in files]))
        print(split, emb.shape, flush=True)
    print(f"vram puncak {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
