"""Unduh bobot classifier ImageNet-22k ke cache lokal (safetensors dan config saja).

    python temp/exp_008_in22k_map/unduh.py
"""
from huggingface_hub import snapshot_download

MODEL = ["timm/convnext_large.fb_in22k", "timm/swinv2_large_window12_192.ms_in22k",
         "timm/beitv2_large_patch16_224.in1k_ft_in22k"]
for m in MODEL:
    p = snapshot_download(m, allow_patterns=["model.safetensors", "config.json"], cache_dir=r"D:\cache_moved\huggingface\hub")
    print(m, p, flush=True)
