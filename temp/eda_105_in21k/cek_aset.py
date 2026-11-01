"""Cek berkas synset timm dan isi cache checkpoint ImageNet-21k. Tidak memuat citra.

    python temp/eda_105_in21k/cek_aset.py
"""
import json
import os
from pathlib import Path

import timm

base = Path(timm.__file__).parent / "data" / "_info"
for f in sorted(base.iterdir()):
    print(f.name, f.stat().st_size)
try:
    from nltk.corpus import wordnet as wn
    print("wordnet synset uji:", wn.synsets("bottle")[:2])
except Exception as e:
    print("nltk wordnet gagal:", repr(e)[:120])

hub = Path(r"D:\cache_moved\huggingface\hub")
for m in ("models--timm--convnext_large.fb_in22k", "models--microsoft--swinv2-large-patch4-window12-192-22k",
          "models--facebook--convnext-base-224-22k", "models--timm--vit_small_patch16_224.augreg_in21k"):
    d = hub / m
    print(m, d.exists())
    if d.exists():
        for p in d.rglob("*"):
            if p.is_file() and p.stat().st_size > 1000:
                print("   ", p.relative_to(d), round(p.stat().st_size / 1e6, 1), "MB")
        for p in d.rglob("config.json"):
            c = json.loads(p.read_text())
            print("    num_classes", c.get("num_classes"), "label_names?", "label_names" in c, "input", c.get("pretrained_cfg", {}).get("input_size"),
                  "id2label?", len(c.get("id2label", {})))
print("HF env", {k: v for k, v in os.environ.items() if k.startswith(("HF_", "TORCH_"))})
