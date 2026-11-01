"""Periksa nama resmi saja: subset ImageNet tiap model, rentang synset, dan taksonomi DMS46.

Tidak memuat bobot, tidak menyentuh citra, tidak membaca keluaran inferensi.

    python temp/eda_112_label_v2/lihat_nama.py
"""
import json
import os
import re
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import timm
from timm.data import ImageNetInfo, infer_imagenet_subset

ROOT = Path(__file__).resolve().parents[2]
for arch in ("convnext_large.fb_in22k", "swinv2_large_window12_192.ms_in22k", "beitv2_large_patch16_224.in1k_ft_in22k"):
    m = timm.create_model(arch, pretrained=False)
    print(arch, "num_classes", m.num_classes, "subset", infer_imagenet_subset(m))

info = ImageNetInfo("imagenet-22k")
ids = [info.index_to_label_name(i) for i in range(info.num_classes())]
print("urut menaik:", ids == sorted(ids), "n", len(ids))
# rentang synset per juta offset
for awal in range(0, 16):
    pre = f"n{awal:02d}"
    idx = [i for i, s in enumerate(ids) if s.startswith(pre)]
    if not idx:
        continue
    contoh = [info.index_to_description(idx[int(len(idx) * q)]).split(",")[0][:22] for q in (0, .25, .5, .75, .999)]
    print(pre, len(idx), contoh)

for tgt in ("food, nutrient", "foodstuff", "beverage, drink", "fruit:", "artifact, artefact", "clothing, article of clothing", "fabric, cloth", "electronic equipment"):
    print(tgt, [(ids[i], i) for i in range(len(ids)) if info.index_to_description(i, detailed=True).startswith(tgt)][:3])

t = json.loads(Path(r"D:\cache_moved\dms46\taxonomy.json").read_text(encoding="utf-8"))
print("kunci:", list(t))
DMS46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33, 34, 35,
         36, 37, 38, 39, 41, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 56]
print([(i, t["names"][k]) for i, k in enumerate(DMS46)])
