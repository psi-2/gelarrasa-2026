"""Daftar checkpoint visi lengkap di cache beserta ukuran dan kosakatanya. Tidak memuat citra.

Hanya model non-VLM: nama yang memuat clip, siglip, sam3, radio, steervit dilewati.

    python temp/eda_105_in21k/cek_bobot.py
"""
import json
from pathlib import Path

hub = Path(r"D:\cache_moved\huggingface\hub")
LARANG = ("clip", "siglip", "siglino", "sam3", "radio", "steervit", "qwen", "minicpm", "waste", "blip", "llava")
pil = []
for d in sorted(hub.iterdir()):
    n = d.name.lower()
    if not n.startswith("models--") or any(x in n for x in LARANG):
        continue
    if not any(x in n for x in ("timm", "eomt", "dinov2", "convnext", "swin", "segformer", "mask2former", "upernet", "detr", "vit", "wd-", "deit", "beit", "resnet", "efficientnet")):
        continue
    ber = [p for p in d.rglob("*") if p.is_file() and p.suffix in (".safetensors", ".bin", ".pth", ".pt") and p.stat().st_size > 5e6]
    ukuran = sum(p.stat().st_size for p in ber) / 1e6
    cfg = None
    for p in d.rglob("config.json"):
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
            cfg = (c.get("num_classes"), len(c.get("id2label", {})))
        except Exception:
            pass
        break
    print(f"{d.name:70s} {ukuran:9.1f} MB  cfg={cfg}")
for p in hub.glob("models--tue-mps--eomt*/snapshots/*/config.json"):
    c = json.loads(p.read_text(encoding="utf-8"))
    v = c.get("id2label", {})
    print(p.parts[-4], len(v), list(v.values()))
