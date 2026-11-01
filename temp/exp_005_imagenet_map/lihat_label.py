"""Tulis daftar kelas ConvNeXtV2 ke output/label_model.txt. Tidak menyentuh data kompetisi."""
import json
from pathlib import Path

SRC = Path(r"D:\cache_moved\convnextv2_huge_clean")
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

cfg = json.loads((SRC / "config.json").read_text(encoding="utf-8"))
pre = json.loads((SRC / "preprocessor_config.json").read_text(encoding="utf-8"))
id2 = cfg["id2label"]
print("arsitektur:", cfg.get("architectures"), "| depth:", cfg.get("depths"), "| hidden:", cfg.get("hidden_sizes"))
print("image_size cfg:", cfg.get("image_size"), "| jumlah kelas:", len(id2))
print("preprocessor:", pre)
baris = [f"{int(k)}\t{v.split(',')[0]}" for k, v in sorted(id2.items(), key=lambda x: int(x[0]))]
(OUT / "label_model.txt").write_text("\n".join(baris), encoding="utf-8")
print("\n".join(b.replace("\t", ":") for b in baris))
