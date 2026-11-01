"""Inventaris classifier citra di cache lokal: bobot lengkap, jumlah kelas, nama kelas material.

Tidak menyentuh data kompetisi. Model berbasis CLIP atau teks dilewati (NO VLM).

    python temp/exp_008_in22k_map/inventaris.py
"""
import json
from pathlib import Path

HUBS = [Path(r"D:\cache_moved\huggingface\hub"), Path(r"D:\cache_moved\hub")]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
LARANG = ("clip", "siglip", "blip", "llava", "qwen", "text", "sentence", "bert", "gemma")
KATA = ["bottle", "can", "cardboard", "carton", "paper", "newspaper", "plastic", "glass", "jar", "tin", "aluminum",
        "compost", "garbage", "trash", "food", "cloth", "fabric", "circuit", "battery", "crumpled"]

baris = []
for hub in HUBS:
    for d in sorted(hub.glob("models--*")):
        nama = d.name.replace("models--", "").replace("--", "/")
        if any(k in nama.lower() for k in LARANG):
            continue
        snaps = sorted((d / "snapshots").glob("*"))
        if not snaps:
            continue
        s = snaps[-1]
        bobot = [f for f in list(s.glob("*.safetensors")) + list(s.glob("*.bin")) + list(s.glob("*.pth"))]
        ukuran = sum(f.stat().st_size for f in bobot if f.exists())
        if ukuran < 1e7:
            continue
        cfg = {}
        if (s / "config.json").exists():
            cfg = json.loads((s / "config.json").read_text(encoding="utf-8"))
        n = cfg.get("num_classes") or len(cfg.get("id2label", {})) or None
        tag = cfg.get("pretrained_cfg", {}).get("tag") or ""
        lab = cfg.get("label_names") or list(cfg.get("id2label", {}).values())
        baris.append({"model": nama, "gb": round(ukuran / 1e9, 2), "kelas": n, "tag": tag, "snapshot": str(s),
                      "n_nama": len(lab), "contoh": lab[:3]})
        if len(lab) > 5000:
            hit = {k: sum(k in x.lower() for x in lab) for k in KATA}
            print("\n==", nama, "nama kelas:", len(lab), hit)
            for k in ("cardboard", "tin can", "beer can", "soda", "newspaper", "plastic bag", "plastic", "jar", "compost",
                      "garbage", "circuit", "crumpled"):
                print(" ", k, ":", [x[:48] for x in lab if k in x.lower()][:8])

baris.sort(key=lambda r: -(r["kelas"] or 0))
for r in baris:
    print(f"{r['model']:<62} {r['gb']:>6} GB kelas={r['kelas']} nama={r['n_nama']} tag={r['tag']}")
(OUT / "inventaris.json").write_text(json.dumps(baris, indent=2, ensure_ascii=False), encoding="utf-8")
