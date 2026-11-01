"""Cek kelengkapan berkas: ID, ekstensi, kecocokan sample submission, ukuran citra, sumber label.

Hanya membaca metadata dan header citra. Tidak menampilkan citra ke siapa pun.

    python temp/eda_002_cek_berkas/cek_berkas.py
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


def ringkas_folder(folder):
    files = sorted(p for p in folder.iterdir() if p.is_file())
    subdir = sorted(p.name for p in folder.iterdir() if p.is_dir())
    stems = [p.stem for p in files]
    angka = sorted(int(s) for s in stems if s.isdigit())
    hilang = sorted(set(range(1, len(files) + 1)) - set(angka))
    modes, sizes, gagal = Counter(), [], []
    for p in files:
        try:
            with Image.open(p) as im:
                modes[f"{im.format}/{im.mode}"] += 1
                sizes.append(im.size)
                im.verify()
        except Exception as e:
            gagal.append(f"{p.name}: {e}")
    lebar = pd.Series([s[0] for s in sizes])
    tinggi = pd.Series([s[1] for s in sizes])
    return {
        "jumlah": len(files),
        "subfolder": subdir,
        "ekstensi": dict(Counter(p.suffix.lower() for p in files)),
        "panjang_nama": dict(Counter(len(s) for s in stems)),
        "id_min": angka[0] if angka else None,
        "id_max": angka[-1] if angka else None,
        "id_hilang": hilang[:20],
        "stem_duplikat": [k for k, v in Counter(stems).items() if v > 1][:20],
        "format_mode": dict(modes),
        "gagal_buka": gagal[:20],
        "lebar": lebar.describe().round(1).to_dict(),
        "tinggi": tinggi.describe().round(1).to_dict(),
    }, stems


hasil = {}
hasil["train"], train_stems = ringkas_folder(DATA / "train")
hasil["test"], test_stems = ringkas_folder(DATA / "test")

ss = pd.read_csv(ROOT / "data" / "SampleSubmission_DSC.csv", dtype=str)
hasil["sample_submission"] = {
    "baris": len(ss),
    "kolom": list(ss.columns),
    "panjang_id": dict(Counter(ss["image_id"].str.len())),
    "id_cocok_test": set(ss["image_id"]) == set(test_stems),
    "distribusi_kelas": ss["waste_category"].value_counts().to_dict(),
}

label_like = [str(p.relative_to(ROOT)) for p in (ROOT / "data").rglob("*")
              if p.is_file() and p.suffix.lower() not in {".jpg", ".jpeg", ".png"}]
hasil["berkas_non_citra_di_data"] = label_like

print(json.dumps(hasil, indent=2, ensure_ascii=False))
(OUT / "cek_berkas.json").write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")
