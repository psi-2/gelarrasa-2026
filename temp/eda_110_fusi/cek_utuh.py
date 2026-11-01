"""Cek keutuhan keluaran EDA setelah komputer mati mendadak. Hanya bentuk array dan jumlah baris.

    python temp/eda_110_fusi/cek_utuh.py
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "temp"
for d in sorted(ROOT.glob("eda_1*")):
    print(d.name, "notes.md" if (d / "notes.md").exists() else "TANPA notes.md")
    o = d / "output"
    if not o.exists():
        print("   tanpa output")
        continue
    for f in sorted(o.iterdir()):
        t = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(f)))
        try:
            if f.suffix == ".npy":
                info = str(np.load(f, mmap_mode="r").shape)
            elif f.suffix == ".csv":
                info = f"{len(pd.read_csv(f))} baris"
            else:
                info = f"{f.stat().st_size} byte"
        except Exception as e:
            info = "RUSAK " + repr(e)[:80]
        print(f"   {f.name:28s} {t} {info}")
