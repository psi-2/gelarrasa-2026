"""Hipotesis A: sidik jari berkas per citra (byte header, kuantisasi JPEG, EXIF, ukuran).

Tidak ada citra yang ditampilkan. Keluaran hanya tabel numerik.

    python temp/eda_101_metadata/scan_meta.py
"""
import hashlib
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ExifTags

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
DATA = ROOT / "data"


def parse_jpeg(b):
    """Baca marker JPEG sampai SOS. Kembalikan ringkasan struktur."""
    r = {"markers": [], "app_list": [], "jfif": "", "dqt": [], "comment": "", "progressive": 0,
         "sof": "", "subsamp": "", "n_dqt": 0, "n_dht": 0, "dri": 0, "adobe": 0, "icc": 0, "exif_len": 0,
         "app_bytes": 0}
    if b[:2] != b"\xff\xd8":
        return r
    i = 2
    n = len(b)
    while i + 4 <= n:
        if b[i] != 0xFF:
            i += 1
            continue
        m = b[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        L = struct.unpack(">H", b[i + 2:i + 4])[0]
        seg = b[i + 4:i + 2 + L]
        r["markers"].append(f"{m:02X}")
        if 0xE0 <= m <= 0xEF:
            r["app_bytes"] += L
            tag = seg[:12].split(b"\x00")[0].decode("latin1", "replace")
            r["app_list"].append(f"E{m - 0xE0:X}:{tag}")
            if m == 0xE0 and seg[:5] == b"JFIF\x00":
                r["jfif"] = f"{seg[5]}.{seg[6]:02d}|d{seg[7]}|{struct.unpack('>H', seg[8:10])[0]}x{struct.unpack('>H', seg[10:12])[0]}"
            if m == 0xE1 and seg[:4] == b"Exif":
                r["exif_len"] = L
            if m == 0xE2 and seg[:11] == b"ICC_PROFILE":
                r["icc"] = 1
            if m == 0xEE and seg[:5] == b"Adobe":
                r["adobe"] = 1
        elif m == 0xFE:
            r["comment"] = seg[:200].decode("latin1", "replace")
        elif m == 0xDB:
            r["n_dqt"] += 1
            j = 0
            while j < len(seg):
                pq, tq = seg[j] >> 4, seg[j] & 15
                j += 1
                sz = 64 * (1 + pq)
                t = np.frombuffer(seg[j:j + sz], dtype=">u2" if pq else "u1").astype(int)
                r["dqt"].append((tq, t))
                j += sz
        elif m == 0xC4:
            r["n_dht"] += 1
        elif m == 0xDD:
            r["dri"] = struct.unpack(">H", seg[:2])[0]
        elif m in (0xC0, 0xC1, 0xC2):
            r["progressive"] = int(m == 0xC2)
            r["sof"] = f"C{m - 0xC0}"
            nc = seg[5]
            comp = [(seg[6 + 3 * k + 1] >> 4, seg[6 + 3 * k + 1] & 15) for k in range(nc)]
            r["subsamp"] = "/".join(f"{h}{v}" for h, v in comp)
        elif m == 0xDA:
            break
        i += 2 + L
    return r


def kualitas_ijg(t):
    """Perkiraan quality IJG dari tabel luma, cocokkan ke tabel standar."""
    std = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])
    zig = np.array([0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11, 18, 24,
                    31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34, 37, 47, 50,
                    56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63])
    t2 = np.zeros(64, int)
    t2[zig.ravel()] = 0
    # t dalam urutan zigzag: posisi raster r punya indeks zigzag zig[r]
    raster = t[zig]
    terbaik, sel = 0, 1e18
    for q in range(1, 101):
        s = 5000 / q if q < 50 else 200 - 2 * q
        e = np.clip(np.floor((std * s + 50) / 100), 1, 255).astype(int)
        d = np.abs(e - raster).sum()
        if d < sel:
            terbaik, sel = q, d
    return terbaik, int(sel)


def exif_ringkas(p):
    out = {}
    try:
        with Image.open(p) as im:
            ex = im.getexif()
            for k, v in ex.items():
                nama = ExifTags.TAGS.get(k, str(k))
                if nama in ("Make", "Model", "Software", "DateTime", "Orientation", "Artist", "ImageDescription", "Copyright"):
                    out["exif_" + nama] = str(v)[:80]
            out["exif_ntag"] = len(ex)
    except Exception as e:
        out["exif_err"] = repr(e)[:60]
    return out


def main():
    rows = []
    for split in ("train", "test"):
        for p in sorted((DATA / split).iterdir()):
            b = p.read_bytes()
            r = {"split": split, "image_id": p.stem, "ext": p.suffix.lower(), "size": len(b),
                 "sha1": hashlib.sha1(b).hexdigest()}
            with Image.open(p) as im:
                r.update(w=im.width, h=im.height, fmt=im.format, mode=im.mode,
                         dpi=str(im.info.get("dpi", "")), n_info=len(im.info), info_keys=",".join(sorted(im.info)))
            r["trail"] = len(b) - (b.rfind(b"\xff\xd9") + 2) if r["fmt"] == "JPEG" else -1
            if r["fmt"] == "JPEG":
                j = parse_jpeg(b)
                r.update(jfif=j["jfif"], comment=j["comment"], progressive=j["progressive"], sof=j["sof"],
                         subsamp=j["subsamp"], n_dqt=j["n_dqt"], n_dht=j["n_dht"], dri=j["dri"], adobe=j["adobe"],
                         icc=j["icc"], exif_len=j["exif_len"], app_bytes=j["app_bytes"],
                         app_list="|".join(j["app_list"]), markers="".join(j["markers"][:12]))
                if j["dqt"]:
                    h = hashlib.md5()
                    for tq, t in j["dqt"]:
                        h.update(bytes([tq]))
                        h.update(t.astype(np.int32).tobytes())
                    r["qhash"] = h.hexdigest()[:10]
                    q, sel = kualitas_ijg(j["dqt"][0][1])
                    r["q_ijg"], r["q_sel"] = q, sel
                    r["q_mean"] = float(j["dqt"][0][1].mean())
            r.update(exif_ringkas(p))
            rows.append(r)
    df = pd.DataFrame(rows)
    df["ar"] = df.w / df.h
    df["bpp"] = df["size"] * 8 / (df.w * df.h)
    df["mp"] = df.w * df.h / 1e6
    df.to_csv(OUT / "meta.csv", index=False)
    print(df.groupby("split").size())
    print(df.columns.tolist())


if __name__ == "__main__":
    main()
