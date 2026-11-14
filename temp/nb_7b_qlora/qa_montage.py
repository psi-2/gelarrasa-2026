"""QA preprocessing: montage SEMUA citra train per kelas + grid sebelum/sesudah augmentasi.

    .venv/Scripts/python.exe temp/nb_7b_qlora/qa_montage.py

Hanya untuk memeriksa preprocessing (trim pita hitam, orientasi EXIF, stretch, augmentasi). Hasil penglihatan
tidak dipakai untuk melabeli, memindah, atau membuang citra (QnA #16, #20). Fungsi preprocessing diambil dari
temp/prep_302_verifikasi_run1/pipa.py, salinan persis fungsi notebook.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms import v2 as T

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR / "temp/prep_302_verifikasi_run1"))
from pipa import aug, muat_citra, tf_eval_tampil, trim_hitam  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
UK, KOL, BARIS = 96, 16, 12
RRC = (0.5, 1.0)

label = pd.read_csv(AKAR / "results/label_train.csv")
berkas = {int(p.stem): p for p in (AKAR / "data/train").iterdir()}
assert len(berkas) == 1920 and set(berkas) == set(label.image_id)


def info(i):
    p = berkas[i]
    with Image.open(p) as im:
        a = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    t = trim_hitam(a)
    return dict(image_id=i, w=a.shape[1], h=a.shape[0], trim=t.shape != a.shape,
                aspek=t.shape[1] / t.shape[0], kecil=min(a.shape[:2]) < 200)


with ThreadPoolExecutor(8) as ex:
    X = dict(zip(label.image_id, ex.map(lambda i: muat_citra(berkas[i]), label.image_id)))
    meta = pd.DataFrame(list(ex.map(info, label.image_id))).set_index("image_id")
print("trim", int(meta.trim.sum()), "| aspek ekstrem", int(((meta.aspek > 1.6) | (meta.aspek < 0.62)).sum()),
      "| sisi pendek < 200", int(meta.kecil.sum()))


def kotak(i):
    t = Image.fromarray(X[i]).resize((UK, UK), Image.BILINEAR)
    d = ImageDraw.Draw(t)
    tanda = ("T" if meta.trim[i] else "") + ("A" if not 0.62 <= meta.aspek[i] <= 1.6 else "") + ("K" if meta.kecil[i] else "")
    d.rectangle([0, 0, 40, 11], fill=(0, 0, 0))
    d.text((1, 0), f"{i}{tanda}", fill=(255, 255, 0) if tanda else (255, 255, 255))
    return t


grup = {k: d.image_id.tolist() for k, d in label[label.berlabel].groupby("nama")}
grup["tak_berlabel"] = label.image_id[~label.berlabel].tolist()
halaman = []
for kls, ids in grup.items():
    per = KOL * BARIS
    for h in range(0, len(ids), per):
        bag = ids[h:h + per]
        kanvas = Image.new("RGB", (KOL * UK, (len(bag) + KOL - 1) // KOL * UK + 16), (40, 40, 40))
        ImageDraw.Draw(kanvas).text((4, 2), f"{kls} | {len(ids)} citra | halaman {h // per + 1} | T=trim A=aspek ekstrem K=sisi<200", fill=(255, 255, 255))
        for j, i in enumerate(bag):
            kanvas.paste(kotak(i), (j % KOL * UK, 16 + j // KOL * UK))
        nama = OUT / f"montage_{kls}_{h // per + 1}.jpg"
        kanvas.save(nama, quality=85)
        halaman.append(nama.name)
print(len(halaman), "halaman montage:", halaman)

torch.manual_seed(42)
tampil = T.Compose([T.ToImage(), *aug(RRC)])
rng = np.random.default_rng(42)
kasus = {kls: list(rng.choice(ids, 3, replace=False)) for kls, ids in grup.items()}
kasus["tepi_trim"] = meta[meta.trim].sample(3, random_state=1).index.tolist()
kasus["tepi_aspek"] = meta[(meta.aspek > 1.6) | (meta.aspek < 0.62)].sample(3, random_state=1).index.tolist()
kasus["tepi_kecil"] = meta[meta.kecil].sample(3, random_state=1).index.tolist()
U2 = 160
baris = [(k, i) for k, ids in kasus.items() for i in ids]
for h in range(0, len(baris), 18):
    bag = baris[h:h + 18]
    kanvas = Image.new("RGB", (6 * U2, len(bag) * U2 + 16), (40, 40, 40))
    ImageDraw.Draw(kanvas).text((4, 2), "kolom: asli EXIF | trim+stretch 512 | eval 384 | 3x augmentasi train (RRC 0,5 flip rot90 jitter)", fill=(255, 255, 255))
    for r, (kls, i) in enumerate(bag):
        with Image.open(berkas[i]) as im:
            asli = ImageOps.exif_transpose(im).convert("RGB")
        sel = [ImageOps.pad(asli, (U2, U2), color=(90, 0, 90))] + [Image.fromarray(X[i]), tf_eval_tampil(X[i])] + [tampil(X[i]) for _ in range(3)]
        for c, s in enumerate(sel):
            s = s if isinstance(s, Image.Image) else T.functional.to_pil_image(s)
            kanvas.paste(s.resize((U2, U2), Image.BILINEAR), (c * U2, 16 + r * U2))
        ImageDraw.Draw(kanvas).text((3, 18 + r * U2), f"{kls} {i}", fill=(255, 255, 0))
    nama = OUT / f"augmentasi_{h // 18 + 1}.jpg"
    kanvas.save(nama, quality=85)
    print("grid", nama.name, len(bag), "baris")
meta.to_csv(OUT / "meta_qa.csv")
