"""Peta kategori ImageNet-22k dan DMS46 dari nama resmi saja. Dibekukan sebelum inferensi.

ImageNet-22k: subset diambil lewat timm.data.infer_imagenet_subset, nama dari ImageNetInfo (lemma dan gloss).
Rumus: (1) gerbang rentang synset membuang organisme, manusia, tempat. (2) cocokkan regex pada lemma.
(3) bila lemma tidak kena, cocokkan regex material pada gloss. (4) multi-kategori dibagi rata.
Tidak ada keputusan per cluster dan tidak ada data kompetisi di sini.

DMS46: nama material resmi. Electronic tidak pernah dipetakan, karena layar dilabeli glass.

    python temp/eda_112_label_v2/peta.py          # tinjau nama
    python temp/eda_112_label_v2/peta.py --beku   # bekukan, gagal bila hasil inferensi sudah ada
"""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import timm
from timm.data import ImageNetInfo, infer_imagenet_subset

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
E8 = ROOT / "temp" / "exp_008_in22k_map" / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
ARCH = ("convnext_large.fb_in22k", "swinv2_large_window12_192.ms_in22k", "beitv2_large_patch16_224.in1k_ft_in22k")

# DMS46: indeks kolom ke kategori. Dari nama taksonomi resmi.
DMS_PETA = {"Cardboard": "cardboard", "Carpet/rug": "textile", "Fabric/cloth": "textile", "Food": "foodwaste", "Glass": "glass",
            "Metal": "metal", "Paper": "paper", "Plastic, clear": "plastic", "Plastic, non-clear": "plastic"}

LEMMA = {
    "cardboard": r"\b(cardboard|carton|corrugated|paperboard|pasteboard|boxboard|cardboard box)\b",
    "paper": r"\b(paper|newspaper|newsprint|magazine|book|paperback|hardback|notebook|notepad|envelope|tissue|napkin|postcard|"
             r"brochure|pamphlet|leaflet|flyer|handbill|receipt|stationery|poster|comic|ticket|manuscript|letter paper|paper towel|document)\b",
    "plastic": r"\b(plastic|plastics|polyethylene|polystyrene|styrofoam|cellophane|plastic bag|garbage bag|shopping bag|water bottle|"
               r"pop bottle|soda bottle|squeeze bottle|tupperware|bubble wrap|shrink wrap|plastic wrap)\b",
    "glass": r"\b(glass|glassware|goblet|wineglass|tumbler|beaker|carafe|decanter|flagon|snifter|vial|phial|ampule|jar|wine bottle|beer bottle|"
             r"vase|champagne flute|demijohn|carboy)\b",
    "metal": r"\b(can|tin can|tin|beer can|soda can|canister|aluminum|aluminium|foil|tinfoil|steel|ironware|copper|brass|bronze|metal|metalware|"
             r"scrap metal|cutlery|tableware fork|nail|screw|bolt|nut and bolt|washer|pliers|wrench|spanner|hammer|saucepan|frying pan|skillet|kettle)\b",
    "textile": r"\b(textile|fabric|cloth|clothing|garment|apparel|shirt|t-shirt|tee shirt|sweater|jumper|jeans|trousers|pants|dress|skirt|blouse|"
               r"jacket|coat|sock|towel|blanket|curtain|rug|carpet|scarf|shawl|hoodie|sweatshirt|underwear|pajama|denim|wool|cotton|linen|silk|tweed)\b",
    "electronic": r"\b(computer|laptop|notebook computer|desktop computer|telephone|phone|smartphone|cellular telephone|television|tv|monitor|radio|printer|"
                  r"keyboard|loudspeaker|speaker|headphone|earphone|charger|battery|circuit|circuit board|transistor|remote control|calculator|router|modem|"
                  r"camera|camcorder|video recorder|stereo|amplifier|tuner|microwave|toaster|hair dryer|vacuum cleaner|electric fan|blender|electric)\b",
}
GLOSS = {
    "cardboard": r"\b(cardboard|paperboard|pasteboard|corrugated)\b",
    "paper": r"\b(made of paper|of paper|paper container|sheet of paper)\b",
    "plastic": r"\b(made of plastic|of plastic|plastic container|plastic)\b",
    "glass": r"\b(made of glass|of glass|glass container|glass bottle|glass vessel|blown glass)\b",
    "metal": r"\b(metal|metallic|aluminum|aluminium|steel|cast iron|tin)\b",
    "textile": r"\b(garment|clothing|fabric|textile|woven|knitted|cloth|worn (by|on|over|under))\b",
    "electronic": r"\b(electronic|electrical device|electric device|computer|circuit|powered by electricity|electrically)\b",
}
# lemma yang memuat kata material tetapi bukan benda bermaterial itu
BUANG = r"\b(sunglass|sunglasses|spyglass|hourglass|eyeglass|eyeglasses|magnifying glass|looking glass|field glass|opera glass|glasses|stained glass|flypaper|sandpaper|wallpaper|" \
        r"paper clip|paperweight|can opener|tin opener|fiberglass|fibreglass|bookcase|bookshelf|bookend|cannon|canopy|nut|washer|electric chair|electric guitar|electric organ)\b"


def boleh(s):
    """Gerbang rentang synset WordNet: artefak, komunikasi, makanan, substansi. Organisme dan manusia ditolak."""
    n = int(s[1:])
    return 2666000 <= n < 4990000 or 6000000 <= n < 7000000 or 7555000 <= n < 8000000 or 14000000 <= n < 16000000


def makanan(s):
    """Blok makanan WordNet, tanpa minuman."""
    n = int(s[1:])
    return 7555000 <= n < 7881800 or 7933000 <= n < 8000000


def peta22k(info):
    n = info.num_classes()
    M = np.zeros((8, n), dtype=np.float32)
    teks = []
    for i in range(n):
        s = info.index_to_label_name(i)
        d = info.index_to_description(i, detailed=True)
        lem, _, gl = d.partition(":")
        lem, gl = lem.lower(), gl.lower()
        teks.append((s, lem, gl))
        if not boleh(s):
            continue
        if re.search(BUANG, lem):
            continue
        if makanan(s):
            M[7, i] = 1
            continue
        kena = [c for c in LEMMA if re.search(LEMMA[c], lem)]
        if not kena:
            kena = [c for c in GLOSS if re.search(GLOSS[c], gl)]
        for c in kena:
            M[NAMES.index(c), i] = 1 / len(kena)
    return M, teks


def peta_dms():
    t = json.loads(Path(r"D:\cache_moved\dms46\taxonomy.json").read_text(encoding="utf-8"))
    DMS46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33, 34, 35,
             36, 37, 38, 39, 41, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 56]
    nama = [t["names"][i] for i in DMS46]
    M = np.zeros((8, 46), dtype=np.float32)
    for k, c in DMS_PETA.items():
        M[NAMES.index(c), nama.index(k)] = 1
    return M, nama


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beku", action="store_true")
    a = ap.parse_args()
    subset = {x: infer_imagenet_subset(timm.create_model(x, pretrained=False)) for x in ARCH}
    assert set(subset.values()) == {"imagenet-22k"}, subset
    info = ImageNetInfo("imagenet-22k")
    M, teks = peta22k(info)
    Md, nama = peta_dms()
    assert M[2].sum() > 0 and Md[2].sum() == 0
    print("kelas terpetakan:", int((M.sum(0) > 0).sum()), "dari", M.shape[1], {c: round(float(M[i].sum()), 1) for i, c in enumerate(NAMES)})
    print("DMS:", {c: [nama[j] for j in np.where(Md[i])[0]] for i, c in enumerate(NAMES)})
    rng = np.random.default_rng(7)
    for i, c in enumerate(NAMES):
        idx = np.where(M[i] > 0)[0]
        pil = rng.choice(idx, min(40, len(idx)), replace=False)
        print(f"\n{c} ({len(idx)}):", "; ".join(sorted(teks[j][1].split(",")[0][:24] for j in pil)))
    if a.beku:
        ada = list(E8.glob("p22k_*.npy")) + list(E8.glob("dms_*.npy"))
        assert not ada, f"hasil inferensi sudah ada: {ada[:3]}"
        np.save(OUT / "peta_in22k.npy", M)
        np.save(OUT / "peta_dms.npy", Md)
        kode = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        rek = {"waktu": time.strftime("%Y-%m-%d %H:%M:%S"), "sha256_peta_in22k": hashlib.sha256(M.tobytes()).hexdigest(),
               "sha256_peta_dms": hashlib.sha256(Md.tobytes()).hexdigest(), "sha256_kode": kode, "subset": subset,
               "catatan": "dibekukan sebelum inferensi p22k dan dms ada"}
        (OUT / "peta_beku.json").write_text(json.dumps(rek, indent=2), encoding="utf-8")
        print("\nBEKU", rek)


if __name__ == "__main__":
    main()
