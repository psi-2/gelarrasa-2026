"""Klasifikasi sampah non-VLM: ConvNeXtV2 ImageNet-1k lokal + pemetaan kelas ke 8 kategori.

Tanpa prompt teks dan tanpa text encoder. Massa softmax kelas ImageNet dijumlahkan per kategori.
Kelas ImageNet yang tidak dipetakan diabaikan. Pemetaan dibuat dari nama kelas saja.

    python temp/exp_005_imagenet_map/imagenet_map.py [--limit 32]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.utils.seed_utils import prediction_hash, seed_everything  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import AutoImageProcessor, AutoModelForImageClassification  # noqa: E402

DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "output"
SRC = r"D:\cache_moved\convnextv2_huge_clean"
VRAM_MAKS_GIB = 13.5

PETA = {
    "glass": ["beer bottle", "beer glass", "wine bottle", "goblet", "vase", "whiskey jug", "pitcher",
              "beaker", "measuring cup", "perfume", "Petri dish"],
    "paper": ["envelope", "binder", "paper towel", "toilet tissue", "comic book", "crossword puzzle",
              "book jacket", "menu", "web site"],
    "electronic": ["cellular telephone", "computer keyboard", "desktop computer", "dial telephone",
                   "hand-held computer", "hard disc", "iPod", "laptop", "loudspeaker", "modem", "monitor",
                   "mouse", "notebook", "pay-phone", "photocopier", "printer", "projector", "radio",
                   "remote control", "screen", "space bar", "television", "typewriter keyboard",
                   "cassette player", "CD player", "tape player", "cassette", "digital clock",
                   "digital watch", "joystick", "microphone", "oscilloscope", "Polaroid camera",
                   "reflex camera", "switch", "home theater", "entertainment center", "microwave",
                   "toaster", "vacuum", "electric fan", "hand blower", "refrigerator", "washer",
                   "dishwasher", "espresso maker", "waffle iron", "iron", "space heater",
                   "sewing machine", "cash machine"],
    "metal": ["bottlecap", "cocktail shaker", "milk can", "bucket", "caldron", "Dutch oven", "frying pan",
              "wok", "chain", "nail", "screw", "screwdriver", "hammer", "hatchet", "cleaver",
              "letter opener", "safety pin", "buckle", "padlock", "combination lock", "hook", "steel drum",
              "strainer", "ladle", "spatula", "can opener", "corkscrew", "oil filter", "hair spray",
              "lighter", "disk brake", "car wheel", "manhole cover", "safe", "coffeepot", "brass", "shovel",
              "radiator", "rain barrel"],
    "plastic": ["pop bottle", "water bottle", "water jug", "plastic bag", "packet", "pill bottle", "lotion",
                "soap dispenser", "sunscreen", "tub", "syringe", "tray"],
    "cardboard": ["carton", "crate", "pencil box"],
    "textile": ["abaya", "academic gown", "apron", "bath towel", "bib", "bikini", "bolo tie", "bonnet",
                "bow tie", "brassiere", "cardigan", "cloak", "dishrag", "doormat", "feather boa", "fur coat",
                "gown", "handkerchief", "hoopskirt", "jean", "jersey", "kimono", "lab coat", "maillot",
                "military uniform", "miniskirt", "mitten", "overskirt", "pajama", "pillow", "poncho",
                "prayer rug", "quilt", "sarong", "sock", "stole", "suit", "sweatshirt", "swimming trunks",
                "trench coat", "velvet", "vestment", "Windsor tie", "wool", "backpack", "purse", "knot",
                "sleeping bag", "cowboy hat", "sombrero", "clog", "cowboy boot", "Loafer", "running shoe",
                "sandal", "bathing cap", "Christmas stocking", "teddy"],
    "foodwaste": ["guacamole", "consomme", "hot pot", "trifle", "ice cream", "ice lolly", "French loaf", "bagel",
                  "pretzel", "cheeseburger", "hotdog", "mashed potato", "head cabbage", "broccoli",
                  "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash",
                  "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith",
                  "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit",
                  "custard apple", "pomegranate", "carbonara", "chocolate sauce", "dough", "meat loaf",
                  "pizza", "potpie", "burrito", "corn", "acorn", "hip", "ear", "plate", "soup bowl"],
}
NAMES = list(PETA)


class Citra(Dataset):
    def __init__(self, files, proc):
        self.files, self.proc = files, proc

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as im:
            rgb = im.convert("RGB")
        return self.proc(images=[rgb, rgb.transpose(Image.FLIP_LEFT_RIGHT)],
                         return_tensors="pt")["pixel_values"]


def matriks_peta(id2label):
    """Baris kategori, kolom kelas ImageNet. Gagal bila ada nama salah atau kelas ganda."""
    nama = [id2label[i].split(",")[0].strip() for i in range(len(id2label))]
    M = np.zeros((len(NAMES), len(nama)), dtype=np.float32)
    for c, daftar in enumerate(NAMES):
        for n in PETA[daftar]:
            idx = [i for i, x in enumerate(nama) if x == n]
            assert idx, f"nama kelas tidak ditemukan: {n}"
            assert not M[:, idx].any(), f"kelas dipetakan dua kali: {n}"
            M[c, idx] = 1
    return M, nama


@torch.no_grad()
def inferensi(model, files, proc, bs):
    hasil = []
    for px in DataLoader(Citra(files, proc), batch_size=bs, num_workers=4, shuffle=False):
        b, v = px.shape[:2]
        px = px.flatten(0, 1).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(pixel_values=px).logits
        hasil.append(torch.softmax(logit.float(), 1).view(b, v, -1).mean(1).cpu())
    return torch.cat(hasil).numpy()


def ringkas(split, ids, p1k, M, nama):
    mapped = (M @ p1k.T).T
    cakupan = mapped.sum(1)
    relatif = mapped / np.clip(cakupan[:, None], 1e-9, None)
    pred = np.array(NAMES)[relatif.argmax(1)]
    keyakinan = relatif.max(1)
    teratas = p1k.argmax(1)
    terpetakan = M.sum(0) > 0

    df = pd.DataFrame(relatif, columns=NAMES).round(5)
    df.insert(0, "image_id", ids)
    df.insert(1, "pred", pred)
    df.insert(2, "keyakinan", keyakinan.round(5))
    df.insert(3, "cakupan", cakupan.round(5))
    df.insert(4, "top1_imagenet", [nama[i] for i in teratas])
    df.to_csv(OUT / f"prob_{split}.csv", index=False)

    tak_terpetakan = pd.Series([nama[i] for i in teratas[~terpetakan[teratas]]]).value_counts()
    per_kelas = {c: pd.Series([nama[i] for i in teratas[pred == c]]).value_counts().head(8).to_dict() for c in NAMES}
    return {
        "distribusi": pd.Series(pred).value_counts().reindex(NAMES).fillna(0).astype(int).to_dict(),
        "cakupan_median": round(float(np.median(cakupan)), 4),
        "porsi_cakupan_lt_0.2": round(float((cakupan < 0.2).mean()), 4),
        "porsi_top1_tak_terpetakan": round(float((~terpetakan[teratas]).mean()), 4),
        "keyakinan_rata": round(float(keyakinan.mean()), 4),
        "porsi_keyakinan_ge_0.9": round(float((keyakinan >= 0.9).mean()), 4),
        "top1_tak_terpetakan": tak_terpetakan.head(30).to_dict(),
        "top1_per_kategori": per_kelas,
        "hash": prediction_hash(relatif),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    seed_everything(42, warn_only=True)
    OUT.mkdir(exist_ok=True)
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(VRAM_MAKS_GIB / total_gib, 0)

    proc = AutoImageProcessor.from_pretrained(SRC, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(SRC, local_files_only=True, dtype=torch.bfloat16).cuda().eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    M, nama = matriks_peta(id2label)
    print(f"kelas terpetakan: {int(M.sum())} dari {M.shape[1]} | per kategori: "
          f"{dict(zip(NAMES, M.sum(1).astype(int).tolist()))}", flush=True)

    laporan, t0 = {}, time.time()
    for split in ("test", "train"):
        files = sorted((DATA / split).iterdir())[:args.limit]
        ids = [p.stem for p in files]
        p1k = inferensi(model, files, proc, args.bs)
        np.save(OUT / f"imagenet_{split}.npy", p1k.astype(np.float32))
        laporan[split] = ringkas(split, ids, p1k, M, nama)
        print(f"{split} selesai, {time.time() - t0:.0f}s", flush=True)
        if split == "test":
            df = pd.read_csv(OUT / "prob_test.csv", dtype={"image_id": str})
            df[["image_id", "pred"]].rename(columns={"pred": "waste_category"}).to_csv(
                OUT / "draft_submission_convnextv2.csv", index=False)

    laporan["vram_puncak_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    laporan["detik"] = round(time.time() - t0, 1)
    print(json.dumps(laporan, indent=2, ensure_ascii=False))
    (OUT / "laporan.json").write_text(json.dumps(laporan, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
