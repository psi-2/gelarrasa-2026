"""Uji pemetaan otomatis kelas sumber ke 8 kategori dari gloss WordNet.

Hanya membaca berkas teks daftar kelas di paket timm. Tidak menyentuh citra kompetisi,
tidak memakai text encoder, tidak memakai LLM. Aturannya regex deterministik atas gloss,
yaitu frasa bahan ("made of paper", "of glass") dan frasa kategori.

Keluaran dipakai untuk menilai cakupan pada head 1k dan 12k, dan membandingkan dengan
tabel PETA tulisan tangan di EXP-005.

    python temp/research_unsup/audit_gloss_map.py
"""
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import timm

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)
INFO = Path(timm.__file__).parent / "data" / "_info"
ROOT = BASE.parents[1]


def baca(nama):
    return [x.strip() for x in (INFO / nama).read_text(encoding="utf-8").splitlines() if x.strip()]


lemma = dict(b.split("\t", 1) for b in baca("imagenet_synset_to_lemma.txt"))
defin = dict(b.split("\t", 1) for b in baca("imagenet_synset_to_definition.txt"))
set_1k = baca("imagenet_synsets.txt")
set_12k = baca("imagenet12k_synsets.txt")

MATERIAL = {
    "paper": r"paper|newsprint|cardboard-free",
    "cardboard": r"cardboard|pasteboard|paperboard|corrugated",
    "plastic": r"plastic|polyethylene|polystyrene|vinyl|nylon|polymer",
    "glass": r"glass|crystal",
    "metal": r"metal|steel|\biron\b|\btin\b|aluminum|aluminium|brass|copper|bronze",
    "textile": r"cloth|fabric|textile|wool|cotton|linen|silk|knit|woven|garment",
}
# Pola bahan: gloss harus menyatakan benda terbuat dari bahan itu, bukan sekadar menyebut kata.
POLA_BAHAN = {k: re.compile(rf"(made (of|from|out of)|consisting of|composed of|of (thin |heavy |stiff |sheet |clear |flexible |"
                            rf"strong |lightweight |transparent )*)\s*({v})", re.I) for k, v in MATERIAL.items()}
POLA_KATEGORI = {
    "electronic": re.compile(r"electronic|electric(al)? device|computer|circuit|telephone|digital|battery", re.I),
    "foodwaste": re.compile(r"\b(edible|eaten|food|fruit|vegetable|meat|dish of|baked|cooked)\b", re.I),
}


def petakan(s):
    g = defin.get(s, "")
    kandidat = [k for k, p in POLA_BAHAN.items() if p.search(g)]
    kandidat += [k for k, p in POLA_KATEGORI.items() if p.search(g)]
    return kandidat


laporan = {}
baris = []
for nama, himpunan in (("1k", set_1k), ("12k", set_12k)):
    tunggal, jamak = Counter(), Counter()
    for s in himpunan:
        k = petakan(s)
        if len(k) == 1:
            tunggal[k[0]] += 1
        elif len(k) > 1:
            jamak["+".join(sorted(k))] += 1
        if k:
            baris.append(f"{nama}\t{s}\t{'+'.join(sorted(k))}\t{lemma.get(s, '')[:45]}\t{defin.get(s, '')[:100]}")
    n_t, n_j = sum(tunggal.values()), sum(jamak.values())
    laporan[nama] = {"total": len(himpunan), "satu_kandidat": n_t, "banyak_kandidat": n_j,
                     "per_kategori": dict(tunggal.most_common()), "pasangan_teratas": dict(jamak.most_common(8))}
    print(nama, json.dumps(laporan[nama], ensure_ascii=False))

(OUT / "gloss_map.tsv").write_text("\n".join(baris), encoding="utf-8")

# Bandingkan dengan peta tulisan tangan EXP-005 tanpa menjalankan modelnya: parse blok PETA dari sumbernya.
src = (ROOT / "temp" / "exp_005_imagenet_map" / "imagenet_map.py").read_text(encoding="utf-8")
blok = src[src.index("PETA = {"):src.index("NAMES = list(PETA)")]
ruang = {}
exec(blok, {}, ruang)  # blok hanya berisi literal dict
PETA = ruang["PETA"]
nama1k = {}
for s in set_1k:
    nama1k.setdefault(lemma[s].split(",")[0].strip().lower(), []).append(s)
cocok, beda, kosong = 0, [], 0
for kat, daftar in PETA.items():
    for n in daftar:
        for s in nama1k.get(n.lower(), []):
            k = petakan(s)
            if not k:
                kosong += 1
            elif kat in k:
                cocok += 1
            else:
                beda.append((n, kat, k))
print(f"\nbanding PETA EXP-005: cocok {cocok}, tanpa kandidat gloss {kosong}, berbeda {len(beda)}")
print("contoh berbeda:", beda[:25])

print("\ncek kelas ambigu:")
for n in ["packet", "tray", "plate", "can opener", "water bottle", "pop bottle", "beer bottle", "carton", "envelope",
          "plastic bag", "ashcan", "paper towel", "bucket", "crate", "wine bottle", "cup", "mailbag"]:
    for s in nama1k.get(n, []):
        print(f"  {n:15s} {petakan(s)} | {defin.get(s, '')[:95]}")
