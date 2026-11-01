"""Audit daftar kelas ImageNet-21k/22k/12k/1k untuk 8 kategori sampah.

Hanya membaca berkas teks daftar kelas di paket timm. Tidak menyentuh citra kompetisi.
Pertanyaan: seberapa banyak kelas material dan benda sampah yang tersedia di head 21k/12k
dibanding head 1k, dan apakah kelas kunci yang hilang di 1k (kaleng, kardus, kertas) ada.

    python temp/research_unsup/audit_label_21k.py
"""
import re
from pathlib import Path

import timm

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
INFO = Path(timm.__file__).parent / "data" / "_info"


def baca(nama):
    return [x.strip() for x in (INFO / nama).read_text(encoding="utf-8").splitlines() if x.strip()]


lemma = {}
for baris in baca("imagenet_synset_to_lemma.txt"):
    k, v = baris.split("\t", 1)
    lemma[k] = v
defin = {}
for baris in baca("imagenet_synset_to_definition.txt"):
    k, v = baris.split("\t", 1)
    defin[k] = v

set_1k = set(baca("imagenet_synsets.txt"))
set_12k = set(baca("imagenet12k_synsets.txt"))
set_21k = set(baca("imagenet21k_goog_synsets.txt"))
set_22k = set(baca("imagenet22k_synsets.txt"))
print("ukuran:", {"1k": len(set_1k), "12k": len(set_12k), "21k": len(set_21k), "22k": len(set_22k)})
print("lemma tersedia:", len(lemma), "definisi:", len(defin))

KUNCI = {
    "cardboard": r"cardboard|corrugated|pasteboard|carton|paperboard|egg box|shoebox|box\b",
    "paper": r"\bpaper\b|newspaper|newsprint|tissue|napkin|magazine|wrapping paper|paper bag|receipt|notebook paper|"
             r"envelope|paper towel|cup, paper|handbill|leaflet|brochure",
    "plastic": r"plastic|polyethylene|polystyrene|styrofoam|cellophane|vinyl|pvc|polymer|nylon|tupperware|"
               r"squeeze bottle|shopping bag",
    "glass": r"\bglass\b|jar\b|mason jar|jam jar|pop bottle|beer bottle|wine bottle|bottle\b|decanter|tumbler|"
             r"carafe|flask|shard",
    "metal": r"\bmetal\b|\bcan\b|tin can|beer can|aluminum|aluminium|foil|\bsteel\b|\biron\b|\btin\b|scrap|"
             r"copper|brass|wire\b|\bcan,|canister",
    "textile": r"fabric|textile|cloth|garment|\bwool\b|cotton|denim|linen|shirt|t-shirt|rag\b|towel|\bsock\b",
    "electronic": r"electronic|circuit|computer|battery|charger|\bcable\b|printed circuit|keyboard|television|"
                  r"cell phone|cellular telephone|smartphone|laptop|hard disk|motherboard|headphone|"
                  r"calculator|game console",
    "foodwaste": r"\bfood\b|garbage|compost|peel|rind|leftover|scrap|eggshell|coffee grounds|\bcore\b|"
                 r"banana|apple|vegetable|\bbread\b|\bmeat\b",
    "sampah_umum": r"garbage|rubbish|trash|\bwaste\b|refuse|litter|dump|landfill|scrap|junk|debris",
}

baris_out = []
for kat, pola in KUNCI.items():
    r = re.compile(pola, re.I)
    # Cocokkan pada lemma saja, bukan definisi, agar tidak terlalu luas.
    hit = sorted(s for s in set_21k if s in lemma and r.search(lemma[s]))
    h1 = [s for s in hit if s in set_1k]
    h12 = [s for s in hit if s in set_12k]
    print(f"\n== {kat}: 21k {len(hit)} | 12k {len(h12)} | 1k {len(h1)}")
    for s in hit:
        baris_out.append(f"{kat}\t{s}\t{'1k' if s in set_1k else ''}\t{'12k' if s in set_12k else ''}\t{lemma[s]}\t{defin.get(s, '')[:90]}")

(OUT / "audit_21k_kandidat.tsv").write_text("\n".join(baris_out), encoding="utf-8")

CEK = ["paper", "cardboard", "corrugated fastener", "plastic", "plastic bag", "glass", "tin can", "can", "beer can",
       "aluminum foil", "tinfoil", "scrap", "garbage", "rag", "newspaper", "carton", "cloth", "fabric", "compost",
       "leftovers", "food", "shredded paper", "printed circuit", "battery", "styrofoam", "jar", "bottle", "cup",
       "paper cup", "paper bag", "wrapping paper", "cellophane", "wrapper", "metal", "steel", "soda can",
       "cardboard box", "box", "egg carton", "motherboard", "charger", "keyboard", "smartphone"]
print("\n== cek lemma persis ==")
for c in CEK:
    pat = re.compile(rf"(^|, ){re.escape(c)}(,|$)", re.I)
    ada = [s for s in sorted(set_21k) if s in lemma and pat.search(lemma[s])]
    keterangan = [f"{s}[{'1k' if s in set_1k else ('12k' if s in set_12k else '21k')}]" for s in ada[:6]]
    print(f"{c:22s} -> {keterangan}")
