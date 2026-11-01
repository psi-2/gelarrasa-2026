"""Periksa nama kelas ImageNet-22k dari timm dan ketersediaan WordNet. Tidak menyentuh data kompetisi.

    python temp/exp_008_in22k_map/lihat_taksonomi.py
"""
import re
from pathlib import Path

from timm.data import ImageNetInfo

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

info = ImageNetInfo("imagenet-22k")
n = info.num_classes()
print("kelas:", n)
baris = []
for i in range(n):
    s = info.index_to_label_name(i)
    desc = info.index_to_description(i, detailed=True)
    baris.append(f"{i}\t{s}\t{desc}")
(OUT / "in22k_kelas.tsv").write_text("\n".join(baris), encoding="utf-8")

try:
    from nltk.corpus import wordnet as wn
    print("wordnet synsets:", len(list(wn.all_synsets("n"))))
except Exception as e:
    print("wordnet tidak tersedia:", type(e).__name__, str(e)[:100])

pola = {
    "kaleng": r"\b(tin can|tin|beer can|aluminum can|soda can|can, tin|canister|canned)\b",
    "kardus": r"\b(cardboard|carton|corrugated|pasteboard|paperboard)\b",
    "kertas": r"\b(newspaper|paper|magazine|tissue|wrapping paper|notepaper|sheet of paper)\b",
    "plastik": r"\b(plastic|polyethylene|polystyrene|styrofoam)\b",
    "kaca": r"\b(glass|glassware|jar|mason jar|bottle)\b",
    "sisa": r"\b(garbage|trash|rubbish|waste|refuse|litter|compost|scrap)\b",
}
for k, p in pola.items():
    hit = [b for b in baris if re.search(p, b.split("\t")[2].lower())]
    print(f"\n{k}: {len(hit)}")
    for b in hit[:14]:
        print("  ", b.split("\t")[0], b.split("\t")[2][:90])
