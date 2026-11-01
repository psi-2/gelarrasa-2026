"""Grup kosakata sempit dari nama resmi ImageNet-22k. Regex pada lemma. Tanpa data kompetisi di modul ini.

Dijalankan langsung untuk meninjau nama yang kena.

    python temp/eda_114_paper/kosakata_material.py
"""
import re

import numpy as np
from timm.data import ImageNetInfo

GRUP = {
    "foil": r"\b(tinfoil|tin foil|aluminum foil|aluminium foil|foil|silver paper|gold foil|gold leaf|metal foil)\b",
    "kain": r"\b(diaper|nappy|napkin|serviette|handkerchief|hankie|towel|washcloth|dishcloth|dishrag|rag|tatter|sheet|bedsheet|gauze|cheesecloth|muslin|cotton|tissue paper|facial tissue|tippet|cloth|fabric)\b",
    "kertas": r"\b(paper|newspaper|tabloid|gazette|daily|envelope|file folder|folder|notebook|exercise book|magazine|pamphlet|brochure|leaflet|flyer|printout|document|wrapping paper|crepe paper|tissue paper|paper towel)\b",
    "kardus": r"\b(carton|cardboard|packing box|packing case|bandbox|chipboard|corrugated|mailer|shoebox|box|pasteboard|paperboard)\b",
    "kemasan": r"\b(packet|sachet|packaging|package|parcel|wrapper|wrapping|blister pack|pouch|cellophane|plastic wrap|shrink-wrap|bin liner|plastic bag|grocery bag|shopping bag|garbage bag)\b",
    "botol_plastik": r"\b(bottled water|mineral water|branch water|seltzer|water bottle|drinking water|soda water|pop bottle)\b",
    "kaca": r"\b(phial|vial|jar|jampot|mason jar|glass|tumbler|flute|liqueur glass|wineglass|highball glass|shot glass|flagon|carboy|demijohn|carafe|decanter|goblet|champagne cup|wine bottle|beer bottle)\b",
    "kaleng": r"\b(can|tin|beer can|soda can|coffee can|canister|tinderbox|aerosol|canned food)\b",
}
BUANG = r"\b(hydrofoil|thermometer|wallpaper|sandpaper|flypaper|paper clip|paperweight|box wrench|box spring|sentry box|music box|penalty box|sound box|box kite|box camera|jury box|glasses|sunglasses|spyglass|hourglass|field glass|opera glass|magnifying glass|looking glass|bird|fish|tree|plant)\b"


def bangun(info=None):
    info = info or ImageNetInfo("imagenet-22k")
    n = info.num_classes()
    M = {k: np.zeros(n, dtype=bool) for k in GRUP}
    tek = []
    for i in range(n):
        s = info.index_to_label_name(i)
        nomor = int(s[1:])
        lem = info.index_to_description(i).lower()
        tek.append(lem)
        boleh = 2666000 <= nomor < 4990000 or 6000000 <= nomor < 7000000 or 7555000 <= nomor < 8000000 or 14000000 <= nomor < 16000000
        if not boleh or re.search(BUANG, lem):
            continue
        for k, r in GRUP.items():
            if re.search(r, lem):
                M[k][i] = True
    return M, tek


if __name__ == "__main__":
    M, tek = bangun()
    for k, v in M.items():
        idx = np.where(v)[0]
        print(f"\n{k} ({len(idx)}):", "; ".join(tek[i].split(",")[0][:26] for i in idx))
