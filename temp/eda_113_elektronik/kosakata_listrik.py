"""Kosakata listrik dari nama resmi ImageNet-22k. Regex pada lemma, tanpa data kompetisi di modul ini.

Dijalankan langsung untuk meninjau nama yang kena.

    python temp/eda_113_elektronik/kosakata_listrik.py
"""
import re

import numpy as np
from timm.data import ImageNetInfo

GRUP = {
    "sel": r"\b(battery|batteries|dry cell|dry battery|storage cell|storage battery|voltaic cell|voltaic battery|galvanic cell|primary cell|secondary cell|"
           r"solar cell|fuel cell|nicad|nickel-cadmium|alkaline cell|lead-acid|energizer|weston cell|daniell cell|clark cell|cadmium cell|mercury cell)\b",
    "lampu": r"\b(light bulb|lightbulb|incandescent lamp|fluorescent lamp|fluorescent|fluorescent fixture|neon lamp|neon tube|electric lamp|bulb|floodlight|spotlight|halogen lamp|mercury-vapor lamp|sodium-vapor lamp|led)\b",
    "kabel": r"\b(electric cable|power cord|extension cord|extension|cord|cable|wire|wiring|coaxial cable|adapter|charger|battery charger|plug|socket|electric outlet|power strip|connector|jack|fuse|circuit breaker)\b",
    "papan": r"\b(circuit board|printed circuit|cpu board|motherboard|mother board|circuitry|circuit|integrated circuit|chip|microchip|transistor|resistor|capacitor|diode|semiconductor device|computer circuit|logic chip|memory chip)\b",
}
# lemma mirip yang bukan benda listrik
BUANG = r"\b(thermometer|battery park|artillery|gun battery|poker chip|potato chip|wood chip|chip shot|bulb onion|tulip|onion|barbed wire|chicken wire|wire cloth|jackfruit|jack-o'-lantern|jackstraw|jackknife|lumberjack|bootjack|car jack)\b"


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
        artefak = 2666000 <= nomor < 4990000 or 6000000 <= nomor < 7000000
        if re.search(BUANG, lem):
            continue
        for k, r in GRUP.items():
            if re.search(r, lem) and (artefak or k == "sel"):
                M[k][i] = True
    return M, tek


if __name__ == "__main__":
    M, tek = bangun()
    for k, v in M.items():
        idx = np.where(v)[0]
        print(f"\n{k} ({len(idx)}):", "; ".join(tek[i].split(",")[0][:30] for i in idx))
