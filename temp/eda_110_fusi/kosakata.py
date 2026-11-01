"""Peta kosakata a priori: nama kelas EoMT ke 8 kategori, ditentukan dari arti nama kelas saja.

Bobot pecahan dipakai untuk kelas wadah yang materialnya tidak ditentukan oleh namanya.
Tidak ada penamaan cluster di sini. Kelas latar tidak dipetakan.
"""
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]

COCO = {
    "wine glass": {"glass": 1}, "vase": {"glass": 1},
    "bottle": {"glass": .5, "plastic": .5}, "cup": {"glass": .34, "plastic": .33, "metal": .33},
    "paper-merged": {"paper": 1}, "book": {"paper": 1},
    "tv": {"electronic": 1}, "laptop": {"electronic": 1}, "mouse": {"electronic": 1}, "remote": {"electronic": 1},
    "keyboard": {"electronic": 1}, "cell phone": {"electronic": 1}, "microwave": {"electronic": 1},
    "toaster": {"electronic": 1}, "refrigerator": {"electronic": 1}, "hair drier": {"electronic": 1},
    "oven": {"electronic": 1}, "clock": {"electronic": 1},
    "fork": {"metal": 1}, "knife": {"metal": 1}, "spoon": {"metal": 1}, "scissors": {"metal": 1},
    "toothbrush": {"plastic": 1}, "frisbee": {"plastic": 1},
    "cardboard": {"cardboard": 1},
    "tie": {"textile": 1}, "towel": {"textile": 1}, "blanket": {"textile": 1}, "pillow": {"textile": 1},
    "curtain": {"textile": 1}, "rug-merged": {"textile": 1}, "handbag": {"textile": 1}, "backpack": {"textile": 1},
    "teddy bear": {"textile": 1},
    "banana": {"foodwaste": 1}, "apple": {"foodwaste": 1}, "sandwich": {"foodwaste": 1}, "orange": {"foodwaste": 1},
    "broccoli": {"foodwaste": 1}, "carrot": {"foodwaste": 1}, "hot dog": {"foodwaste": 1}, "pizza": {"foodwaste": 1},
    "donut": {"foodwaste": 1}, "cake": {"foodwaste": 1}, "fruit": {"foodwaste": 1}, "food-other-merged": {"foodwaste": 1},
    "bowl": {"foodwaste": .5, "glass": .25, "plastic": .25},
}

ADE = {
    "glass": {"glass": 1}, "vase": {"glass": 1},
    "bottle": {"glass": .5, "plastic": .5},
    "book": {"paper": 1}, "poster": {"paper": 1},
    "computer": {"electronic": 1}, "monitor": {"electronic": 1}, "crt screen": {"electronic": 1},
    "television receiver": {"electronic": 1}, "screen": {"electronic": 1}, "microwave": {"electronic": 1},
    "oven": {"electronic": 1}, "washer": {"electronic": 1}, "dishwasher": {"electronic": 1}, "fan": {"electronic": 1},
    "stove": {"electronic": 1}, "clock": {"electronic": 1}, "refrigerator": {"electronic": 1},
    "pot": {"metal": 1}, "barrel": {"metal": .5, "plastic": .5}, "tank": {"metal": 1}, "radiator": {"metal": 1},
    "bag": {"plastic": .5, "textile": .25, "paper": .25}, "plaything": {"plastic": 1}, "ball": {"plastic": 1},
    "box": {"cardboard": 1},
    "apparel": {"textile": 1}, "towel": {"textile": 1}, "blanket": {"textile": 1}, "rug": {"textile": 1},
    "cushion": {"textile": 1}, "pillow": {"textile": 1}, "curtain": {"textile": 1},
    "food": {"foodwaste": 1}, "plate": {"foodwaste": .5, "glass": .25, "plastic": .25},
    "tray": {"plastic": .5, "metal": .5}, "basket": {"textile": .5, "plastic": .5},
}
