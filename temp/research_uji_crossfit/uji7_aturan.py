"""Uji 7: validasi aturan beku pada noise sintetis, kontrol lupa nol, dan efek pada hold-out.

Bagian A, kontrol. Detektor dengan lupa 0 memakai pengoptimal dan epoch sama. Bila recall noise cluster tetap tinggi, penyebabnya
bukan mekanisme co-teaching.

Bagian B, aturan utuh pada seluruh 1.626 label, konfigurasi produksi. Seed noise baru, tidak dipakai pada uji2 dan uji6.

Bagian C, efek pada hold-out. Label latih dirusak, aturan dijalankan pada bagian latih saja, lalu ViT LR plus CNN LR dilatih pada
label hasil aturan dan dinilai pada hold-out berlabel asli. R dan S dikecilkan agar waktu wajar.

    python temp/research_uji_crossfit/uji7_aturan.py
"""
from common import *  # noqa: F401,F403
from common import OUT, Waktu, bulat, lipat, lr_fit_predict, muat, muat_cnn, simpan_json
from deteksi import LUPA, TOLAK, YAKIN, oof_coteach, putuskan
from sim_label import injeksi

import numpy as np
import pandas as pd

log = Waktu()
d = muat()
il = np.where(d["y"] >= 0)[0]
Ev, V, y0, g0 = d["E"][il], d["V"][il], d["y"][il], d["grp"][il]
Ec = muat_cnn(d["ids"], d["ids_te"])[0][il]

# --- A: kontrol lupa nol ---
ka = []
for jenis, laju in (("bingung", .05), ("cluster", .05), ("cluster", .15)):
    for ul in range(2):
        seed = 7100 + 10 * int(laju * 100) + ul + (0 if jenis == "bingung" else 1000)
        Yk = injeksi(y0, V, Ev, jenis, laju, seed)
        kotor = Yk != y0
        for lupa in (0.0, LUPA):
            a, b = oof_coteach(Ev, Ec, Yk, g0, lupa=lupa, R=1, S=1, seed=seed)
            hasil = putuskan(a, b, Yk, tolak=.5)
            k = hasil["koreksi"]
            ka.append({"jenis": jenis, "laju": laju, "lupa": lupa, "n_kotor": int(kotor.sum()), "diubah": int(k.sum()), "pulih": int((hasil["alt"][k] == y0[k]).sum())})
    log(f"kontrol {jenis} {laju}")
tk = pd.DataFrame(ka).groupby(["jenis", "laju", "lupa"], sort=False).mean().round(1)
tk["recall"] = (tk.pulih / tk.n_kotor).round(3)
tk["presisi"] = (tk.pulih / tk.diubah).round(3)
print(tk.to_string())
tk.to_csv(OUT / "uji7_kontrol.csv")

# --- B: aturan utuh ---
bb = []
for jenis, laju in (("bersih", 0.0), ("acak", .05), ("bingung", .05), ("cluster", .03), ("cluster", .05), ("cluster", .10)):
    for ul in range(2):
        seed = 7300 + 100 * int(laju * 100) + ul + hash(jenis) % 1
        Yk = y0.copy() if laju == 0 else injeksi(y0, V, Ev, jenis, laju, seed + (7 if jenis == "acak" else 0))
        kotor = Yk != y0
        a, b = oof_coteach(Ev, Ec, Yk, g0, seed=seed)
        h = putuskan(a, b, Yk)
        k, x = h["koreksi"], h["buang"]
        bb.append({"jenis": jenis, "laju": laju, "ul": ul, "n_kotor": int(kotor.sum()), "koreksi": int(k.sum()), "koreksi_pulih": int((h["alt"][k] == y0[k]).sum()),
                   "koreksi_kena_bersih": int((k & ~kotor).sum()), "buang": int(x.sum()), "buang_kotor": int((x & kotor).sum()), "kotor_tersentuh": int(((k | x) & kotor).sum())})
    log(f"aturan {jenis} {laju}")
    pd.DataFrame(bb).to_csv(OUT / "uji7_aturan_mentah.csv", index=False)
tb = pd.DataFrame(bb).drop(columns="ul").groupby(["jenis", "laju"], sort=False).mean().round(1)
tb["recall_koreksi"] = (tb.koreksi_pulih / tb.n_kotor).round(3)
tb["presisi_koreksi"] = (tb.koreksi_pulih / tb.koreksi).round(3)
tb["presisi_buang"] = (tb.buang_kotor / tb.buang).round(3)
tb["kotor_tersisa_porsi"] = (1 - tb.kotor_tersentuh / tb.n_kotor).round(3)
print(tb.to_string())
tb.to_csv(OUT / "uji7_aturan.csv")

# --- C: efek pada hold-out ---
cc = []
for jenis, laju in (("bersih", 0.0), ("bingung", .05), ("cluster", .05), ("cluster", .10)):
    for lf, (tr, va) in enumerate(lipat(y0, g0, 5, seed=77)):
        seed = 7700 + 10 * int(laju * 100) + lf
        Yb = y0[tr]
        Yk = Yb.copy() if laju == 0 else injeksi(Yb, V[tr], Ev[tr], jenis, laju, seed)
        Ea, Eb, G = Ev[tr], Ec[tr], g0[tr]
        a, b = oof_coteach(Ea, Eb, Yk, G, R=2, S=2, seed=seed)
        h = putuskan(a, b, Yk)
        cur = Yk.copy()
        cur[h["koreksi"]] = h["alt"][h["koreksi"]]
        jaga = ~h["buang"]
        yv = y0[va]

        def nilai(Y, m):
            pa = lr_fit_predict(Ea[m], Y[m], [Ev[va]])[0]
            pb = lr_fit_predict(Eb[m], Y[m], [Ec[va]])[0]
            return float((pa.argmax(1) == yv).mean()), float((((pa + pb) / 2).argmax(1) == yv).mean())
        semua = np.ones(len(Yk), bool)
        v0, g0_ = nilai(Yk, semua)
        v1, g1 = nilai(cur, jaga)
        cc.append({"jenis": jenis, "laju": laju, "lf": lf, "vit_kotor": v0, "gabung_kotor": g0_, "vit_aturan": v1, "gabung_aturan": g1})
    log(f"hold-out {jenis} {laju}")
tc = pd.DataFrame(cc).drop(columns="lf").groupby(["jenis", "laju"], sort=False).mean().round(4)
print(tc.to_string())
tc.to_csv(OUT / "uji7_holdout.csv")
simpan_json("uji7_info.json", {"LUPA": LUPA, "TOLAK": TOLAK, "YAKIN": YAKIN, "contoh": bulat(.5)})
log("selesai")
