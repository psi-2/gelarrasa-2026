"""Uji 10: bangun berkas label usulan v3 dan saring isian dengan aturan beku.

Langkah: label bersih (aturan_label.py), tambah isian 294 pada ambang AMBANG_ISI (dua head sepakat, p >= ambang pada keduanya), jalankan
aturan beku deteksi.py sekali lagi pada gabungan, buang isian yang ditandai. Label lama tidak dikoreksi di ronde kedua, supaya satu ronde
aturan tidak mengubah hasil ronde pertama.

AMBANG_ISI 0,7 dipilih setelah membandingkan 5 varian di uji9 memakai wasit 22k + DMS, bukan temuan. Pilihan ini pasca-lihat, jadi dukungan
wasit 11 lawan 0 sedikit optimistis.

fold dan group dipertahankan. Isian dari group baru masuk fold 1 (latih). Isian dalam group hold-out ikut fold 0, agar group tidak bocor,
tetapi fold 0 baru sebaiknya dinilai pada baris lama saja. Kolom `sumber` memberi tahu asal baris.

    python temp/research_uji_crossfit/uji10_label_v3.py
"""
from common import *  # noqa: F401,F403
from common import E9, NAMES, OUT, Waktu, muat, muat_cnn, simpan_json
from deteksi import oof_coteach, putuskan

import numpy as np
import pandas as pd

AMBANG_ISI = .7
log = Waktu()
d = muat()
Ev, y, grp, ids, V, fam = d["E"], d["y"], d["grp"], d["ids"], d["V"], d["fam"]
Ec, _ = muat_cnn(d["ids"], d["ids_te"])
isisa = np.where(y < 0)[0]
s = np.load(OUT / "uji8_sisa.npz")
pa, pb = s["pa"], s["pb"]
ua = pa.argmax(1)
isi = (ua == pb.argmax(1)) & (pa.max(1) >= AMBANG_ISI) & (pb.max(1) >= AMBANG_ISI)

bersih = pd.read_csv(OUT / "label_folds_bersih.csv", dtype={"image_id": str})
pos = {k: i for i, k in enumerate(ids)}
idx = np.array([pos[k] for k in bersih.image_id])
Y = np.full(1920, -1)
Y[idx] = [NAMES.index(c) for c in bersih.label]
Y[isisa[isi]] = ua[isi]
ii = np.where(Y >= 0)[0]
log(f"bersih {len(idx)}, isian {int(isi.sum())}, gabungan {len(ii)}")

a, b = oof_coteach(Ev[ii], Ec[ii], Y[ii], grp[ii], seed=3030)
h = putuskan(a, b, Y[ii])
tanda = h["koreksi"] | h["buang"]
adalah_isian = y[ii] < 0
isi_ditolak = tanda & adalah_isian
lama_ditandai = tanda & ~adalah_isian
print("ditandai ronde 2:", int(tanda.sum()), "| isian ditolak:", int(isi_ditolak.sum()), "| baris lama ditandai, dibiarkan:", int(lama_ditandai.sum()))
print("isian ditolak per kelas:", pd.Series(np.array(NAMES)[Y[ii][isi_ditolak]]).value_counts().to_dict())
pd.DataFrame({"image_id": ids[ii][lama_ditandai], "label": np.array(NAMES)[Y[ii][lama_ditandai]], "alt": np.array(NAMES)[h["alt"][lama_ditandai]],
              "koreksi": h["koreksi"][lama_ditandai]}).to_csv(OUT / "ronde2_lama_ditandai.csv", index=False)

terima = ii[adalah_isian & ~tanda]
fold_grup = bersih.groupby("group").fold.first().to_dict()
g_map = pd.Series(bersih.group.values, index=[pos[k] for k in bersih.image_id])
# Cek penomoran group sama dengan label_folds EXP-009.
assert (g_map.sort_index().values == grp[g_map.sort_index().index]).all(), "penomoran group berbeda"
bersih["sumber"] = np.where(bersih.image_id.isin(pd.read_csv(OUT / "daftar_label_diragukan.csv", dtype={"image_id": str}).image_id), "koreksi", "exp009")
tambah = pd.DataFrame({"image_id": ids[terima], "label": np.array(NAMES)[Y[terima]], "suara": V[terima].max(1).round(4), "group": grp[terima], "fam": fam[terima],
                       "fold": [fold_grup.get(g, 1) for g in grp[terima]], "sumber": "isian"})
v3 = pd.concat([bersih, tambah], ignore_index=True).sort_values("image_id").reset_index(drop=True)
assert v3.image_id.is_unique and not (v3.groupby("group").fold.nunique() > 1).any()
asli = pd.read_csv(E9 / "label_folds.csv", dtype={"image_id": str})
assert set(bersih.image_id) <= set(asli.image_id)
v3.to_csv(OUT / "label_folds_v3.csv", index=False)
print("v3:", len(v3), v3.label.value_counts().reindex(NAMES).to_dict())
print(pd.crosstab(v3.sumber, v3.fold).to_string())
print("isian diterima:", tambah.label.value_counts().reindex(NAMES).fillna(0).astype(int).to_dict())
simpan_json("uji10_info.json", {"isian_kandidat": int(isi.sum()), "isian_ditolak": int(isi_ditolak.sum()), "isian_diterima": len(tambah), "n_v3": len(v3),
                                "baris_lama_ditandai_ronde2": int(lama_ditandai.sum()), "fold0": int((v3.fold == 0).sum())})
log("selesai")
