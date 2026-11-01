"""Langkah 4: kesepakatan antar sumber independen dan dengan EXP-009, tanpa temuan.

Prediksi test EXP-009 hanya dibandingkan, tidak dipakai untuk fit atau aturan label.
Vote cluster berbagi embedding DINOv3 yang sama, jadi argmax per citra dilaporkan juga sebagai ukuran yang lebih independen.

Keluaran: output/tabel_train.csv

    python temp/eda_112_label_v2/sepakat.py
"""
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
E9 = ROOT / "temp" / "exp_009_t1_dinov3" / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
nt = 1920
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)
ids = np.load(E6 / "ids_train.npy").astype(str)
ids_te = np.load(E6 / "ids_test.npy").astype(str)
lf = pd.read_csv(E9 / "label_folds.csv", dtype={"image_id": str})
m = pd.read_csv(ROOT / "temp" / "eda_102_sumber" / "output" / "meta_assign.csv", dtype={"image_id": str})
fam = m[m.split == "train"].set_index("image_id").loc[ids].fam.values
fam_te = m[m.split == "test"].set_index("image_id").loc[ids_te].fam.values

d = pd.DataFrame({"image_id": ids, "fam": fam}).merge(lf[["image_id", "label", "suara", "fold"]], on="image_id", how="left")
for k in ("in1k", "in22k", "dms", "in22k_dms", "semua"):
    V = np.load(OUT / f"vote_{k}.npy")
    d[f"b_{k}"] = np.array(NAMES)[V[:nt].argmax(1)]
    d[f"s_{k}"] = V[:nt].max(1).round(4)
for k in ("in1k", "in22k", "dms"):
    d[f"c_{k}"] = np.array(NAMES)[np.load(OUT / f"citra_{k}.npy")[:nt]]
sisa = sorted(set(ids) - set(lf.image_id))
ps = np.load(E9 / "holdout" / "sisa_fold0.npy")
assert len(sisa) == len(ps) == 294
d = d.merge(pd.DataFrame({"image_id": sisa, "e9": np.array(NAMES)[ps.argmax(1)], "e9_p": ps.max(1).round(4)}), on="image_id", how="left")
oof = np.load(E9 / "holdout" / "oof_fold0.npy")
va = lf[lf.fold == 0].image_id.values
assert len(va) == len(oof)
o = pd.DataFrame({"image_id": va, "e9o": np.array(NAMES)[oof.argmax(1)], "e9o_p": oof.max(1).round(4)})
d = d.merge(o, on="image_id", how="left")
d["e9"] = d.e9.fillna(d.e9o)
d["e9_p"] = d.e9_p.fillna(d.e9o_p)
d.to_csv(OUT / "tabel_train.csv", index=False)

lab = d[d.label.notna()].copy()
lab["y_"] = lab.label
print("n berlabel", len(lab), "tanpa label", int(d.label.isna().sum()))
print("\n== kesepakatan dengan label lama (t1), per sumber ==")
cols = ["b_in22k", "b_dms", "b_in22k_dms", "c_in1k", "c_in22k", "c_dms"]
print(pd.Series({c: round(float((lab[c] == lab.label).mean()), 4) for c in cols}).to_string())
print("\n== antar sumber baru, semua train; DMS dihitung tanpa label electronic ==")
ne = d.b_in22k != "electronic"
print("bag in22k vs dms (tanpa electronic):", round(float((d.b_in22k == d.b_dms)[ne].mean()), 4))
print("citra in22k vs dms (tanpa electronic):", round(float((d.c_in22k == d.c_dms)[d.c_in22k != 'electronic'].mean()), 4))
print("citra in1k vs in22k:", round(float((d.c_in1k == d.c_in22k).mean()), 4))

print("\n== per kelas label lama: porsi setuju ==")
t = lab.groupby("label").apply(lambda g: pd.Series({"n": len(g), **{c: (g[c] == g.y_).mean() for c in cols}}), include_groups=False).reindex(NAMES).round(3)
print(t.to_string())
print("\n== per keluarga sumber: porsi setuju ==")
t = lab.groupby("fam").apply(lambda g: pd.Series({"n": len(g), **{c: (g[c] == g.y_).mean() for c in cols}}), include_groups=False).round(3)
print(t.to_string())

print("\n== label lama (baris) vs bag in22k_dms (kolom) ==")
print(pd.crosstab(lab.label, lab.b_in22k_dms).reindex(index=NAMES, columns=NAMES).fillna(0).astype(int).to_string())
print("\n== label lama vs citra in22k ==")
print(pd.crosstab(lab.label, lab.c_in22k).reindex(index=NAMES, columns=NAMES).fillna(0).astype(int).to_string())
print("\n== label lama vs citra dms ==")
print(pd.crosstab(lab.label, lab.c_dms).reindex(index=NAMES, columns=NAMES).fillna(0).astype(int).to_string())

print("\n== pasangan fokus: label lama A, kedua bag baru (in22k dan dms) bilang B ==")
for a, b in [("metal", "electronic"), ("electronic", "metal"), ("plastic", "paper"), ("paper", "plastic"), ("plastic", "glass"), ("glass", "plastic"), ("metal", "plastic"), ("plastic", "metal"), ("metal", "glass"), ("glass", "metal")]:
    s = lab[lab.label == a]
    print(f"{a:10s}->{b:10s} in22k {int((s.b_in22k == b).sum()):3d} dms {int((s.b_dms == b).sum()):3d} keduanya {int(((s.b_in22k == b) & (s.b_dms == b)).sum()):3d} citra22k {int((s.c_in22k == b).sum()):3d} citradms {int((s.c_dms == b).sum()):3d}")

print("\n== 294 tanpa label ==")
u = d[d.label.isna()]
print("EXP-009 prediksi:", u.e9.value_counts().reindex(NAMES).fillna(0).astype(int).to_dict())
print("median keyakinan", round(float(u.e9_p.median()), 3), "| >=.9:", int((u.e9_p >= .9).sum()))
print("fam:", u.fam.value_counts().to_dict())
for c in ("b_in1k", "b_in22k", "b_dms", "b_in22k_dms", "c_in22k", "c_dms"):
    print(f"EXP-009 = {c}: {(u.e9 == u[c]).mean():.3f}")
print("EXP-009 per kelas setuju b_in22k / b_dms:")
print(pd.DataFrame([{"kelas": c, "n": int((u.e9 == c).sum()), "in22k": round(float((u.b_in22k == c)[u.e9 == c].mean()), 3), "dms": round(float((u.b_dms == c)[u.e9 == c].mean()), 3),
                     "keduanya": round(float(((u.b_in22k == c) & (u.b_dms == c))[u.e9 == c].mean()), 3)} for c in NAMES]).to_string(index=False))

print("\n== hold-out 388: EXP-009 vs label dan sumber ==")
h = d[d.fold == 0]
print("EXP-009 = label", round(float((h.e9 == h.label).mean()), 4), "| = b_in22k", round(float((h.e9 == h.b_in22k).mean()), 4), "| = b_dms", round(float((h.e9 == h.b_dms).mean()), 4))
print(h[h.e9 != h.label][["label", "e9", "b_in22k", "b_dms", "c_in22k", "c_dms"]].to_string(index=False))

print("\n== test, hanya perbandingan ==")
pt = np.load(E9 / "holdout" / "test_fold0.npy")
yt = np.array(NAMES)[pt.argmax(1)]
print("EXP-009 test:", pd.Series(yt).value_counts().reindex(NAMES).to_dict())
ev = {}
for k in ("in1k", "in22k", "dms", "in22k_dms", "semua"):
    ev[k] = np.array(NAMES)[np.load(OUT / f"vote_{k}.npy")[nt:].argmax(1)]
    print(k, "setuju EXP-009", round(float((ev[k] == yt).mean()), 4))
for k in ("in22k", "dms"):
    ev["c_" + k] = np.array(NAMES)[np.load(OUT / f"citra_{k}.npy")[nt:]]
    print("citra", k, "setuju EXP-009", round(float((ev["c_" + k] == yt).mean()), 4))
b = (ev["in22k"] != yt) & (ev["dms"] != yt) & (ev["in22k"] == ev["dms"])
print("test: kedua bag sepakat melawan EXP-009:", int(b.sum()))
print(pd.crosstab(pd.Series(yt[b], name="e9"), pd.Series(ev["in22k"][b], name="sumber")).to_string())
print(pd.crosstab(pd.Series(yt, name="e9"), pd.Series(ev["in22k_dms"], name="in22k_dms")).reindex(index=NAMES, columns=NAMES).fillna(0).astype(int).to_string())
print("keluarga test tempat tidak setuju:", pd.Series(fam_te[ev["in22k_dms"] != yt]).value_counts().to_dict())
