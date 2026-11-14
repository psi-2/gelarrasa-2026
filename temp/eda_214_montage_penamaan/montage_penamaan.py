"""EDA 214: bahan penamaan cluster untuk TIM (bukan untuk Claude).

Montage Run 1 hanya 26 citra kecil (sekitar 96 px). Di sini tiap cluster mendapat montage sampai 40 citra 150 px,
dipisah per sub-cluster usulan k=2 (EDA 210), dengan judul merah untuk citra stab < 0,7. Tidak ada nama kelas,
tidak ada skor ImageNet. Juga ditulis tabel prioritas tinjau dan templat nama.
Keluaran: results/gambar_v2/cluster_<k>.png, results/prioritas_tinjau.csv, results/nama_cluster_templat.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda_common import ROOT, TEMP, buka, path_train

OUT = ROOT / "results/gambar_v2"
OUT.mkdir(parents=True, exist_ok=True)
sub = pd.read_csv(TEMP / "eda_210_audit_cluster/output/subcluster_train.csv")
cl = pd.read_csv(ROOT / "results/cache/cluster_train.csv").merge(sub[["image_id", "sub"]], on="image_id")
au = pd.read_csv(TEMP / "eda_210_audit_cluster/output/audit_cluster.csv")
dr = pd.read_csv(TEMP / "eda_213_dua_ruang/output/dua_ruang_cluster.csv")
at = pd.read_csv(TEMP / "eda_212_jalan_pintas/output/cluster_atribut_dominan.csv")
P = au.merge(dr, on="cluster")
P["atribut_dominan"] = P.cluster.map(at.groupby("cluster").apply(lambda d: "; ".join(f"{a}={v} {p:.0%}" for a, v, p in zip(d.atribut, d.nilai, d.porsi)))).fillna("")
# skor prioritas: makin rapuh, makin campur, makin terpecah di 7B -> makin perlu ditinjau teliti
P["prioritas"] = ((1 - P.jaccard) + (P.rasio_pisah.clip(0, 1.5) / 1.5) + (1 - P.utuh_7b) + (1 - P.knn7b_mean)).round(3)
P = P.sort_values("prioritas", ascending=False)
P[["cluster", "n", "prioritas", "calon_pecah", "n_a", "n_b", "stab_mean", "jaccard", "utuh_7b", "knn7b_mean", "tukar_utama",
   "porsi_tukar", "tetangga", "atribut_dominan"]].to_csv(ROOT / "results/prioritas_tinjau.csv", index=False)
print(P[["cluster", "n", "prioritas", "calon_pecah", "jaccard", "utuh_7b", "tukar_utama"]].head(12).to_string(index=False))

rng = np.random.default_rng(42)
NCOL = 10
for k, d in cl.groupby("cluster"):
    bagian = []
    for s, ds in d.groupby("sub"):
        jatah = round(40 * len(ds) / len(d)) if len(d) > 40 else len(ds)
        ds = ds.sort_values("stab", ascending=False)
        pilih = ds.iloc[: max(jatah - 3, 1)]
        pinggir = ds[ds.stab < .7].iloc[::-1][:3]  # selalu tampilkan sampai 3 citra paling ragu
        bagian.append((s, pd.concat([pilih, pinggir]).drop_duplicates("image_id")))
    nrow = sum(int(np.ceil(len(b) / NCOL)) for _, b in bagian)
    fig, ax = plt.subplots(nrow, NCOL, figsize=(NCOL * 1.6, nrow * 1.75 + .5), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    r = 0
    for s, b in bagian:
        for j, (_, row) in enumerate(b.iterrows()):
            a = ax[r + j // NCOL, j % NCOL]
            a.imshow(buka(path_train(row.image_id), 150))
            a.set_title(f"{row.image_id} {s[-1]} {row.stab:.2f}", fontsize=7, color="r" if row.stab < .7 else "k")
        r += int(np.ceil(len(b) / NCOL))
    p = P.set_index("cluster").loc[k]
    fig.suptitle(f"cluster {k} | n={len(d)} | sub a={p.n_a} b={p.n_b}{' | CALON PECAH' if p.calon_pecah else ''} | "
                 f"stab {p.stab_mean:.2f} | sering bertukar dengan {p.tukar_utama} | judul: id, sub, stab (merah < 0,7)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"cluster_{k}.png", dpi=85)
    plt.close(fig)
tpl = []
for k in sorted(cl.cluster.unique()):
    tpl.append(dict(kode=str(k), n=int((cl.cluster == k).sum()), nama=""))
    for s in ("a", "b"):
        tpl.append(dict(kode=f"{k}{s}", n=int((cl["sub"] == f"{k}{s}").sum()), nama=""))
pd.DataFrame(tpl).to_csv(ROOT / "results/nama_cluster_templat.csv", index=False)
print("tulis", len(list(OUT.glob("cluster_*.png"))), "montage ke", OUT)
