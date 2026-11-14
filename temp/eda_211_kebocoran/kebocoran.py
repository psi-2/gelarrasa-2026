"""EDA 211: near-duplicate dalam train, dampaknya ke validasi, dan konflik label di dalam grup.

1. Grup train-only memakai aturan notebook: pHash dihedral Hamming <= 4 ATAU cos ViT-L >= ambang kuantil 0,0721%.
2. Grup yang anggotanya tersebar di lebih dari satu cluster Run 1 = calon konflik label setelah penamaan.
3. Simulasi kebocoran tanpa label: target = cluster Run 1 (48 kelas), fitur = (a) thumbnail piksel 24x24, (b) embedding
   DINOv3 7B. kNN-1 dinilai dengan StratifiedKFold acak lawan StratifiedGroupKFold. Selisihnya = optimisme akibat salinan.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from eda_common import PHASH, RUN1, SEED, buka, montage, norml2, out_dir, path_train, run1

OUT = out_dir(__file__)
cl, boot, eg, zp = run1()
c = cl.cluster.values
N = len(c)
PH = np.load(PHASH)[:N]  # hanya train
iu = np.triu_indices(N, 1)

basis = PH[:, 0, :].astype(np.float32) * 2 - 1
HAM = np.min([(63 - (PH[:, t, :].astype(np.float32) * 2 - 1) @ basis.T) / 2 for t in range(8)], axis=0)
HAM = np.minimum(HAM, HAM.T)
C = eg @ eg.T
amb = np.sort(C[iu])[::-1][int(.000721 * len(iu[0]))]
print(f"ambang cos (kuantil 0,0721%): {amb:.4f}")
for h in (0, 2, 4):
    print(f"pasangan Hamming <= {h}: {int((HAM[iu] <= h).sum())}")
print(f"pasangan cos >= ambang: {int((C[iu] >= amb).sum())}")
_, G = connected_components(csr_matrix((HAM <= 4) | (C >= amb)), directed=False)
uk = np.bincount(G)
print(f"grup: {len(uk)} | terbesar {uk.max()} | citra dalam grup > 1: {int((uk[G] > 1).sum())} ({(uk[G] > 1).mean():.3f})")
_, G0 = connected_components(csr_matrix(HAM <= 2), directed=False)
print(f"salinan hampir identik (Hamming <= 2): {int((np.bincount(G0)[G0] > 1).sum())} citra di {int((np.bincount(G0) > 1).sum())} grup")

print("\n== konflik cluster di dalam grup")
g = pd.DataFrame({"grup": G, "cluster": c, "stab": cl.stab})
per = g[uk[G] > 1].groupby("grup").agg(n=("cluster", "size"), n_cluster=("cluster", "nunique"), stab=("stab", "mean"))
print(f"grup > 1 anggota: {len(per)} | tersebar di >1 cluster: {(per.n_cluster > 1).sum()} grup, "
      f"{int(per.n[per.n_cluster > 1].sum())} citra")
lintas_img = per.n_cluster.reindex(G).fillna(1).values > 1
print(f"stab rata-rata citra di grup lintas cluster {cl.stab[lintas_img].mean():.3f} vs citra lain {cl.stab[~lintas_img].mean():.3f}")
pas_h = np.flatnonzero((HAM[iu] <= 2))
beda = np.mean(c[iu[0][pas_h]] != c[iu[1][pas_h]])
print(f"pasangan hampir identik (Hamming <= 2) yang jatuh di cluster berbeda: {beda:.3f}")

print("\n== simulasi kebocoran (target cluster Run 1, kNN-1)")
thumb = np.stack([np.asarray(buka(path_train(i)).resize((24, 24), 2), np.float32).ravel() for i in cl.image_id])
thumb = norml2(thumb - thumb.mean(0))
e7 = {k: np.load(RUN1 / f"emb_7b_train_{k}.npy").astype(np.float32) for k in ("asli_cls", "flip_cls")}
z7 = norml2(e7["asli_cls"] + e7["flip_cls"])
hasil = []
for nama, X in (("piksel 24x24", thumb), ("DINOv3 7B CLS", z7)):
    for skema, sp in (("acak", StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, c)),
                      ("grup", StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, c, G))):
        acc, bocor = [], []
        for a, b in sp:
            nn = a[np.argmax(X[b] @ X[a].T, 1)]
            acc.append(np.mean(c[nn] == c[b]))
            bocor.append(np.mean(G[nn] == G[b]))
        hasil.append(dict(fitur=nama, skema=skema, acc=np.mean(acc), std=np.std(acc), nn_satu_grup=np.mean(bocor)))
H = pd.DataFrame(hasil)
print(H.round(4).to_string(index=False))
for f, d in H.groupby("fitur", sort=False):
    print(f"{f}: optimisme split acak = {d.acc.iloc[0] - d.acc.iloc[1]:+.4f}")

pd.DataFrame({"image_id": cl.image_id, "grup": G, "ukuran": uk[G]}).to_csv(OUT / "grup_train.csv", index=False)
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].hist(uk[uk > 1], bins=np.arange(2, uk.max() + 2), log=True)
ax[0].set(title="ukuran grup near-duplicate (> 1)", xlabel="anggota")
ax[1].hist(C[iu][C[iu] > .6], 80, log=True)
ax[1].axvline(amb, c="k", ls="--")
ax[1].set(title="cos pasangan train (ViT-L resep), garis = ambang")
x = np.arange(2)
for j, (f, d) in enumerate(H.groupby("fitur", sort=False)):
    ax[2].bar(x + j * .35, d.acc, .35, yerr=d["std"], label=f)
ax[2].set_xticks(x + .17, ["acak", "grup"])
ax[2].set(title="kNN-1 ke cluster Run 1: split acak vs grup", ylabel="accuracy")
ax[2].legend()
fig.tight_layout()
fig.savefig(OUT / "kebocoran.png", dpi=110)
lintas = per[per.n_cluster > 1].sort_values("n", ascending=False).index[:4]
idx = [i for gg in lintas for i in np.flatnonzero(G == gg)[:6]]
montage([path_train(cl.image_id[i]) for i in idx], [f"{cl.image_id[i]} g{G[i]} c{c[i]}" for i in idx], OUT / "montage_grup_lintas_cluster.png", ncol=6)
print("simpan kebocoran.png, montage_grup_lintas_cluster.png, grup_train.csv di", OUT)
