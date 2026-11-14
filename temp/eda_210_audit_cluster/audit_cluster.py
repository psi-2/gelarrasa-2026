"""EDA 210: audit partisi Run 1 (DINOv3 ViT-L, K=48) tanpa label.

Pertanyaan: cluster mana yang rapuh, pasangan cluster mana yang saling bertukar anggota antar bootstrap,
dan cluster mana yang berisi dua kelompok terpisah (calon PECAH). Semua metrik train-only dan tanpa nama kelas.
Keluaran: output/audit_cluster.csv, output/subcluster_train.csv (usulan pecah k=2, huruf kanonik), gambar.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score

from eda_common import SEED, norml2, out_dir, run1

OUT = out_dir(__file__)
cl, boot, eg, zp = run1()
c = cl.cluster.values
K = c.max() + 1
ids = cl.image_id.values

print("== 1. reproduksi partisi")
ulang = KMeans(K, n_init=20, random_state=SEED).fit_predict(zp)
print(f"ARI partisi Run 1 (Kaggle) vs KMeans ulang lokal: {adjusted_rand_score(c, ulang):.4f}")

print("\n== 2. stabilitas dan pertukaran anggota antar bootstrap")
F = np.zeros((K, K))
for b in boot:
    np.add.at(F, (c, b), 1)
F /= F.sum(1, keepdims=True)
jac = np.array([[((c == k) & (b == k)).sum() / ((c == k) | (b == k)).sum() for b in boot] for k in range(K)]).mean(1)
tukar = F.copy()
np.fill_diagonal(tukar, 0)
print(f"citra stab >= 0,7: {(cl.stab >= .7).mean():.3f} | cluster Jaccard < 0,6: {(jac < .6).sum()} dari {K}")
pas = sorted(((tukar[i, j] + tukar[j, i], i, j) for i in range(K) for j in range(i + 1, K)), reverse=True)[:10]
print("10 pasangan cluster paling sering bertukar anggota (porsi pindah dua arah):")
for v, i, j in pas:
    print(f"  {i:2d} <-> {j:2d}: {v:.3f}")

print("\n== 3. kedekatan centroid")
cen = norml2(np.stack([zp[c == k].mean(0) for k in range(K)]))
S = cen @ cen.T
np.fill_diagonal(S, -1)
tet = np.argsort(-S, 1)[:, :3]
sebar = np.array([1 - (zp[c == k] @ cen[k]).mean() for k in range(K)])

print("\n== 4. calon pecah: KMeans k=2 di dalam tiap cluster")
rows, sub_kode = [], np.empty(len(c), object)
for k in range(K):
    idx = np.flatnonzero(c == k)
    s = KMeans(2, n_init=20, random_state=SEED).fit_predict(zp[idx])
    if s[0] != 0:  # huruf kanonik: sub-cluster yang memuat ID terkecil menjadi 'a'
        s = 1 - s
    sub_kode[idx] = [f"{k}{'ab'[v]}" for v in s]
    sc = norml2(np.stack([zp[idx[s == v]].mean(0) for v in (0, 1)]))
    lain = np.delete(np.arange(K), k)
    tet_sub = [int(lain[np.argmax(cen[lain] @ sc[v])]) for v in (0, 1)]
    rows.append(dict(
        cluster=k, n=len(idx), stab_mean=cl.stab.values[idx].mean().round(3), p_stab_lt07=(cl.stab.values[idx] < .7).mean().round(3),
        jaccard=jac[k].round(3), tukar_utama=int(np.argmax(tukar[k])), porsi_tukar=tukar[k].max().round(3),
        tetangga=" ".join(f"{t}({S[k, t]:.2f})" for t in tet[k]), sebaran=sebar[k].round(4),
        sil_pecah=round(float(silhouette_score(zp[idx], s, metric="cosine")), 3),
        rasio_pisah=((1 - sc[0] @ sc[1]) / (1 - S[k].max())).round(3),
        n_a=int((s == 0).sum()), n_b=int((s == 1).sum()), tetangga_a=tet_sub[0], tetangga_b=tet_sub[1]))
A = pd.DataFrame(rows)
# rasio_pisah >= 1: dua sub-kelompok berjarak sejauh atau lebih jauh dari jarak cluster ke tetangga terdekatnya
A["calon_pecah"] = (A.rasio_pisah >= 1) & (A[["n_a", "n_b"]].min(1) >= 5) & (A.tetangga_a != A.tetangga_b)
A.to_csv(OUT / "audit_cluster.csv", index=False)
pd.DataFrame({"image_id": ids, "cluster": c, "sub": sub_kode}).to_csv(OUT / "subcluster_train.csv", index=False)
with pd.option_context("display.width", 220, "display.max_columns", 30, "display.max_rows", 60):
    print(A.sort_values("rasio_pisah", ascending=False).head(15).to_string(index=False))
print(f"\ncalon pecah (rasio >= 1, sub >= 5, tetangga sub berbeda): {A.calon_pecah.sum()} cluster -> {A.cluster[A.calon_pecah].tolist()}")
print(f"korelasi Spearman stab_mean vs rasio_pisah: {A.stab_mean.corr(A.rasio_pisah, method='spearman'):.3f}")
print(f"korelasi Spearman stab_mean vs sebaran: {A.stab_mean.corr(A.sebaran, method='spearman'):.3f}")

print("\n== 5. peta 2D (t-SNE dari ZP, hanya visual)")
xy = TSNE(2, perplexity=30, init="pca", random_state=SEED).fit_transform(zp)
fig, ax = plt.subplots(2, 2, figsize=(17, 14))
cmap = plt.get_cmap("tab20")
ax[0, 0].scatter(xy[:, 0], xy[:, 1], c=[cmap(k % 20) for k in c], s=5, alpha=.7)
ax[0, 0].scatter(*xy[cl.stab < .7].T, facecolors="none", edgecolors="k", s=14, lw=.4)
for k in range(K):
    m = np.median(xy[c == k], 0)
    ax[0, 0].text(*m, str(k), fontsize=8, weight="bold", ha="center", va="center", bbox=dict(boxstyle="round,pad=.1", fc="w", alpha=.7, lw=0))
ax[0, 0].set(title="t-SNE ZP, angka = cluster Run 1, lingkaran hitam = stab < 0,7", xticks=[], yticks=[])
o = np.argsort([np.median(xy[c == k, 0]) for k in range(K)])
im = ax[0, 1].imshow(F[np.ix_(o, o)] ** .5, cmap="magma")
ax[0, 1].set_xticks(range(K), o, fontsize=5, rotation=90)
ax[0, 1].set_yticks(range(K), o, fontsize=5)
ax[0, 1].set(title="porsi pindah antar bootstrap (akar), baris = cluster asal", xlabel="cluster tujuan")
plt.colorbar(im, ax=ax[0, 1], fraction=.04)
ax[1, 0].bar(range(K), A.n, color=["C3" if j < .6 else "C0" for j in A.jaccard])
ax2 = ax[1, 0].twinx()
ax2.plot(range(K), A.stab_mean, "k.-", lw=.8)
ax2.set_ylim(0, 1.05)
ax[1, 0].set(title="ukuran cluster (merah = Jaccard bootstrap < 0,6), garis = stabilitas rata-rata", xlabel="cluster")
ax[1, 1].scatter(A.sil_pecah, A.rasio_pisah, c=A.stab_mean, cmap="viridis", s=40)
for _, r in A.iterrows():
    ax[1, 1].text(r.sil_pecah, r.rasio_pisah, str(r.cluster), fontsize=7)
ax[1, 1].axhline(1, c="k", ls="--", lw=.8)
ax[1, 1].set(xlabel="silhouette pecah k=2", ylabel="rasio pisah (jarak sub / jarak ke tetangga)", title="calon pecah (warna = stab_mean)")
fig.tight_layout()
fig.savefig(OUT / "audit_cluster.png", dpi=110)
print("simpan audit_cluster.png, audit_cluster.csv, subcluster_train.csv di", OUT)
