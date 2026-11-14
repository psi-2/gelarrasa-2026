"""EDA 213: kesepakatan dua ruang embedding (DINOv3 ViT-L Run 1 vs DINOv3 7B) per cluster dan per citra.

Cluster yang juga utuh di ruang 7B lebih bisa dipercaya sebagai satu kelompok visual. Citra yang tetangga 7B-nya
mayoritas berada di cluster lain adalah kandidat buangan (aturan kemiripan tetangga, QnA 2 dan 5 Nov). Train-only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from eda_common import RUN1, SEED, norml2, out_dir, run1

OUT = out_dir(__file__)
cl, boot, eg, zp = run1()
c = cl.cluster.values
K = c.max() + 1
G = pd.read_csv(Path(__file__).resolve().parents[1] / "eda_211_kebocoran/output/grup_train.csv").grup.values
e7 = {k: np.load(RUN1 / f"emb_7b_train_{k}.npy").astype(np.float32) for k in ("asli_cls", "asli_patch", "flip_cls", "flip_patch")}
eg7 = norml2(np.hstack([norml2(e7["asli_cls"] + e7["flip_cls"]), norml2(e7["asli_patch"] + e7["flip_patch"])]))
z7 = norml2(PCA(128, random_state=SEED).fit_transform(eg7))
c7 = KMeans(K, n_init=20, random_state=SEED).fit_predict(z7)
print(f"ARI partisi ViT-L (Run 1) vs 7B, K=48: {adjusted_rand_score(c, c7):.3f}")
ct = pd.crosstab(c, c7)
utuh = (ct.max(1) / ct.sum(1)).values
print(f"porsi terbesar cluster ViT-L yang tetap bersama di 7B: median {np.median(utuh):.3f}, < 0,6 pada {(utuh < .6).sum()} cluster")


def knn_setuju(z, k=10):
    S = z @ z.T
    S[G[:, None] == G[None, :]] = -np.inf  # tetangga satu grup duplikat tidak dihitung
    nn = np.argsort(-S, 1)[:, :k]
    return (c[nn] == c[:, None]).mean(1)


s_vit, s_7b = knn_setuju(zp), knn_setuju(z7)
print(f"kesepakatan kNN-10 dengan cluster sendiri (tanpa tetangga satu grup): ViT-L {s_vit.mean():.3f}, 7B {s_7b.mean():.3f}")
for t in (.3, .5):
    print(f"  porsi citra < {t}: ViT-L {(s_vit < t).mean():.3f}, 7B {(s_7b < t).mean():.3f}, keduanya {((s_vit < t) & (s_7b < t)).mean():.3f}")
print(f"Spearman stab bootstrap vs kNN 7B: {pd.Series(cl.stab).corr(pd.Series(s_7b), method='spearman'):.3f}")
A = pd.DataFrame({"cluster": range(K), "utuh_7b": utuh.round(3),
                  "knn7b_mean": pd.Series(s_7b).groupby(c).mean().values.round(3),
                  "knnvit_mean": pd.Series(s_vit).groupby(c).mean().values.round(3)})
A.to_csv(OUT / "dua_ruang_cluster.csv", index=False)
pd.DataFrame({"image_id": cl.image_id, "cluster": c, "cluster_7b": c7, "knn_vit": s_vit.round(3), "knn_7b": s_7b.round(3),
              "stab": cl.stab}).to_csv(OUT / "dua_ruang_citra.csv", index=False)
print(A.sort_values("utuh_7b").head(12).to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
ax[0].bar(range(K), utuh, color=["C3" if u < .6 else "C0" for u in utuh])
ax[0].set(title="porsi cluster ViT-L yang tetap satu cluster di 7B", xlabel="cluster Run 1")
ax[1].hist([s_vit, s_7b], 11, label=["ViT-L", "7B"])
ax[1].set(title="kesepakatan kNN-10 dengan cluster sendiri", xlabel="porsi tetangga di cluster sama")
ax[1].legend()
ax[2].scatter(cl.stab + np.random.default_rng(0).normal(0, .01, len(c)), s_7b, s=4, alpha=.4)
ax[2].set(xlabel="stab bootstrap (ViT-L)", ylabel="kNN-10 7B", title="dua sinyal keraguan per citra")
fig.tight_layout()
fig.savefig(OUT / "dua_ruang.png", dpi=110)
print("simpan dua_ruang.png, dua_ruang_cluster.csv, dua_ruang_citra.csv di", OUT)
