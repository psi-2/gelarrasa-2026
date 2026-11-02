"""EDA 208: imbalance antar-cluster + metrik kualitas cluster tanpa label (baseline ViT-L dan ConvNeXt-L).

Fungsi `metrik_ruang` identik dengan yang dipakai notebook Kaggle untuk membandingkan DINOv3 7B, jadi angka
baseline di sini bisa langsung dibandingkan. Semua fit train-only. Metrik:
- silhouette cosine (PCA 128, KMeans K)
- stabilitas seed: ARI antar 5 seed KMeans
- stabilitas bootstrap: subsample 80%, KMeans, tetapkan semua titik ke centroid terdekat, ARI vs partisi acuan;
  plus Jaccard per cluster acuan (Hennig clusterboot), cluster dengan Jaccard < 0,6 dianggap rapuh
- kesepakatan kNN-10 dengan referensi independen: argmax head ImageNet-1k ConvNeXtV2 setelah normalisasi prior
  (tidak berasal dari embedding mana pun yang dibandingkan)
- kesepakatan dengan label lama: NMI dan kemurnian cluster (melingkar untuk ViT-L, karena label lama lahir di ruang ViT-L)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi, silhouette_score

from eda_common import KELAS, TEMP, emb, kelompok, out_dir

OUT = out_dir(__file__)


def siapkan(x, d=128, seed=42):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    z = PCA(d, random_state=seed).fit_transform(x)
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def km(z, k, seed):
    return KMeans(k, n_init=4, random_state=seed).fit(z)


def jaccard_per_cluster(ref, lab, idx):
    out = []
    for c in np.unique(ref):
        a = set(idx[ref[idx] == c])
        best = max((len(a & set(idx[lab[idx] == d])) / len(a | set(idx[lab[idx] == d])) for d in np.unique(lab[idx])), default=0)
        out.append(best)
    return np.array(out)


def metrik_ruang(x, k, ref8, lama, B=20, seed=42):
    z = siapkan(x, seed=seed)
    acuan = km(z, k, seed).labels_
    rng = np.random.default_rng(seed)
    seeds = [ari(acuan, km(z, k, seed + s).labels_) for s in range(1, 5)]
    boot, jac = [], []
    for b in range(B):
        idx = np.sort(rng.choice(len(z), int(.8 * len(z)), replace=False))
        m = km(z[idx], k, seed + 100 + b)
        lab = m.predict(z)
        boot.append(ari(acuan, lab))
        jac.append(jaccard_per_cluster(acuan, lab, idx))
    jac = np.mean(jac, 0)
    S = z @ z.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, 1)[:, :10]
    uk = np.bincount(acuan)
    pur = sum(pd.Series(lama[acuan == c]).value_counts().iloc[0] for c in range(k)) / len(lama)
    return dict(k=k, silhouette=silhouette_score(z, acuan, metric="cosine"), ari_seed=np.mean(seeds), ari_boot=np.mean(boot),
                ari_boot_std=np.std(boot), jaccard_med=np.median(jac), rapuh=int((jac < .6).sum()),
                knn_ref=np.mean(ref8[nn] == ref8[:, None]), nmi_lama=nmi(lama, acuan), murni_lama=pur,
                uk_min=uk.min(), uk_max=uk.max(), rasio_imb=uk.max() / uk.min()), acuan


if __name__ == "__main__":
    p = pd.read_csv(TEMP / "exp_005_imagenet_map/output/prob_train.csv")[KELAS].values
    ref8 = np.argmax(p / p.mean(0), 1)
    lama = kelompok(8).label_lama.values
    print("referensi IN1k (prior-norm) per kelas:", dict(zip(KELAS, np.bincount(ref8, minlength=8))))
    print("label lama per kelas:", pd.Series(lama).value_counts().to_dict(), f"rasio max/min {pd.Series(lama).value_counts().max() / pd.Series(lama).value_counts().min():.2f}")
    rows = []
    for nama in ("vit", "cnn"):
        x, _ = emb(nama)
        for k in (8, 16, 32, 48, 64):
            r, acuan = metrik_ruang(x, k, ref8, lama)
            r["ruang"] = nama
            rows.append(r)
            print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
            if nama == "vit" and k == 48:
                uk48 = np.sort(np.bincount(acuan))[::-1]
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "metrik_baseline.csv", index=False)
    print("\n", t.pivot(index="k", columns="ruang", values=["silhouette", "ari_boot", "jaccard_med", "rapuh", "knn_ref", "rasio_imb"]).round(3).to_string())

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for nama, mk in (("vit", "o-"), ("cnn", "s--")):
        x = t[t.ruang == nama]
        ax[0].plot(x.k, x.silhouette, mk, label=f"{nama} silhouette"); ax[0].plot(x.k, x.ari_boot, mk, label=f"{nama} ARI boot")
        ax[1].plot(x.k, x.knn_ref, mk, label=nama)
    ax[0].set(xlabel="K", title="silhouette dan stabilitas bootstrap"); ax[0].legend(fontsize=7)
    ax[1].set(xlabel="K", title="kNN-10 sepakat referensi IN1k"); ax[1].legend()
    ax[2].bar(range(len(uk48)), uk48); ax[2].set(title="ukuran cluster ViT-L K=48 (urut)", xlabel="cluster", ylabel="n")
    fig.tight_layout(); fig.savefig(OUT / "cluster_imbalance.png", dpi=120)
    print("simpan cluster_imbalance.png, metrik_baseline.csv di", OUT)
