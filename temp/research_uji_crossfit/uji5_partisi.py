"""Uji 5: ensemble partisi. Apakah KMeans bagging di ruang CNN atau gabungan memperbaiki label t1_in1k?

Aturan nama sama persis dengan temp/eda_110_fusi/varian.py: tau = 1, argmax log-odds massa ConvNeXtV2 per cluster.
Test tidak ikut. Blok seed independen dipakai untuk mengukur stabilitas antar ensemble.

Varian: vit (replika t1), cnn, gabung (run ViT + run CNN), consensus (agglomerative atas co-association ViT + CNN).

    python temp/research_uji_crossfit/uji5_partisi.py
"""
from common import *  # noqa: F401,F403
from common import F110, NAMES, OUT, Waktu, bulat, fitur_pandangan, lr_fit_predict, muat, muat_cnn, pandangan_eomt

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

log = Waktu()
d = muat()
Ev, y, Vt = d["E"], d["y"], d["V"]
Ec, _ = muat_cnn(d["ids"], d["ids_te"])
nt = 1920
S = np.load(F110 / "skor_conv.npy")[:nt].astype(np.float64)
v = S / (S.mean(0) + 1e-6)
Xe, _ = pandangan_eomt()


def z(E):
    p = PCA(128, random_state=42).fit_transform(E)
    return p / np.linalg.norm(p, axis=1, keepdims=True)


Zv, Zc = z(Ev), z(Ec)


def nama(lab, k):
    r = np.stack([v[lab == c].mean(0) if (lab == c).any() else np.ones(8) for c in range(k)])
    r = r / r.sum(1, keepdims=True)
    return np.log(r + 1e-3).argmax(1)


def run(Z, seeds, Ks=(48, 64, 96)):
    H = []
    for k in Ks:
        for s in seeds:
            H.append(KMeans(k, n_init=1, random_state=1000 * k + s).fit(Z).labels_)
    return H


def suara(H):
    V = np.zeros((nt, 8))
    for lab in H:
        k = lab.max() + 1
        V[np.arange(nt), nama(lab, k)[lab]] += 1
    return V / len(H)


def co_assoc(H):
    C = np.zeros((nt, nt), dtype=np.float32)
    for lab in H:
        o = np.eye(lab.max() + 1, dtype=np.float32)[lab]
        C += o @ o.T
    return C / len(H)


blok = {"A": range(12), "B": range(100, 112)}
Hv = {b: run(Zv, s) for b, s in blok.items()}
log("run ViT selesai")
Hc = {b: run(Zc, s) for b, s in blok.items()}
log("run CNN selesai")

varian = {}
for b in blok:
    varian[f"vit_{b}"] = suara(Hv[b])
    varian[f"cnn_{b}"] = suara(Hc[b])
    varian[f"gabung_{b}"] = suara(Hv[b] + Hc[b])
    C = (co_assoc(Hv[b]) + co_assoc(Hc[b])) / 2
    for K in (64, 96):
        lab = AgglomerativeClustering(K, metric="precomputed", linkage="average").fit_predict(1 - C)
        varian[f"cons{K}_{b}"] = np.eye(8)[nama(lab, K)[lab]]
    log(f"blok {b} selesai")

# Pembanding independen: pandangan EoMT dilatih cross-fit di luar sini akan melingkar, jadi dipakai sebagai skor per citra.
il = np.where(y >= 0)[0]
Fe, [Fq] = fitur_pandangan(Xe[il], [Xe])
pe = lr_fit_predict(Fe, y[il], [Fq], C=.03)[0].argmax(1)
ft = np.full(nt, -1)
ft[y < 0] = d["sisa"].argmax(1)

rows = []
t1 = Vt.argmax(1)
for nama_v, V in varian.items():
    u = V.argmax(1)
    pakai = V.max(1) >= .75
    rows.append({"varian": nama_v, "label": int(pakai.sum()), "setuju_t1_pada_pakai": bulat((u[pakai] == t1[pakai]).mean()),
                 "setuju_label_exp009": bulat((u[y >= 0] == y[y >= 0]).mean()),
                 "baru_vs_t1": int((pakai & (y < 0)).sum()), "baru_setuju_finetune": bulat((u[pakai & (y < 0)] == ft[pakai & (y < 0)]).mean() if (pakai & (y < 0)).any() else 0),
                 "baru_setuju_eomt": bulat((u[pakai & (y < 0)] == pe[pakai & (y < 0)]).mean() if (pakai & (y < 0)).any() else 0),
                 **{n: int((u[pakai] == c).sum()) for c, n in enumerate(NAMES)}})
t = pd.DataFrame(rows)
print(t.to_string(index=False))
t.to_csv(OUT / "uji5_varian.csv", index=False)

st = []
for m in ("vit", "cnn", "gabung", "cons64", "cons96"):
    a, b = varian[f"{m}_A"], varian[f"{m}_B"]
    ua, ub = a.argmax(1), b.argmax(1)
    pa, pb = a.max(1) >= .75, b.max(1) >= .75
    st.append({"metode": m, "ari_antar_blok": bulat(adjusted_rand_score(ua, ub)), "setuju_antar_blok": bulat((ua == ub).mean()),
               "himpunan_label_sama": bulat((pa == pb).mean()), "setuju_pada_keduanya": bulat((ua[pa & pb] == ub[pa & pb]).mean())})
ts = pd.DataFrame(st)
print(ts.to_string(index=False))
ts.to_csv(OUT / "uji5_stabil.csv", index=False)
np.save(OUT / "uji5_gabung_A.npy", varian["gabung_A"].astype(np.float32))
log("selesai")
