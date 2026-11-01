"""Metode pengurang kesalahan label pada dua pandangan embedding, dan generator noise sintetis.

Semua metode menerima indeks latih dan label latih yang boleh jadi kotor. Tidak ada metode yang melihat label asli
dari citra yang diinjeksi, dan tidak ada yang menyentuh test selain lewat predict.
"""
import numpy as np
import torch
from sklearn.cluster import KMeans

from common import N_KELAS, kandidat, lipat, lr_fit_predict, lr_lunak, satu_panas


def injeksi(y, V, E, jenis, laju, seed):
    """Kembalikan salinan label dengan noise. Sasaran noise: kelas kedua menurut suara, mirip kesalahan nama asli.

    acak: kelas lain acak per citra.
    bingung: per citra ke kelas kedua menurut suara bagging.
    cluster: seluruh isi cluster KMeans K=96 berpindah ke satu kelas lain, meniru salah nama cluster.
    """
    rng = np.random.default_rng(seed)
    y2 = y.copy()
    n = len(y)
    if jenis == "acak":
        pilih = rng.choice(n, int(laju * n), replace=False)
        y2[pilih] = (y[pilih] + rng.integers(1, N_KELAS, len(pilih))) % N_KELAS
    elif jenis == "bingung":
        pilih = rng.choice(n, int(laju * n), replace=False)
        W = V.copy()
        W[np.arange(n), y] = -1
        W = W + rng.random(W.shape) * 1e-6
        y2[pilih] = W.argmax(1)[pilih]
    elif jenis == "cluster":
        km = KMeans(96, n_init=1, random_state=seed).fit(E)
        W = np.stack([V[km.labels_ == c].mean(0) if (km.labels_ == c).any() else np.ones(N_KELAS) for c in range(96)])
        target = int(laju * n)
        for c in rng.permutation(96):
            if (y2 != y).sum() >= target:
                break
            m = km.labels_ == c
            if not m.any():
                continue
            top = np.bincount(y[m], minlength=N_KELAS).argmax()
            w = W[c].copy()
            w[top] = -1
            y2[m & (y == top)] = (w + rng.random(N_KELAS) * 1e-6).argmax()
    else:
        raise ValueError(jenis)
    return y2


def oof_dua(Ea, Eb, y, grp, C=3.0, n=5, seed=0):
    """Probabilitas OOF berkelompok pada dua pandangan."""
    pa, pb = np.zeros((len(y), N_KELAS)), np.zeros((len(y), N_KELAS))
    for tr, va in lipat(y, grp, n, seed):
        pa[va] = lr_fit_predict(Ea[tr], y[tr], [Ea[va]], C=C)[0]
        pb[va] = lr_fit_predict(Eb[tr], y[tr], [Eb[va]], C=C)[0]
    return pa, pb


def kesepakatan_silang(Ea, Eb, y, grp, ronde=3, ambang=.8, seed=0):
    """Pembaruan label lewat kesepakatan silang.

    Tiap ronde: OOF berkelompok di dua pandangan. Label diganti hanya bila kedua pandangan sepakat pada satu kelas
    lain dengan keyakinan >= ambang pada keduanya. Karena OOF, sebuah citra tidak bisa memilih dirinya sendiri.
    """
    cur = y.copy()
    riwayat = []
    for r in range(ronde):
        pa, pb = oof_dua(Ea, Eb, cur, grp, seed=seed + r)
        a, b = pa.argmax(1), pb.argmax(1)
        ganti = (a == b) & (a != cur) & (pa.max(1) >= ambang) & (pb.max(1) >= ambang)
        riwayat.append(int(ganti.sum()))
        if not ganti.any():
            break
        cur = cur.copy()
        cur[ganti] = a[ganti]
    return cur, riwayat


def bobot_dibuang(Ea, Eb, y, grp, ambang=.1, seed=0):
    """Bobot nol untuk citra yang label-nya ditolak kedua pandangan OOF."""
    pa, pb = oof_dua(Ea, Eb, y, grp, seed=seed)
    i = np.arange(len(y))
    return ~((pa[i, y] < ambang) & (pb[i, y] < ambang))


def lr_gabung_soft(Ea, Eb, T, Qa, Qb, C=3.0, bobot=None):
    """Satu soft target ke dua head, lalu rata-rata probabilitas."""
    pa = lr_lunak(Ea, T, [Qa], C=C, bobot=bobot)[0]
    pb = lr_lunak(Eb, T, [Qb], C=C, bobot=bobot)[0]
    return pa, pb


def co_teaching(Ea, Eb, y, Qa, Qb, lupa=.1, epoch=60, bs=128, lr=.05, wd=1e-4, seed=0, kira_k=10):
    """Co-teaching antar dua head linear atas embedding berbeda (Han dkk. 2018).

    Tiap minibatch, A memilih sampel loss kecil untuk melatih B, dan sebaliknya. Laju lupa naik linear sampai `lupa`
    pada kira_k epoch. Fine-tune penuh tidak diuji di sini, hanya head di atas embedding beku.
    """
    g = torch.Generator().manual_seed(seed)
    xa, xb = torch.as_tensor(Ea, dtype=torch.float32), torch.as_tensor(Eb, dtype=torch.float32)
    t = torch.as_tensor(y)
    m_a = torch.nn.Linear(xa.shape[1], N_KELAS)
    m_b = torch.nn.Linear(xb.shape[1], N_KELAS)
    opt = torch.optim.AdamW(list(m_a.parameters()) + list(m_b.parameters()), lr=lr, weight_decay=wd)
    n = len(y)
    langkah = max(1, n // bs) * epoch
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, langkah)
    for ep in range(epoch):
        buang = lupa * min(1.0, (ep + 1) / kira_k)
        urut = torch.randperm(n, generator=g)
        for i in range(0, n - bs + 1, bs):
            b = urut[i:i + bs]
            la, lb = m_a(xa[b]), m_b(xb[b])
            ra = torch.nn.functional.cross_entropy(la, t[b], reduction="none")
            rb = torch.nn.functional.cross_entropy(lb, t[b], reduction="none")
            k = int((1 - buang) * len(b))
            pa = torch.argsort(ra.detach())[:k]
            pb = torch.argsort(rb.detach())[:k]
            loss = ra[pb].mean() + rb[pa].mean()  # A dilatih pilihan B, B dilatih pilihan A
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
    with torch.no_grad():
        pa = torch.softmax(m_a(torch.as_tensor(Qa, dtype=torch.float32)), 1).numpy()
        pb = torch.softmax(m_b(torch.as_tensor(Qb, dtype=torch.float32)), 1).numpy()
    return pa, pb


def baseline_sgd(Ea, Eb, y, Qa, Qb, seed=0, **kw):
    """Co-teaching dengan laju lupa nol: pembanding yang adil, pengoptimal dan epoch sama."""
    return co_teaching(Ea, Eb, y, Qa, Qb, lupa=0.0, seed=seed, **kw)


__all__ = ["injeksi", "oof_dua", "kesepakatan_silang", "bobot_dibuang", "lr_gabung_soft", "co_teaching", "baseline_sgd",
           "kandidat", "satu_panas"]
