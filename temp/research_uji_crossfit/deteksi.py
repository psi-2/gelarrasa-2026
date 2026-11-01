"""Detektor label salah beku: co-teaching ViT dan CNN dalam cross-fit berkelompok.

Fungsi ini dipanggil oleh uji7 (validasi injeksi) dan aturan_label.py (label asli), jadi kode yang divalidasi sama dengan yang dipakai.
Masukan hanya embedding dua backbone, label, dan group. Tidak ada temuan, test, fam, EoMT, 22k, atau DMS.

Aturan untuk tiap citra, dari R ulangan split kali S seed, total R x S model OOF per head:
- tolak_v, tolak_c: porsi model saat argmax OOF head berbeda dari label.
- alt: kelas selain label dengan rata-rata OOF ViT plus CNN tertinggi.
- KOREKSI ke alt: tolak_v >= TOLAK, tolak_c >= TOLAK, argmax rata-rata tiap head sama dengan alt, dan p(alt) >= YAKIN pada kedua head.
- BUANG: tolak_v >= TOLAK dan tolak_c >= TOLAK, tetapi bukan KOREKSI.
- TETAP: selain itu.

Konstanta dipilih dari uji2 dan uji6 pada noise sintetis, sebelum daftar label asli pertama dihasilkan.
"""
import numpy as np

from common import N_KELAS, lipat
from sim_label import co_teaching

LUPA = .2
R = 5
S = 3
N_FOLD = 5
TOLAK = .9
YAKIN = .9
SEED_DASAR = 2026


def oof_coteach(Ev, Ec, y, grp, lupa=LUPA, R=R, S=S, seed=SEED_DASAR):
    """Kembalikan array (R*S, n, 8) untuk tiap head. Tiap citra dinilai model yang tidak melatihnya, group terjaga."""
    n = len(y)
    a, b = np.zeros((R * S, n, N_KELAS), np.float32), np.zeros((R * S, n, N_KELAS), np.float32)
    for r in range(R):
        split = lipat(y, grp, N_FOLD, seed=seed + r)
        for s in range(S):
            for f, (tr, va) in enumerate(split):
                pa, pb = co_teaching(Ev[tr], Ec[tr], y[tr], Ev[va], Ec[va], lupa=lupa, seed=seed + 1000 * r + 100 * s + f)
                a[r * S + s, va], b[r * S + s, va] = pa, pb
    return a, b


def putuskan(a, b, y, tolak=TOLAK, yakin=YAKIN):
    """Dari OOF ke aksi. Kembalikan dict array per citra."""
    n = len(y)
    i = np.arange(n)
    tv, tc = (a.argmax(2) != y).mean(0), (b.argmax(2) != y).mean(0)
    pa, pb = a.mean(0), b.mean(0)
    g = (pa + pb) / 2
    m = g.copy()
    m[i, y] = -1
    alt = m.argmax(1)
    dua = (tv >= tolak) & (tc >= tolak)
    kor = dua & (pa.argmax(1) == alt) & (pb.argmax(1) == alt) & (pa[i, alt] >= yakin) & (pb[i, alt] >= yakin)
    bua = dua & ~kor
    return dict(koreksi=kor, buang=bua, alt=alt, tolak_v=tv, tolak_c=tc, pa=pa, pb=pb)
