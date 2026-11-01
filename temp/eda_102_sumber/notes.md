# EDA 102: sumber berkas vs cluster (Hipotesis A)

**Pertanyaan**: apakah sumber berkorelasi dengan cluster atau kategori?

**Peringatan**: `sumber.py` memberi nama cluster yang SALAH. Nomor cluster-nya berbeda dari run EXP-006 (kecocokan 3,7%), karena memakai `fit` lalu `transform`, bukan `fit_transform`. Kolom `nama` di output `sumber.csv`, `sumber_vs_nama.csv` dari skrip itu jangan dipakai. `assign.py` mereproduksi persis (ARI 1,0) dan `sumber2.py` adalah versi yang sah. `meta_assign.csv` berasal dari `sumber2.py`.

**Metode**: keluarga sumber `fam` dibentuk dari qhash dan ukuran. Cluster K=64 direproduksi, test dari pusat terdekat tanpa ikut fit.

**Hasil** (`meta_assign.csv`):

| fam | train | test | rasio test |
|---|---|---|---|
| F_416 | 579 | 240 | 0,293 |
| F_640 | 700 | 136 | 0,163 |
| F_lain_web | 173 | 49 | 0,221 |
| F_q2527_400x | 176 | 0 | 0 |
| F_q4885_besar | 70 | 0 | 0 |
| F_q74fb | 91 | 2 | 0,022 |
| F_q75_4x3 | 109 | 30 | 0,216 |
| F_q75_lain | 22 | 23 | 0,511 |

- Sumber sangat terkait kategori (chi2 p mendekati 0). q2527 berisi nama textile 173 dari 176. Cardboard 263 dari 270 di F_416. Electronic 257 dari 278 di F_640.
- Komposisi sumber per kelas bertukar antar split: metal train di F_640 (117) tetapi metal test di F_416 (54). Glass train di F_416 (65) tetapi test F_416 hanya 1.
- Textile test berasal dari web (27 + 4), bukan dari q2527 seperti train.
- Test F_416 tepat 240 citra, sama dengan 4 x 60.

**Tingkat keyakinan**: asosiasi sumber-cluster tinggi. "4 x 60" hanya hipotesis.

**KEBOCORAN**: ini kebocoran sumber (artefak akuisisi). Metadata berkas memberi informasi kategori. Bukan bocoran ID.
