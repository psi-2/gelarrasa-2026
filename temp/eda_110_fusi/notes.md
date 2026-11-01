# EDA 110: fusi kosakata untuk nama otomatis

**Tujuan**: mengganti penamaan cluster yang hanya mengandalkan ImageNet-1k.

**Metode**: `kosakata.py` memetakan nama kelas EoMT ke 8 kategori lewat arti nama kelas. `fusi.py` menggabungkan log-odds per cluster. `em.py` mempelajari emisi kata per kelas.

**Catatan kejujuran**:

1. Peta COCO ditulis SETELAH agen melihat profil COCO per cluster, jadi tidak sepenuhnya a priori. Peta ADE ditulis sebelum hasil ADE dilihat. Peta ini dibuat LLM dari nama kelas, sama sifatnya dengan `PETA` EXP-005, bukan penamaan cluster. Koordinator perlu memutuskan apakah ini dapat diterima.
2. `fusi.py` terakhir dijalankan lewat `Select-Object -First 8`, sehingga mungkin terhenti sebelum menyimpan. `em.py` berikutnya mungkin memakai `gab.npy` lama yang hanya COCO. Hasil EM di bawah harus dianggap sementara sampai dijalankan ulang.
3. `em.py` versi pertama transduktif: test ikut di M-step, prior, dan kNN. Ini tidak sesuai `docs/lock.md` (pseudo label test: tidak). Hanya sah sebagai diagnosis, bukan jalur label.

**Hasil sementara** (tanpa kendala keseimbangan):

| Metode | glass | paper | electronic | metal | plastic | cardboard | textile | foodwaste |
|---|---|---|---|---|---|---|---|---|
| EXP-006 | 42 | 38 | 54 | 67 | 106 | 59 | 57 | 57 |
| K=64 ConvNeXt + COCO | 45 | 60 | 64 | 64 | 59 | 56 | 52 | 80 |
| EM global (transduktif) | 52 | 58 | 51 | 78 | 50 | 62 | 61 | 68 |

Jumlah mendekati 60 per kelas bukan bukti akurasi. Skor hanya bisa diukur koordinator lewat eval temuan.

## Pembaruan setelah crash

- `em_train.py` train-only menggantikan `em.py`. Hasilnya TIDAK stabil: varian satu pandangan runtuh (metal 0 pada in1k dan ADE, paper 929 pada ADE), kesepakatan ketiga pandangan 37%, dan metal menjadi kelas residu dengan 0% dukungan minimal dua pandangan. EM dinyatakan gagal sebagai jalur label.
- `bagging.py` dan `varian.py`: nama cluster otomatis, 36 run KMeans (K 48, 64, 96 kali 12 seed), fit train-only, nama = argmax log-odds. Tanpa nomor cluster dan tanpa nama tertulis tangan di kode.

**Temuan inti**: `plastic` sebagai kelas penampung adalah bias prior massa. Porsi massa plastic pada ImageNet-1k 0,319, kelas lain 0,072 sampai 0,129.

| Varian (test) | glass | paper | electronic | metal | plastic | cardboard | textile | foodwaste | L1 ke 60 |
|---|---|---|---|---|---|---|---|---|---|
| mentah_in1k, tau 0 | 35 | 18 | 61 | 70 | 154 | 59 | 34 | 49 | 210 |
| t05_in1k, tau 0,5 | 46 | 36 | 61 | 78 | 94 | 59 | 53 | 53 | 106 |
| t1_in1k, tau 1 | 66 | 50 | 54 | 79 | 58 | 58 | 52 | 63 | 56 |
| t1_semua | 61 | 37 | 59 | 90 | 55 | 60 | 54 | 64 | 70 |
| t05_abst | 56 | 49 | 61 | 67 | 70 | 60 | 54 | 63 | 42 |
| t1_in1k_coco_abst | 58 | 55 | 63 | 69 | 54 | 59 | 55 | 67 | 38 |

abstain: kosakata dengan porsi kelas < 0,03 tidak bersuara untuk kelas itu (metal pada COCO 0,029).

Keseimbangan 60 per kelas tetap hipotesis, jadi L1 hanya petunjuk tanpa label, bukan skor.

## Berkas siap pakai (`label_final.py`)

Label train-only: porsi suara >= 0,75. Group: cos >= 0,9 atau pHash Hamming <= 4, terbesar 60. Hold-out StratifiedGroupKFold 5, fold 0, seed 42.

| Varian | Label | Hold-out | Test LR (glass, paper, electronic, metal, plastic, cardboard, textile, foodwaste) | Setuju EXP-006 |
|---|---|---|---|---|
| t1_in1k | 1.626 | 388 | 57, 67, 53, 64, 56, 59, 61, 63 | 0,8625 |
| t1_in1k_coco_abst | 1.610 | 273 | 51, 65, 60, 63, 54, 59, 61, 67 | 0,8646 |
| t05_abst | 1.610 | 353 | 48, 56, 62, 58, 70, 59, 61, 66 | 0,8646 |

Kesepakatan LR hold-out 0,994 sampai 0,996 itu melingkar: mengukur kesepakatan dengan label dari embedding yang sama, bukan akurasi.

Antar varian LR setuju 0,933 sampai 0,967. Vote dan LR setuju hanya sekitar 0,89, jadi yang direkomendasikan adalah LR atau training penuh, bukan vote.
