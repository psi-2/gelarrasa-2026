# EDA 112: label train v2 dari bukti independen

**Tujuan**: memperkuat label, target akhir minimal 471 dari 480. Classifier ImageNet-22k dan DMS46 hanya sumber label di tahap ini, bukan model prediksi.

**Urutan kepatuhan**: `peta.py --beku` dijalankan SEBELUM inferensi. Skrip gagal bila berkas `p22k_*` atau `dms_*` sudah ada. Hash ada di `output/peta_beku.json`.

**Peta ImageNet-22k**: subset lewat `infer_imagenet_subset` (ketiga model `imagenet-22k`). Rumus: gerbang rentang synset, regex lemma, lalu regex material pada gloss, multi-kategori dibagi rata. `bottle` dan `cup` generik tidak dipetakan karena materialnya tidak ditentukan nama. Gloss resmi ikut dipakai, jadi peta memakai nama dan definisi resmi.

**Peta DMS46**: Cardboard, Carpet/rug, Fabric/cloth, Food, Glass, Metal, Paper, Plastic clear, Plastic non-clear. Electronic tidak dipetakan.

**Tinjauan pra-beku**: hanya daftar nama. Satu perbaikan: kata `printed` dibuang dari gloss paper karena menarik kain. Perbaikan terjadi sebelum inferensi ada.

**GPU**: `jalankan_gpu.py` membungkus skrip asli dengan batas 5 GiB dan 3 worker. EoMT tidak dipakai di jalur ini karena izinnya belum dijawab.

**Status**: selesai. VRAM puncak 2,07 GiB (22k) dan 0,58 GiB (DMS). Peta beku 2026-10-31 15:36:22, hash cocok saat dipakai.

## Validasi pipeline

- `in1k` mereproduksi `t1_in1k` persis, selisih maksimum 0,0.
- Top-1 antar model 22k setuju 0,23 sampai 0,44, jadi urutan indeks selaras.

## Kesepakatan (train berlabel 1.626)

| Pembanding | Setuju label lama |
|---|---|
| b_in22k | 0,9084 |
| b_dms | 0,7540 (electronic tidak bisa cocok) |
| b_in22k_dms | 0,9692 |
| c_in1k | 0,8143 |
| c_in22k | 0,8020 |
| c_dms | 0,5504 |

Vote `b_` berbagi cluster DINOv3 yang sama, jadi nilainya lebih tinggi dari independensi sebenarnya. Argmax per citra `c_` lebih jujur.

Bias sumber: `c_in22k` membaca 100 dari 221 plastic sebagai foodwaste. `c_dms` membaca 96 dari 179 electronic sebagai plastic.

## Aturan beku (`label_v2.py`, sha256 239142f667e2615f3d557036f99306c19887a38fb4fb43659645e23c7e80266b)

Dawid-Skene train-only dengan annotator c_in1k, c_in22k, c_dms, b_in22k, b_dms bobot 0,5 dan e_lr bobot 1. Q_MIN 0,90, DUKUNG_MIN 2 kanal. Dijalankan sekali.

| Keputusan | Jumlah |
|---|---|
| tetap | 1.623 |
| tambah | 211 |
| koreksi | 3 |
| kosong | 83 |

Total v2 1.837. Fold 0 tetap 388 citra yang sama, 211 tambahan semuanya fold 1. `suara`: nilai lama untuk label tetap, posterior DS untuk tambah dan koreksi.

Koreksi: 0550 glass ke paper, 0680 metal ke foodwaste, 1920 paper ke cardboard. EXIF ketiganya kosong, jadi tidak ada bukti tambahan.

## Batas label

| Kelas | n | A harapan | B harapan | B q<.5 | A dan B q<.5 |
|---|---|---|---|---|---|
| glass | 206 | 6,4 | 35,6 | 32 | 6 |
| paper | 239 | 2,1 | 17,9 | 8 | 1 |
| electronic | 186 | 0,1 | 55,0 | 23 | 0 |
| metal | 240 | 4,9 | 34,5 | 24 | 4 |
| plastic | 277 | 2,5 | 41,4 | 16 | 1 |
| cardboard | 227 | 1,1 | 12,2 | 13 | 1 |
| textile | 218 | 1,0 | 6,7 | 6 | 0 |
| foodwaste | 244 | 0,2 | 27,5 | 20 | 0 |
| Total | 1.837 | 18,3 | 230,8 | 142 | 13 |

A memakai e_lr dan mewarisi label lama, jadi batas bawah. B tanpa e_lr mewarisi bias kosakata dan kelemahan DMS pada electronic, jadi batas atas. Rentang 18 sampai 231 belum memberi titik estimasi yang sah.

Pemeriksaan hold-out: B q<.9 menandai 108 citra dan memuat 4 dari 5 ketidaksetujuan EXP-009. B q<.5 hanya 1 dari 5. EXP-009 dilatih dari label sejenis, jadi ini bukan bukti kebenaran label.

Metal-electronic belum terselesaikan: 23 electronic dan 14 metal dibantah in22k, tetapi didukung in1k dan e_lr. Kata `metal` di peta in22k luas (362 kelas), jadi bisa artefak kosakata. Peta tidak diubah.

KOREKSI dari eda_113: dugaan artefak hanya benar untuk 23 electronic (setrika, mesin cuci, pengering). 14 metal yang dibantah adalah baterai dan in22k benar. Label v2 untuk 14 citra itu salah.

## Batasan

- DS sangat terlalu yakin: q<.5 16 vs q<.9 18.
- Test hanya pembandingan, tidak ada berkas prediksi test dari DS, karena 22k dan DMS bukan model prediksi.
- Hold-out tidak memuat tambahan, jadi tidak mengukur kualitas 211 citra itu.
