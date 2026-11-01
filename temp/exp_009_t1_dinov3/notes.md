# EXP-009: DINOv3 ViT-L pada label t1_in1k

**Tujuan**: mencapai accuracy temuan di atas 0,95 tanpa VLM dan tanpa label manual.

**Label** (`temp/eda_110_fusi/label_final.py`, otomatis dan train-only):

1. Sumber: massa kategori ConvNeXtV2 ImageNet-1k dengan `PETA` EXP-005.
2. Normalisasi prior `tau = 1`: massa dibagi porsi massa kelas di train. Ini menghapus bias `plastic` (0,319 massa, kelas lain 0,072 sampai 0,129).
3. KMeans di-bagging 36 run (K 48, 64, 96 kali 12 seed) pada embedding DINOv3 train. Nama cluster keluar dari argmax log-odds, tanpa nomor cluster dan tanpa nama tertulis tangan.
4. Label dipakai bila porsi suara >= 0,75: 1.626 dari 1.920 citra, 294 dibiarkan.
5. Group: cos DINOv3 >= 0,9 atau pHash Hamming <= 4. Hold-out StratifiedGroupKFold fold 0: latih 1.238, hold-out 388.

**Model**: `train_dino.py` sama dengan EXP-006. 384 px, 8 epoch, batch 8 accumulation 2, layer decay 0,75, label smoothing 0,1, bobot kelas invers akar frekuensi, flip TTA. 886 detik, reserved 10,9 GiB, tanpa bocor.

**Hasil**:

| Ukuran | Nilai |
|---|---|
| Accuracy temuan | 0,9542 (458 dari 480) |
| Macro F1 temuan | 0,9543, CI 95% 0,9343 sampai 0,9725 |
| Kesepakatan hold-out dengan label | 0,9871, melingkar, bukan akurasi |

F1 per kelas: glass 0,9492, paper 0,9431, electronic 0,9381, metal 0,9120, plastic 0,9661, cardboard 0,9672, textile 0,9752, foodwaste 0,9833.

**Batasan**:

- 458 hanya 2 citra di atas batas 0,95 (456). Batas bawah CI ada di bawah 0,95, jadi selisihnya belum aman secara statistik.
- Kandidat t1 dipilih dari tiga varian lewat skor temuan, jadi skor ini sedikit optimistis.
- Temuan belum terverifikasi. Bila tidak sama dengan label Kaggle, skor asli bisa berbeda.
- Diagnosis `plastic` sebagai penampung ikut dipengaruhi skor temuan EXP-006, tetapi kode label tidak membaca temuan.
- Uji determinism tidak dijalankan, jadi pipeline belum boleh dipromosikan ke `scripts/`.
- Tanpa VLM: ConvNeXtV2 fcmae dan DINOv3 LVD-1689M hanya menerima citra. EoMT tidak dipakai pada varian ini.
