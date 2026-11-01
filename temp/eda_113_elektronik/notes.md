# EDA 113: elektronik vs metal

**Konteks**: agen utama memberi hasil agregat tingkat kelas: 18 kesalahan test gabungan, 14 disepakati semua model, 7 di antaranya electronic dengan prediksi terutama metal. Informasi ini berasal dari temuan dan mengarahkan fokus, tetapi tidak masuk kode dan tanpa ID citra.

**Batasan kerja**: CPU saja, maksimal 4 thread. EoMT COCO dan ADE diizinkan koordinator sebagai bukti.

**Catatan data**: `vit_v2` dan `cnn_v2` dilatih pada label v2, bukan v4. Prediksi out-of-sample hanya ada untuk 388 hold-out, 83 sisa, dan test. Untuk fold 1 dipakai LR cross-fit sendiri (setuju v4 0,9864).

## Temuan utama: baterai

ImageNet-1k tidak punya kelas baterai, jadi label t1 awal tidak bisa melihatnya. Baterai berbentuk silinder logam dan jatuh ke metal.

| Median 3 model `sel` >= 0,2 | n | Kelas | Keluarga |
|---|---|---|---|
| Train | 14 | semua metal | web 10, F_640 4 |
| Test | 6 | semua prediksi metal | web 5, F_640 1 |

Pencemaran metal train per keluarga: F_lain_web 10 dari 22 (45%), F_640 4 dari 125 (3,2%), F_q74fb 0 dari 52, F_q75_4x3 0 dari 35, F_416 0 dari 7.
Test: F_lain_web 5 dari 5, F_640 1 dari 1, F_416 0 dari 58.

Ketegasan kosakata: `papan` 29 citra train semuanya electronic. `lampu` 13 citra: 9 kosong, 4 electronic. Jumlah di ambang 0,1 sampai 0,5 stabil pada test (6) dan train (13 sampai 15).

EoMT tidak membantu: COCO dan ADE tidak punya kata baterai. Median massa elektronik EoMT pada test metal rendah, dan citra baterai tidak menonjol.

## Koreksi laporan tahap 2

Tahap 2 menyebut sengketa metal-electronic "bisa jadi artefak". Ternyata terbagi dua:

- 18 metal dibantah in22k: 14 baterai, jadi in22k benar.
- 23 electronic dibantah in22k: lemma `iron` (setrika), `washer`, `clothes dryer`. Ini artefak peta metal, peralatan listrik memang electronic.

## Aturan beku (`label_v5.py`)

D = maks(median 3 model massa sel, lampu, papan). Bila D >= 0,20, label menjadi electronic. `kabel` tidak dipakai karena regex berisik.

Riwayat hash `label_v5.py`:
- Pertama 96b481d55bb8e65deeddcd541ea2f26e6e35cf708516fae10a6f4c179641e203 (17:05:17). Rumus berjalan, tetapi assert berhenti sebelum berkas tertulis karena dua baris v4 tanpa `suara`.
- Final b0635488356ebdd32865acad0d043c655faccc6dd636e9c9731a080f2b3dcda4 (17:05:37). Hanya assert yang berubah. Rumus dan konstanta sama.

`kosakata_listrik.py`: 30e0010f03425cc77a2849a0d8655de35e89f1a82a4b492815c47464aa1fc5d8.

## Hasil v5

| Kelas | v4 | v5 | Selisih |
|---|---|---|---|
| electronic | 186 | 209 | +23 |
| metal | 244 | 230 | -14 |
| kelas lain | tetap | tetap | 0 |

Total 1.843 ke 1.852. 14 metal menjadi electronic, 9 kosong menjadi electronic. Fold 0 tetap 384. Dua baris v4 tanpa group dan fam diisi (group 6, seluruhnya fold 1, tidak bocor). `suara` keduanya tetap kosong.

## Uji tanpa temuan (`uji_v5.py`)

- Recall leave-group-out baterai pada LR: 14 dari 14, median p_electronic 0,958 pada bobot 1. Oversampling tidak perlu.
- Test LR v4 ke v5: berubah 6, seluruhnya citra baterai, metal ke electronic.
- Jumlah test LR v5: glass 58, paper 67, electronic 60, metal 59, plastic 55, cardboard 58, textile 62, foodwaste 61.

## Batasan

- Baterai = electronic belum terverifikasi. Bukti: definisi komponen elektronik, keluarga F_640, dan kecocokan dengan laporan agregat. Bila host melabeli baterai sebagai metal, aturan ini merugikan sampai 6 citra.
- Jumlah test sempat dilihat sebelum aturan beku, hanya hitungan per kelas. Ambang berasal dari celah sebaran train, dan jumlah test sama pada ambang 0,1 sampai 0,5.
- Uji memakai LR, bukan fine-tune.
- Kesalahan ke-7 belum teridentifikasi. Citra metal F_416 tidak menunjukkan massa perangkat berarti.
- Perpindahan 9 lampu tidak berdampak ke test, karena test tidak punya citra lampu.
