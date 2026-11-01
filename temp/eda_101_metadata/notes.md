# EDA 101: sidik jari berkas (Hipotesis A)

**Pertanyaan**: apakah data gabungan banyak sumber?

**Metode**: `scan_meta.py` mem-parsing marker JPEG, tabel kuantisasi (qhash), EXIF, ukuran. `rangkum_meta.py` membuat tabel frekuensi. Citra tidak ditampilkan. Keluaran `output/meta.csv` (2.400 baris).

**Hasil**:

- Multi-sumber, terbukti. 199 ukuran unik dan 206 qhash unik.
- Dua ukuran dominan: 640x640 (train 700, test 136) dan 416x416 (train 579, test 240), mengisyaratkan ekspor Roboflow.
- 1.839 citra memakai satu qhash `e09fb31b26` (q75 IJG), artinya ditulis ulang oleh satu encoder.
- 512x384 (44 citra) cocok dengan dimensi TrashNet. Ini dugaan dari dimensi, tidak diverifikasi.
- Jejak web: EXIF Dreamstime 11 citra, komentar eBay ImageMagick 1, gd-jpeg 1. Kamera: RaspberryPi RP_ov5647 5 citra, iPhone 11 Pro, Kodak, realme C55.
- Sumber khusus train: 400x533 (74 vs 0), 1599x1599 (57 vs 0), 320x320 (56 vs 0). qhash `2527d1e5ea` 176 train, 0 test. `4885dc9d7c` 70 train, 0 test.
- SHA1 identik: 8 citra.

**Kesimpulan**: test bukan sampel acak dari distribusi train. Lanjut ke eda_102.
