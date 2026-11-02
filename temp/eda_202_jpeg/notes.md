# EDA 202: format, kualitas JPEG, EXIF

Skrip `jpeg_exif.py`, gambar `output/jpeg_exif.png`.

| Temuan | Implikasi teori | Tindakan |
|---|---|---|
| Kualitas IJG ≤ 75: train 87,1%, test 93,5%. Satu qhash `e09fb31b26` mencakup 73,5% train dan 89,6% test | Hampir semua citra di-encode ulang oleh satu pipeline (ekspor Roboflow). Artefak blok 8×8 ada di train maupun test | Tidak perlu augmentasi JPEG khusus. Domain kompresi train dan test sudah sama |
| PNG mode P 1 dan RGBA 1 (train), PNG RGB 1 (test). Ekstensi campur | Loader yang mengasumsikan RGB atau `.jpg` akan gagal atau salah warna | `convert('RGB')`, cari berkas dari stem, urutkan secara numerik |
| EXIF hanya 1,3% train dan 2,3% test. Orientation ≠ 1 cuma 1 citra test | Rotasi EXIF hampir tidak berpengaruh, tetapi satu citra test akan tampil miring tanpa penanganan | `ImageOps.exif_transpose` saat load |
| qhash `2527d1e5ea` (q70) 9,2% train dan 0 test | Sumber khusus train, dan di label lama hampir semuanya textile (eda_102) | Hanya diagnostik. Cluster yang didominasi satu sumber perlu diperiksa di montage |
