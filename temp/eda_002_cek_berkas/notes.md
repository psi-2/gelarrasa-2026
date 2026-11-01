# EXP-002: cek kelengkapan berkas

**Tujuan**: memastikan jumlah, ID, format citra, dan sumber label sebelum research.

**Hasil** (`output/cek_berkas.json`):

- Train 1.920 dan test 480 citra, ID tanpa celah, tidak ada stem ganda, tidak ada citra gagal dibuka.
- ID sample submission cocok persis dengan 480 berkas test.
- Tidak ada berkas label. Satu-satunya berkas non-citra adalah `SampleSubmission_DSC.csv`. Kaggle memuat 2.401 berkas dengan isi yang sama, jadi ini bukan kesalahan unduh.
- Ekstensi campur (`.jpg`, `.jpeg`, `.png`), 2 citra train adalah PNG berekstensi `.jpg`/`.jpeg`.
- Median ukuran test 416x416, train 509x485,5 dengan rentang lebih lebar.

**Kesimpulan**: data citra utuh, tetapi supervised training terhambat sampai label train jelas. Script hanya membaca header dan metadata citra.
