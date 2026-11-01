# EDA 103: urutan ID dan SampleSubmission (Hipotesis B)

**Pertanyaan**: apakah ID berurutan membocorkan kategori?

**Metode**: `urutan.py` menghitung autokorelasi lag 1, 2, 3, 5, 10 pada cluster, nama, keluarga, ukuran, qhash, dan cos embedding, dengan uji permutasi 500 kali. `sample_sub.py` menguji struktur sample submission.

**Hasil**:

- Tidak ada kebocoran ID. |z| maksimum 2,44 dari 60 uji, sesuai harapan acak.
- Komposisi sumber dan nama stabil di 8 blok ID. Spearman ukuran berkas vs ID: -0,053 (train), 0,039 (test).
- Median jarak ID ke tetangga terdekat 585 (acak 640) di train, 130 (acak 160) di test.
- SampleSubmission: 60 per kelas, urutan acak, tidak periodik. Kesepakatan dengan prediksi 0,1375 vs baseline 0,1252, sekitar 0,8 sigma, jadi placeholder tanpa informasi.

**Kesimpulan**: ID sudah diacak. Tidak perlu fitur dan tidak perlu larangan khusus.
