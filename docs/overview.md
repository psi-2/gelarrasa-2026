# Overview: Data Science Competition (DSC) Gelar Rasa 2026

Sumber: teks halaman Kaggle yang ditempel user (2026-10-31), ditambah metadata dari Kaggle CLI.

- **Link**: https://www.kaggle.com/competitions/dsc-gelar-rasa-2026
- **Jenis**: Community competition, preliminary round
- **Tugas**: image classification 8 kelas pada citra sampah
- **Kelas (string persis)**: `glass`, `paper`, `electronic`, `metal`, `plastic`, `cardboard`, `textile`, `foodwaste`
- **Metric**: Accuracy (lebih besar lebih baik)
- **Submission**: maksimal 3 kali per hari

## Code competition?

Tidak ada indikasi code competition. Teks halaman hanya meminta file CSV, tanpa batas waktu run notebook atau syarat internet mati. Status ini belum dikonfirmasi dari halaman Kaggle, tetapi format submission-nya CSV biasa.

## Timeline

| Item | Waktu | Sumber |
|---|---|---|
| Data diunggah | 2026-10-29 13:35 UTC | `kaggle competitions files` |
| Mulai | sekitar 2026-10-30/31 ("12 hours ago" per 2026-10-31) | halaman Kaggle |
| Tutup | 2026-11-14 04:59 UTC (11:59 WIB) | `kaggle competitions list` |

Kaggle CLI menampilkan deadline dalam UTC. Batas pengumpulan komponen proyek lain (rule 13) belum diketahui.

## Deskripsi

Pengelolaan sampah masih menjadi masalah lingkungan: volume yang terus naik tanpa pengelolaan tepat mencemari tanah dan air, merusak ekosistem, menyumbat drainase, dan mengancam kesehatan. Sampah tercampur juga menyulitkan daur ulang. Pemilahan manual lambat, melelahkan, dan rawan keliru karena sampah sangat beragam dalam bentuk, warna, tekstur, dan kondisi, sehingga Computer Vision dimanfaatkan untuk mengenali dan mengklasifikasikan sampah dari citra.

Pada preliminary round peserta membangun model image classification untuk delapan kategori sampah. Dataset sengaja memuat kondisi dunia nyata: noise pada citra, latar belakang beragam, dan class imbalance antar kategori.

Peserta perlu menerapkan tahapan data science menyeluruh (eksplorasi data, preprocessing, pemilihan metode, evaluasi per kategori). Fokus kompetisi bukan hanya hasil prediksi, tetapi juga pendekatan dan pemahaman dalam mengolah data.

## Tujuan

1. Melakukan preprocessing dan data cleaning pada citra.
2. Membangun model klasifikasi berbasis Computer Vision.
3. Memprediksi kategori sampah pada data yang diberikan.
4. Menganalisis hasil model dan menghasilkan insight berdasarkan pola pada data.

## Evaluasi

Submission dievaluasi dengan Accuracy, yaitu proporsi prediksi benar terhadap seluruh data yang dievaluasi.

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

- TP: data positif yang diprediksi benar
- TN: data negatif yang diprediksi benar
- FP: data negatif yang salah diprediksi positif
- FN: data positif yang salah diprediksi negatif

Rumus pada halaman Kaggle tidak ikut tertempel. Rumus di atas diturunkan dari definisi TP/TN/FP/FN yang tertempel. Untuk klasifikasi 8 kelas ini setara dengan jumlah prediksi benar dibagi 480 pada seluruh test (porsi public/private leaderboard belum diketahui).

## Format submission

CSV dua kolom, 480 baris data:

```
image_id,waste_category
001,glass
002,plastic
003,metal
004,paper
005,textile
006,foodwaste
007,cardboard
008,electronic
```

`image_id` adalah string 3 digit dengan nol di depan. Baca dan tulis sebagai string, jangan sebagai integer.
