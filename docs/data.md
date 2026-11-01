# Data

Sumber: deskripsi dataset yang ditempel user (2026-10-31), hasil `temp/eda_002_cek_berkas/cek_berkas.py`, dan daftar berkas `kaggle competitions files dsc-gelar-rasa-2026`.

## MASALAH TERBUKA: label train tidak tersedia

Tidak ada label untuk 1.920 citra train, baik di folder lokal maupun di Kaggle.

- Kaggle memuat tepat 2.401 berkas: 1.920 train, 480 test, dan `SampleSubmission_DSC.csv`.
- `train/` datar tanpa subfolder kelas. Nama berkas hanya nomor (`0001.jpg` sampai `1920.jpg`).
- Tidak ada `train.csv` atau berkas anotasi apa pun.

Supervised training belum bisa dimulai sampai ada kejelasan. Jalur yang benar: tanyakan ke panitia. Claude tidak boleh melihat citra untuk memberi label (lihat Catatan kepatuhan di `docs/rule.md`).

## Berkas

| Path | Isi | Jumlah |
|---|---|---|
| `data/train/` | citra train tanpa label | 1.920 |
| `data/test/` | citra test | 480 |
| `data/SampleSubmission_DSC.csv` | contoh submission | 480 baris |

Total ukuran citra 93,5 MB. Di repo tujuan, gambar berada langsung di data/train/ dan data/test/.

## Kolom submission

| Kolom | Tipe | Arti |
|---|---|---|
| `image_id` | string 3 digit (`001` sampai `480`) | nama berkas test tanpa ekstensi |
| `waste_category` | string | salah satu dari 8 kelas |

ID pada sample submission cocok persis dengan 480 berkas test.

## Kelas

1. `glass`: limbah kaca, seperti botol, wadah, atau pecahan kaca.
2. `paper`: limbah kertas, seperti lembaran kertas, koran, dan bahan berbasis kertas lainnya.
3. `electronic`: limbah peralatan atau komponen elektronik yang tidak digunakan lagi.
4. `metal`: limbah logam, seperti kaleng, wadah logam, dan benda logam lainnya.
5. `plastic`: limbah plastik, seperti botol, kemasan, dan wadah plastik.
6. `cardboard`: limbah kardus, seperti kotak dan kemasan berbahan cardboard.
7. `textile`: limbah tekstil, seperti pakaian, kain, dan material tekstil lainnya.
8. `foodwaste`: sisa makanan atau bahan pangan yang tidak digunakan lagi.

## Karakteristik yang sudah terukur

| Item | Train | Test |
|---|---|---|
| Jumlah | 1.920 | 480 |
| Panjang nama | 4 digit | 3 digit |
| ID | 1 sampai 1920, tanpa celah | 1 sampai 480, tanpa celah |
| Ekstensi | `.jpg` 1.816, `.jpeg` 104 | `.jpg` 477, `.jpeg` 2, `.png` 1 |
| Format asli | JPEG/RGB 1.918, PNG/P 1, PNG/RGBA 1 | JPEG/RGB 479, PNG/RGB 1 |
| Lebar (min / median / max) | 100 / 509 / 2.605 | 129 / 416 / 1.600 |
| Tinggi (min / median / max) | 100 / 485,5 / 4.032 | 168 / 416 / 1.167 |
| Gagal dibuka | 0 | 0 |

Catatan untuk pipeline:

- Ekstensi campur, jadi cari berkas berdasarkan stem, jangan menyusun `f"{id}.jpg"`.
- Dua citra train sebenarnya PNG walau berekstensi `.jpg`/`.jpeg`. Konversi ke RGB saat load karena ada mode P dan RGBA.
- Median test 416 dengan kuartil atas 640, sedangkan train lebih bervariasi (median 509 sampai maksimum 4.032). Ukuran train dan test berbeda distribusinya, perlu dicek saat research.
- Sample submission berisi tepat 60 per kelas. Ini diperlakukan sebagai placeholder, bukan label.

## Catatan dari host

Dataset memuat noise, latar belakang beragam, dan class imbalance antar kelas.
