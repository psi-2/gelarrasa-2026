# EXP-007: skor temuan

**Tujuan**: menghitung skor prediksi test terhadap `data/temuan/anotate.csv`, yang diasumsikan user sebagai label test yang benar. Keaslian file ini belum terverifikasi.

**Validasi file**: 480 baris, ID unik dan cocok dengan test, 8 kelas valid, tanpa nilai kosong. Distribusi nyaris seimbang: 59 sampai 61 per kelas. `image_id` memakai ekstensi `.jpg` dan dinormalisasi ke stem saat dibaca.

**Hasil** (CI 95% dari bootstrap 2.000 kali):

| Prediksi | Macro F1 | CI 95% | Accuracy | Benar |
|---|---|---|---|---|
| EXP-009 DINOv3 label t1 | 0,9543 | 0,9343 sampai 0,9725 | 0,9542 | 458 |
| t1_in1k_lr | 0,9146 | 0,8886 sampai 0,9385 | 0,9146 | 439 |
| t1_coco_abst_lr | 0,8967 | 0,8675 sampai 0,9223 | 0,8979 | 431 |
| t05_abst_lr | 0,8900 | 0,8615 sampai 0,9169 | 0,8917 | 428 |
| EXP-006 DINOv3 label cluster | 0,8574 | 0,8254 sampai 0,8855 | 0,8542 | 410 |
| EXP-005 ConvNeXtV2 pemetaan | 0,7113 | 0,6697 sampai 0,7491 | 0,7104 | 341 |

**Batasan**: skor ini hanya untuk evaluasi. Temuan tidak boleh dipakai untuk menamai cluster, memilih K, atau menyetel model. Enam prediksi non-VLM sudah dinilai di sini, dan penilaian dihentikan setelah target tercapai untuk membatasi overfit ke temuan.
