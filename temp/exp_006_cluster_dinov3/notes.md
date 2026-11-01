# EXP-006: cluster embedding DINOv3 lalu train DINOv3 ViT-L

**Tujuan**: melatih DINOv3 dengan label dari cluster DINOv3, nama cluster dari massa kategori ConvNeXtV2 (EXP-005).

**Metode**:

1. Embedding DINOv3 ViT-L 384 px untuk train dan test. Test tidak ikut clustering.
2. PCA 128 lalu KMeans K=64. K dipilih dari silhouette 0,2695 dengan 42 cluster bersih pada `porsi >= 0,6`.
3. Nama cluster: kategori dengan massa ConvNeXtV2 terbesar. Syarat: `porsi >= 0,4` dan selisih dari kelas kedua `>= 0,07`, khusus `plastic` `porsi >= 0,6`.
4. Ambang tunggal 0,6 ditolak karena `metal` hanya dapat 31 citra. Cluster kaleng tercampur `plastic` tetapi `metal` tetap teratas.
5. Hold-out 20% StratifiedGroupKFold, group DINOv3 cos >= 0,9. Latih 1.263, hold-out 249.

**Label**: 1.512 dari 1.920 citra, 50 dari 64 cluster. glass 109, paper 92, electronic 199, metal 180, plastic 385, cardboard 214, textile 175, foodwaste 158.

**Hasil hold-out**: 0,9558 dengan label cluster, puncak 0,9598 di epoch 5. Per kelas: glass 0,95, paper 0,88, electronic 0,96, metal 0,87, plastic 0,98, cardboard 1,00, textile 1,00, foodwaste 0,93. Ini kesepakatan dengan label cluster, bukan akurasi asli.

**Prediksi test**: glass 42, paper 38, electronic 54, metal 67, plastic 106, cardboard 59, textile 57, foodwaste 57. Setuju ConvNeXtV2 73,96%. Skor temuan ada di EXP-007: macro F1 0,8574, accuracy 0,8542.

**Hardware**: 800 detik, reserved 10,91 GiB, tanpa baris bocor di `output/holdout/vram.csv`.

**Reproducibility**: uji dua kali tidak dijalankan, jadi pipeline belum terverifikasi deterministik dan belum boleh dipromosikan.

**Catatan kebersihan NO VLM**: label, K, dan model tidak memakai keluaran VLM. Namun ambang khusus `plastic` ditetapkan setelah diagnosis yang membandingkan EXP-005 dengan model VLM (sudah dihapus), jadi keputusan itu tidak sepenuhnya bebas pengaruh. Alasan teknisnya tetap berlaku: `packet` dan `tray` ambigu dari namanya.
