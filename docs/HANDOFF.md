# Handoff eksperimen — DSC Gelar Rasa 2026

## Sumber kebenaran

Baca [constraint.md](constraint.md) dan [context.md](context.md) lebih dulu. Batasan terbaru berlaku atas semua catatan lama: submission satu model, tanpa ensemble/blending, tanpa memakai data test untuk training/seleksi, tanpa VLM, dan keputusan model berdasarkan validasi lokal.

Data sudah ada di repo pada `data/train/` (1.920 gambar), `data/test/` (480 gambar), dan `data/SampleSubmission_DSC.csv`. Struktur ini berbeda dari folder sumber lama yang memakai subfolder dataset bernama `Dataset DSC GelarRasa 2026`.

## Status dan hasil historis

- [log_exp.md](../log_exp.md) dan `temp/` menyimpan riwayat eksperimen, skrip, catatan, tabel hasil, dan notebook.
- EXP-009: DINOv3 ViT-L dengan label cluster otomatis; catatan lama melaporkan hold-out 0,9871. Itu mengukur kesepakatan dengan label pseudo, bukan akurasi ground truth.
- EXP-010: eksperimen CNN/ViT. Gunakan CNN saja bila melanjutkan sebagai kandidat satu model; validasi ulang dengan split lokal yang sah.
- EXP-011 dan notebook `temp/nb_final/` melaporkan hasil ensemble 469/480. Ini arsip historis dan melanggar larangan ensemble pada constraint terbaru; jangan pakai sebagai submission.
- Sejumlah skor historis dihitung terhadap anotasi test lokal yang tidak terverifikasi. Anotasi tersebut sengaja tidak disalin ke repo; jangan gunakan skor itu untuk memilih model atau menyetel parameter.

## Cache penting yang disertakan

- `temp/exp_006_cluster_dinov3/output/emb_train.npy`, `emb_test.npy`, `ids_train.npy`, `ids_test.npy`: embedding DINOv3 beserta pemetaan ID.
- `temp/nb_final/output/cache/emb_vit.npy`, `emb_cnn.npy`, ID train/test, pHash, dan keluaran kecil head v5/v6.
- Cache label/fusi kecil dan fitur ImageNet-1k yang dipakai ulang.

Cache ImageNet-22k sekitar 600 MB, token bag-of-words sekitar 169 MB, dan checkpoint CNN sekitar 749 MB tidak disalin. Ketiganya tidak diperlukan untuk melanjutkan kandidat satu model dan akan membuat push membengkak; artefak tersebut dapat dibuat ulang bila eksperimen lama memang dibutuhkan.

## Menjalankan di perangkat lain

1. Buat environment Python/GPU dari [environment.md](environment.md); path dan daftar cache di sana mencatat komputer sumber, bukan konfigurasi universal.
2. Skrip Python yang memakai data kini menunjuk ke `<repo>/data`. Model pretrained perlu tersedia di cache lokal atau diunduh oleh loader. Atur cache lewat variabel lingkungan NGEDATA_CACHE; default memakai D:\cache_moved bila tersedia, lalu folder .cache pengguna.
3. Gunakan skrip dan label di `temp/exp_010_dua_backbone/` sebagai titik lanjut kandidat CNN satu model. Baca catatan eksperimen sebelum menjalankan; beberapa arsip lama tidak sesuai constraint terbaru.

File `data/temuan/anotate.csv` tidak dimasukkan karena memuat anotasi test. Snapshot berbagi progres 3 November 2026 menambahkan referensi, hasil run Kaggle, serta arsip cache/environment; lihat [README](../README.md) dan [petunjuk arsip](../archives/README.md).
