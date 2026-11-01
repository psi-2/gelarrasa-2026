# Notebook final

**Sumber**: rantai EXP-005, EXP-006, EXP-009, label v2 sampai v6, dan `train_dua.py`. SHA256 sumber ada di metadata notebook.

**Build**: `build_final.py`, 85 cell atau lebih, tiga banner, Introduction, Initialization, Table of Contents.

**Uji**:

| Uji | Hasil |
|---|---|
| Struktur dan sintaks semua cell | lulus |
| Smoke test subset (640 train, 32 test, 1 epoch) | lulus, 1.015 detik |
| Kesamaan label notebook lawan `label_folds_v6.csv` | 0 label berbeda di 1.861 citra, 1 isian tambahan di notebook |
| Kesamaan hash prediksi | tidak diuji, karena user meminta tanpa uji determinism dan notebook memakai 0 worker |

`smoke_notebook.py` tidak bisa dipakai langsung karena `exec` menolak baris `!nvidia-smi` dan `%pip`, jadi `smoke.py` membuang baris magic lalu menjalankan logika yang sama.

**Run**: kernel `python3` mengarah ke miniconda tanpa CUDA, jadi run memakai kernel `kaggle-py311-d`. Embedding CNN dan co-teaching dipindah ke GPU atas permintaan user.

**Skor temuan** gabungan ViT dan CNN v6: accuracy 0,975 (468 dari 480), macro F1 0,975, CI 95% 0,9595 sampai 0,989. ViT 466, CNN 466. Aturan foil terbukti di tingkat kelas: metal 60 dari 60.

**Batas**: `Isi nama anggota` di banner masih perlu diisi. Label ada `label_final.csv`, `OUTPUT_DIR/cache` berisi embedding dan prediksi, jadi run ulang hanya menghitung bagian yang belum ada.
