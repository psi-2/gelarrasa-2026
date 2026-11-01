# EXP-010 — draft CNN v5

Berkas v5 yang perlu dipertahankan:

- Label train: `temp/eda_113_elektronik/output/label_folds_v5.csv`, dibuat oleh `temp/eda_113_elektronik/label_v5.py`.
- Rancangan dan hyperparameter CNN: `train_dua.py` dengan `--arch cnn` (ConvNeXt-L DINOv3, ukuran 384 secara default).
- Draft CNN 384: `output/cnn_v5.csv`.
- Ablasi CNN 448: `output/draft_cnn448_v5.csv`.
- Hasil perbandingan: `output/draft_vit_v5.csv`, `output/draft_gabungan_v5.csv`, `output/draft_gabungan448_v5.csv`, dan `output/gabung_v5.json`. Draft gabungan adalah arsip; constraint terbaru hanya mengizinkan satu model pada submission.
- Cache kecil CNN v5 dan v5-448 ada di folder masing-masing: `fold0.json`, `vram.csv`, `oof_fold0.npy`, `sisa_fold0.npy`, dan `test_fold0.npy`. Array OOF adalah kecocokan terhadap label pseudo pada fold, bukan akurasi ground truth.
- Checkpoint `cnn_v5_448/model_fold0.pt` berukuran sekitar 749 MB dan tidak disertakan. Itu checkpoint varian 448; catatan log eksperimen menyebut varian 448 lebih buruk dari CNN 384. Prediksi draft dan ringkasan variannya tetap disertakan.

Untuk menjalankan ulang kandidat CNN v5 satu model tanpa menimpa hasil lama:

```powershell
python temp/exp_010_dua_backbone/train_dua.py --arch cnn --label temp/eda_113_elektronik/output/label_folds_v5.csv --tag cnn_v5_repro
```

Jalankan hanya setelah memeriksa [docs/constraint.md](../../docs/constraint.md) dan memakai validasi lokal yang sesuai.
