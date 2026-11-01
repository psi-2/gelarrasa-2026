> Arsip 2026-10-31; superseded by docs/constraint.md. The old two-backbone/ensemble path is not approved under current constraints.

# Kontrak eksperimen (lock)

Diisi sebelum loop prepro/modelling. Perubahan harus dicatat di bagian Riwayat perubahan beserta alasannya.

| Item | Nilai |
|---|---|
| Metric kompetisi | Accuracy |
| Arah metric | lebih besar lebih baik |
| Acuan | temuan (`data/temuan/anotate.csv`, diasumsikan benar oleh user, keaslian belum terverifikasi), metric Accuracy dan macro F1 |
| Skema validasi | hold-out StratifiedGroupKFold fold 0, group = citra mirip (cos DINOv3 >= 0,9 atau pHash Hamming <= 4) |
| File fold | temp/exp_009_t1_dinov3/output/label_folds.csv |
| Seed | 42 |
| Model terkunci (fase prepro) | belum |
| Baseline | EXP-005, macro F1 temuan 0,7113 |
| Target user | accuracy temuan di atas 0,98 (minimal 471 dari 480), terbaik EXP-009 0,9542 |
| Batas backbone | paling banyak 2 pada model prediksi akhir, arsitektur berbeda: DINOv3 ViT-L dan ConvNeXt-L DINOv3 |
| Ambang kenaikan layak | melebihi variasi bootstrap skor temuan |
| Constraint | NO VLM, lihat docs/rule.md |
| Izin khusus | pseudo label: train saja, label cluster DINOv3 yang dinamai otomatis dari ConvNeXtV2 (permintaan user 2026-10-31); pseudo label test: tidak / data eksternal: sesuai rule.md / penamaan cluster manual oleh LLM: tidak (rule 8, 9) |

## Alasan pemilihan validasi

Label train tidak tersedia, jadi skor hold-out hanya mengukur kesepakatan dengan label cluster dan akan terlalu optimistis. Group wajib karena ada 784 citra mirip dalam 192 group (terbesar 60 citra). Hold-out dipilih atas permintaan user.

## Riwayat perubahan

| Tanggal | Perubahan | Alasan |
|---|---|---|
| 2026-10-31 | Kontrak diisi untuk jalur non-VLM EXP-006, hold-out dengan group DINOv3 | User meminta clustering embedding DINOv3 lalu train DINOv3 dengan hold-out. Nama cluster diambil dari massa kategori ConvNeXtV2 karena cluster tidak punya nama |
| 2026-10-31 | Acuan diganti ke temuan | User memberi `data/temuan/anotate.csv` dan memintanya diperlakukan sebagai label test. Hanya `temp/exp_007_eval_temuan/eval_temuan.py` yang boleh membacanya |
| 2026-10-31 | Constraint NO VLM ditetapkan, eksperimen berbasis VLM dihapus | Keputusan user, sesuai rule 9 |
| 2026-10-31 | Label aktif diganti ke t1_in1k, skema EXP-009 | Normalisasi prior menghapus bias `plastic`. User sempat mengizinkan Claude melabeli cluster, tetapi ditolak karena melanggar rule 8 dan 9 |
