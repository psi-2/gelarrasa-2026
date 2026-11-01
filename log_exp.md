> Arsip historis; constraint terbaru di docs/constraint.md berlaku. EXP-011 adalah ensemble dan tidak sesuai batasan terbaru. Skor terhadap anotasi test lokal tidak boleh dipakai untuk tuning.

# Log eksperimen

Format baku, jangan diubah. Tulis lewat `scripts/utils/log_exp.py`.

| ID | Tanggal | Fase | Deskripsi | Metode/Model | Prepro/Fitur | Validasi | CV | Public Score | Temuan | Status | Path |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | 2026-10-31 | setup | init proyek dan deteksi env | - | - | - | - | - | - | promoted | docs/environment.md |
| EXP-002 | 2026-10-31 | research | cek kelengkapan berkas: citra utuh, label train tidak ada | - | - | - | - | - | - | temp | temp/eda_002_cek_berkas/ |
| EXP-005 | 2026-10-31 | modelling | pemetaan 231 kelas ImageNet ke 8 kategori, plastic jadi kelas penampung | ConvNeXtV2 huge IN1k 384 | flip TTA, jumlah softmax per kategori | tanpa label | - | - | mF1 0.7113, acc 0.7104 | temp | temp/exp_005_imagenet_map/ |
| EXP-006 | 2026-10-31 | modelling | cluster K=64 dinamai ConvNeXtV2, 1.512 label, skor hold-out kesepakatan dengan label cluster | DINOv3 ViT-L 384 | label cluster embedding DINOv3 | hold-out 20% group | 0.9558 | - | mF1 0.8574, acc 0.8542 | temp | temp/exp_006_cluster_dinov3/ |
| EXP-007 | 2026-10-31 | modelling | skor temuan semua prediksi non-VLM, EXP-009 melewati target | - | - | temuan | - | - | - | temp | temp/exp_007_eval_temuan/ |
| EXP-008 | 2026-10-31 | research | inventaris classifier, unduh head ImageNet-22k dan DMS46, belum dijalankan | - | - | - | - | - | - | temp | temp/exp_008_in22k_map/ |
| EXP-009 | 2026-10-31 | modelling | label bagging KMeans dinamai ConvNeXtV2 dengan normalisasi prior, 1.626 label, skor hold-out melingkar | DINOv3 ViT-L 384 | label t1_in1k | hold-out group fold 0 | 0.9871 | - | mF1 0.9543, acc 0.9542 | temp | temp/exp_009_t1_dinov3/ |
| EXP-010 | 2026-10-31 | modelling | dua backbone ViT dan CNN, label v2 sampai v5, CNN 384 terbaik, 448 lebih buruk | DINOv3 ViT-L + ConvNeXt-L DINOv3 | label v5, baterai jadi electronic | full training | - | - | gabungan acc 0.9708, CNN 0.9750 | temp | temp/exp_010_dua_backbone/ |
| EXP-011 | 2026-10-31 | notebook | notebook end-to-end label v6, ensemble 8 model dari 2 backbone, label identik dengan skrip | DINOv3 ViT-L + ConvNeXt-L DINOv3 | label v6, foil jadi metal | full training | - | - | ensemble mF1 0.9771, acc 0.9771 | best | temp/nb_final/ |
