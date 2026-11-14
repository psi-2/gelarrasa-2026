# Notebook final DINOv3 ViT-L + ConvNeXt-L (versi 2, 2026-11-13)

Builder `build_nb.py` menghasilkan `notebook/dsc_gelarrasa_2026_dinov3.ipynb` (89 sel). Kaggle: `riantamaputra/dsc-gelarrasa-2026-dinov3` (versi 3 dipush 2026-11-13 04.30 WIB), dataset privat `riantamaputra/gelarrasa-cache` (embedding ViT-L/7B train, partisi Run 1, bootstrap, sub-cluster).

Perubahan dari versi 1 (Run 1):

- Head ImageNet (ConvNeXtV2-H) dibuang: tidak ada model jadi di alur pelabelan (QnA 4 dan 5 Nov). `knn_top1` dikutip dari Run 1.
- Penamaan: montage sampai 40 citra per cluster dipisah sub-cluster k=2; kunci `NAMA_CLUSTER` boleh `'5'` atau `'5a'`/`'5b'`; `LABEL_TANGAN` dengan assert kuota 5 per kelas dan 40 total.
- Pembersihan label (bagian 8): stab_nama >= 0,7, kNN-10 nama ruang 7B >= 0,5 (tanpa tetangga satu grup), buang grup duplikat yang namanya konflik. Hanya buang, tidak relabel.
- Training: cek presisi fp16 vs fp32 ConvNeXt-L sebelum training, `latih_aman` mengulang fold dalam fp32 bila loss non-finite, gradient checkpointing di fp32, clip grad 1,0. RRC 0,5 (prep_302).
- Uji jalan: bila nama belum lengkap, training 1 epoch label acak + prediksi 16 test untuk menguji kode GPU, tanpa submission.

Uji lokal tanpa model (scratchpad `uji_nb.py`, nama dummy acak): semua sel non-GPU baru jalan; partisi dan sub-cluster dari cache = hitung ulang (1,0); 0 grup terbelah antar fold; pyflakes bersih.

Belum terbukti: semua sel GPU ConvNeXt-L (unduhan bobot, cek presisi, training, inference). Run versi 3 = tahap 1 + uji jalan. Run final butuh `NAMA_CLUSTER` dari tim.
