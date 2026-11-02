# Notebook final DINOv3 7B + ConvNeXt-L (belum dijalankan)

Builder `build_nb.py` menghasilkan `notebook/dsc_gelarrasa_2026_dinov3.ipynb` (81 sel). Notebook konversi bobot fp16: builder `temp/exp_014_dinov3_7b/build_konversi.py` menghasilkan `notebook/konversi_dinov3_7b_fp16.ipynb`. Rencana dan alasan ada di `docs/rencana_7b.md`.

Uji yang sudah dilakukan di lokal tanpa model:

- Semua sel kode lolos `ast.parse` dan pyflakes tanpa nama tak terdefinisi.
- Sel non-GPU (perbandingan ruang, partisi acuan berurutan kanonik, bootstrap, grup near-duplicate, PECAH, validasi `NAMA_CLUSTER`, SGKF) dijalankan pada cache ViT-L lokal dengan nama dummy. Grup yang terbelah antar fold: 0. `phash8` identik bit-per-bit dengan pHash eda_104.

Belum diuji: semua sel GPU (loader 7B dua GPU, `cek_presisi`, ekstraksi, head, training, inference). Run pertama di Kaggle selesai normal dengan `LABEL_SIAP = False` (bagian 8 sampai 11 dilewati) sampai tim mengisi `NAMA_CLUSTER`.
