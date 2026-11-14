# Katalog proyek

Pintu masuk untuk melanjutkan kompetisi DSC Gelar Rasa 2026.

<!-- MANUAL:START -->
## Status sekarang

- **Aturan utama**: [docs/constraint.md](docs/constraint.md), termasuk larangan ensemble dan larangan memakai test untuk training atau seleksi.
- **Data**: `data/train/` (1.920), `data/test/` (480), sample `data/SampleSubmission_DSC.csv`.
- **Handoff**: [docs/HANDOFF.md](docs/HANDOFF.md) menjelaskan hasil lama, cache penting, dan titik lanjut.
- **Riwayat**: [log_exp.md](log_exp.md), dengan detail di `temp/`.
- **Cache DINOv3**: embedding train/test dan ID ada di `temp/exp_006_cluster_dinov3/output/`.
- **Perhatian**: hasil ensemble lama tidak sesuai batasan terbaru. Anotasi test tidak disimpan di repo.
- **Jalur aktif (2026-11-14)**: notebook final `notebook/dsc_gelarrasa_2026_dinov3_7b_qlora.ipynb` untuk Jupyter Vast.ai (builder `temp/nb_7b_qlora/build_nb.py`, catatan `temp/nb_7b_qlora/notes.md`, EXP-019): DINOv3 ViT-7B dengan LoRA di 40 blok (NF4 di RTX 5090, bf16 di GPU 96 GB), label final `results/label_train.csv` dari notebook v2, satu `submission.csv` ke `keluaran_7b/`. Belum dijalankan di GPU; uji logika CPU `temp/nb_7b_qlora/uji_logika.py`.
- **Jalur sebelumnya (2026-11-13)**: notebook v2 `notebook/dsc_gelarrasa_2026_dinov3.ipynb` (builder `temp/nb_7b/build_nb.py`, catatan `temp/nb_7b/notes.md`), Kaggle `riantamaputra/dsc-gelarrasa-2026-dinov3` versi 3 + dataset privat `riantamaputra/gelarrasa-cache`. Menunggu `NAMA_CLUSTER` dari tim: montage `results/gambar_v2/`, prioritas `results/prioritas_tinjau.csv`, templat `results/nama_cluster_templat.csv`. EDA baru `temp/eda_210` s.d. `eda_214`, preprocessing `temp/prep_302_verifikasi_run1/` (RRC 0,5). Cache Run 1 diekstrak ke `results/cache/` (gitignored). EDA lokal memakai `.venv` (uv, Python 3.12, torch CPU hanya untuk transform).
- **Jalur lama (2026-11-02)**: [docs/rencana_7b.md](docs/rencana_7b.md), EDA `temp/eda_201` s.d. `eda_209`, `temp/prep_301_verifikasi/`. Label v4 sampai v6 dan t1 tidak dipakai karena dipengaruhi skor anotasi test.
<!-- MANUAL:END -->

<!-- AUTO:START -->
## Indeks file

### data
- `data/SampleSubmission_DSC.csv` (6 KB)
- `data/test/`: 480 file, 15.0 MB, ekstensi ['.jpeg', '.jpg', '.png']
- `data/train/`: 1920 file, 78.5 MB, ekstensi ['.jpeg', '.jpg']

### docs

| file | ukuran | deskripsi |
|---|---|---|
| `docs/constraint.md` | 7 KB | Constraint — DSC Gelar Rasa 2026 |
| `docs/context.md` | 9 KB | Data Science Competition Gelar Rasa 2026 |
| `docs/data.md` | 3 KB | Data |
| `docs/environment.md` | 3 KB | > Snapshot komputer sumber per 2026-10-31. Path conda, cache model, dan GPU di bawah ini hanya referensi; sesuaikan di p |
| `docs/HANDOFF.md` | 3 KB | Handoff eksperimen — DSC Gelar Rasa 2026 |
| `docs/lock_2026-10-31_archived.md` | 2 KB | > Arsip 2026-10-31; superseded by docs/constraint.md. The old two-backbone/ensemble path is not approved under current c |
| `docs/overview.md` | 3 KB | Overview: Data Science Competition (DSC) Gelar Rasa 2026 |
| `docs/rencana_7b.md` | 10 KB | Rencana DINOv3 ViT-7B untuk clustering train |
| `docs/research_material.md` | 12 KB | Riset pembedaan material untuk penamaan kelas sampah |
| `docs/research_unsupervised.md` | 37 KB | Riset klasifikasi sampah tanpa label dan tanpa VLM |
| `docs/rule.md` | 5 KB | Aturan |

### eda

(kosong)

### scripts

| file | ukuran | deskripsi |
|---|---|---|
| `scripts/__init__.py` | 0 B |  |
| `scripts/utils/__init__.py` | 0 B |  |
| `scripts/utils/check_notebook.py` | 5 KB | Validasi notebook ngedata sebelum dipromosikan ke notebooks/. |
| `scripts/utils/detect_env.py` | 6 KB | Deteksi lingkungan kerja: conda env, kaggle credential, GPU, cache model, disk. |
| `scripts/utils/log_exp.py` | 4 KB | Tambah / update baris log_exp.md dengan format baku. |
| `scripts/utils/model_loader.py` | 3 KB | Resolve model dari cache lokal dulu, fallback ke Hugging Face Hub. |
| `scripts/utils/promote.py` | 2 KB | Promosikan hasil dari temp/ ke folder final, lalu update log_exp dan katalog. |
| `scripts/utils/seed_utils.py` | 4 KB | Seed strict dan deterministik untuk semua eksperimen ngedata. |
| `scripts/utils/update_katalog.py` | 4 KB | Regenerasi bagian AUTO di katalog.md. Bagian MANUAL tidak disentuh. |
| `scripts/utils/vram_monitor.py` | 5 KB | Pantau VRAM dedicated dan shared GPU memory (deteksi bocor ke RAM di Windows). |

### notebook

| file | ukuran | deskripsi |
|---|---|---|
| `notebook/dsc_gelarrasa_2026_dinov3.ipynb` | 75 KB | DSC Gelar Rasa 2026: Klasifikasi Citra Sampah |
| `notebook/kernel-metadata.json` | 480 B |  |
| `notebook/konversi_dinov3_7b_fp16.ipynb` | 7 KB | DSC Gelar Rasa 2026: Konversi Bobot DINOv3 ViT-7B ke fp16 |

### notebooks

(kosong)

### results

| file | ukuran | deskripsi |
|---|---|---|
| `results/__results___files/__results___24_3.png` | 20 KB |  |
| `results/__results___files/__results___27_1.png` | 74 KB |  |
| `results/__results___files/__results___33_0.png` | 833 KB |  |
| `results/__results___files/__results___51_2.png` | 25 KB |  |
| `results/__results___files/__results___59_0.png` | 420 KB |  |
| `results/__results___files/__results___59_1.png` | 559 KB |  |
| `results/__results___files/__results___59_10.png` | 445 KB |  |
| `results/__results___files/__results___59_11.png` | 479 KB |  |
| `results/__results___files/__results___59_12.png` | 348 KB |  |
| `results/__results___files/__results___59_13.png` | 579 KB |  |
| `results/__results___files/__results___59_14.png` | 550 KB |  |
| `results/__results___files/__results___59_15.png` | 363 KB |  |
| `results/__results___files/__results___59_16.png` | 719 KB |  |
| `results/__results___files/__results___59_17.png` | 605 KB |  |
| `results/__results___files/__results___59_18.png` | 487 KB |  |
| `results/__results___files/__results___59_19.png` | 610 KB |  |
| `results/__results___files/__results___59_2.png` | 505 KB |  |
| `results/__results___files/__results___59_20.png` | 217 KB |  |
| `results/__results___files/__results___59_21.png` | 294 KB |  |
| `results/__results___files/__results___59_22.png` | 348 KB |  |
| `results/__results___files/__results___59_23.png` | 465 KB |  |
| `results/__results___files/__results___59_24.png` | 560 KB |  |
| `results/__results___files/__results___59_25.png` | 438 KB |  |
| `results/__results___files/__results___59_26.png` | 445 KB |  |
| `results/__results___files/__results___59_27.png` | 365 KB |  |
| `results/__results___files/__results___59_28.png` | 424 KB |  |
| `results/__results___files/__results___59_29.png` | 563 KB |  |
| `results/__results___files/__results___59_3.png` | 403 KB |  |
| `results/__results___files/__results___59_30.png` | 751 KB |  |
| `results/__results___files/__results___59_31.png` | 344 KB |  |
| `results/__results___files/__results___59_32.png` | 713 KB |  |
| `results/__results___files/__results___59_33.png` | 485 KB |  |
| `results/__results___files/__results___59_34.png` | 558 KB |  |
| `results/__results___files/__results___59_35.png` | 441 KB |  |
| `results/__results___files/__results___59_36.png` | 682 KB |  |
| `results/__results___files/__results___59_37.png` | 691 KB |  |
| `results/__results___files/__results___59_38.png` | 377 KB |  |
| `results/__results___files/__results___59_39.png` | 371 KB |  |
| `results/__results___files/__results___59_4.png` | 397 KB |  |
| `results/__results___files/__results___59_40.png` | 560 KB |  |
| `results/__results___files/__results___59_41.png` | 353 KB |  |
| `results/__results___files/__results___59_42.png` | 331 KB |  |
| `results/__results___files/__results___59_43.png` | 345 KB |  |
| `results/__results___files/__results___59_44.png` | 341 KB |  |
| `results/__results___files/__results___59_45.png` | 176 KB |  |
| `results/__results___files/__results___59_46.png` | 455 KB |  |
| `results/__results___files/__results___59_47.png` | 224 KB |  |
| `results/__results___files/__results___59_5.png` | 406 KB |  |
| `results/__results___files/__results___59_6.png` | 424 KB |  |
| `results/__results___files/__results___59_7.png` | 645 KB |  |
| `results/__results___files/__results___59_8.png` | 517 KB |  |
| `results/__results___files/__results___59_9.png` | 588 KB |  |
| `results/bukti_cluster.csv` | 4 KB |  |
| `results/cache/boot_train.npy` | 300 KB |  |
| `results/cache/cluster_train.csv` | 31 KB |  |
| `results/cache/emb_7b_train_asli_cls.npy` | 15 MB |  |
| `results/cache/emb_7b_train_asli_l20.npy` | 15 MB |  |
| `results/cache/emb_7b_train_asli_l30.npy` | 15 MB |  |
| `results/cache/emb_7b_train_asli_patch.npy` | 15 MB |  |
| `results/cache/emb_7b_train_flip_cls.npy` | 15 MB |  |
| `results/cache/emb_7b_train_flip_l20.npy` | 15 MB |  |
| `results/cache/emb_7b_train_flip_l30.npy` | 15 MB |  |
| `results/cache/emb_7b_train_flip_patch.npy` | 15 MB |  |
| `results/cache/emb_vitl_train_asli_cls.npy` | 4 MB |  |
| `results/cache/emb_vitl_train_asli_patch.npy` | 4 MB |  |
| `results/cache/emb_vitl_train_flip_cls.npy` | 4 MB |  |
| `results/cache/emb_vitl_train_flip_patch.npy` | 4 MB |  |
| `results/cache/head_in1k_train.npy` | 4 MB |  |
| `results/cache/manifest_7b.json` | 522 B |  |
| `results/cache/manifest_vitl.json` | 393 B |  |
| `results/cache/presisi_7b.csv` | 2 KB |  |
| `results/cache/presisi_vitl.csv` | 1 KB |  |
| `results/cache/subcluster_train.csv` | 22 KB |  |
| `results/gambar/cluster_0.png` | 286 KB |  |
| `results/gambar/cluster_1.png` | 376 KB |  |
| `results/gambar/cluster_10.png` | 299 KB |  |
| `results/gambar/cluster_11.png` | 323 KB |  |
| `results/gambar/cluster_12.png` | 237 KB |  |
| `results/gambar/cluster_13.png` | 393 KB |  |
| `results/gambar/cluster_14.png` | 371 KB |  |
| `results/gambar/cluster_15.png` | 253 KB |  |
| `results/gambar/cluster_16.png` | 471 KB |  |
| `results/gambar/cluster_17.png` | 406 KB |  |
| `results/gambar/cluster_18.png` | 334 KB |  |
| `results/gambar/cluster_19.png` | 410 KB |  |
| `results/gambar/cluster_2.png` | 343 KB |  |
| `results/gambar/cluster_20.png` | 149 KB |  |
| `results/gambar/cluster_21.png` | 204 KB |  |
| `results/gambar/cluster_22.png` | 244 KB |  |
| `results/gambar/cluster_23.png` | 319 KB |  |
| `results/gambar/cluster_24.png` | 378 KB |  |
| `results/gambar/cluster_25.png` | 301 KB |  |
| `results/gambar/cluster_26.png` | 302 KB |  |
| `results/gambar/cluster_27.png` | 254 KB |  |
| `results/gambar/cluster_28.png` | 289 KB |  |
| `results/gambar/cluster_29.png` | 376 KB |  |
| `results/gambar/cluster_3.png` | 272 KB |  |
| `results/gambar/cluster_30.png` | 499 KB |  |
| `results/gambar/cluster_31.png` | 239 KB |  |
| `results/gambar/cluster_32.png` | 475 KB |  |
| `results/gambar/cluster_33.png` | 329 KB |  |
| `results/gambar/cluster_34.png` | 376 KB |  |
| `results/gambar/cluster_35.png` | 305 KB |  |
| `results/gambar/cluster_36.png` | 455 KB |  |
| `results/gambar/cluster_37.png` | 462 KB |  |
| `results/gambar/cluster_38.png` | 260 KB |  |
| `results/gambar/cluster_39.png` | 260 KB |  |
| `results/gambar/cluster_4.png` | 271 KB |  |
| `results/gambar/cluster_40.png` | 377 KB |  |
| `results/gambar/cluster_41.png` | 244 KB |  |
| `results/gambar/cluster_42.png` | 228 KB |  |
| `results/gambar/cluster_43.png` | 233 KB |  |
| `results/gambar/cluster_44.png` | 235 KB |  |
| `results/gambar/cluster_45.png` | 122 KB |  |
| `results/gambar/cluster_46.png` | 312 KB |  |
| `results/gambar/cluster_47.png` | 152 KB |  |
| `results/gambar/cluster_5.png` | 279 KB |  |
| `results/gambar/cluster_6.png` | 290 KB |  |
| `results/gambar/cluster_7.png` | 423 KB |  |
| `results/gambar/cluster_8.png` | 354 KB |  |
| `results/gambar/cluster_9.png` | 397 KB |  |
| `results/gambar_v2/cluster_0.png` | 643 KB |  |
| `results/gambar_v2/cluster_1.png` | 762 KB |  |
| `results/gambar_v2/cluster_10.png` | 546 KB |  |
| `results/gambar_v2/cluster_11.png` | 926 KB |  |
| `results/gambar_v2/cluster_12.png` | 380 KB |  |
| `results/gambar_v2/cluster_13.png` | 1 MB |  |
| `results/gambar_v2/cluster_14.png` | 640 KB |  |
| `results/gambar_v2/cluster_15.png` | 626 KB |  |
| `results/gambar_v2/cluster_16.png` | 1 MB |  |
| `results/gambar_v2/cluster_17.png` | 1 MB |  |
| `results/gambar_v2/cluster_18.png` | 645 KB |  |
| `results/gambar_v2/cluster_19.png` | 1 MB |  |
| `results/gambar_v2/cluster_2.png` | 608 KB |  |
| `results/gambar_v2/cluster_20.png` | 249 KB |  |
| `results/gambar_v2/cluster_21.png` | 522 KB |  |
| `results/gambar_v2/cluster_22.png` | 644 KB |  |
| `results/gambar_v2/cluster_23.png` | 845 KB |  |
| `results/gambar_v2/cluster_24.png` | 980 KB |  |
| `results/gambar_v2/cluster_25.png` | 786 KB |  |
| `results/gambar_v2/cluster_26.png` | 437 KB |  |
| `results/gambar_v2/cluster_27.png` | 438 KB |  |
| `results/gambar_v2/cluster_28.png` | 641 KB |  |
| `results/gambar_v2/cluster_29.png` | 790 KB |  |
| `results/gambar_v2/cluster_3.png` | 669 KB |  |
| `results/gambar_v2/cluster_30.png` | 978 KB |  |
| `results/gambar_v2/cluster_31.png` | 649 KB |  |
| `results/gambar_v2/cluster_32.png` | 900 KB |  |
| `results/gambar_v2/cluster_33.png` | 870 KB |  |
| `results/gambar_v2/cluster_34.png` | 887 KB |  |
| `results/gambar_v2/cluster_35.png` | 706 KB |  |
| `results/gambar_v2/cluster_36.png` | 1 MB |  |
| `results/gambar_v2/cluster_37.png` | 1 MB |  |
| `results/gambar_v2/cluster_38.png` | 559 KB |  |
| `results/gambar_v2/cluster_39.png` | 626 KB |  |
| `results/gambar_v2/cluster_4.png` | 486 KB |  |
| `results/gambar_v2/cluster_40.png` | 871 KB |  |
| `results/gambar_v2/cluster_41.png` | 505 KB |  |
| `results/gambar_v2/cluster_42.png` | 262 KB |  |
| `results/gambar_v2/cluster_43.png` | 302 KB |  |
| `results/gambar_v2/cluster_44.png` | 298 KB |  |
| `results/gambar_v2/cluster_45.png` | 113 KB |  |
| `results/gambar_v2/cluster_46.png` | 356 KB |  |
| `results/gambar_v2/cluster_47.png` | 262 KB |  |
| `results/gambar_v2/cluster_5.png` | 805 KB |  |
| `results/gambar_v2/cluster_6.png` | 489 KB |  |
| `results/gambar_v2/cluster_7.png` | 989 KB |  |
| `results/gambar_v2/cluster_8.png` | 873 KB |  |
| `results/gambar_v2/cluster_9.png` | 864 KB |  |
| `results/nama_cluster_templat.csv` | 1 KB |  |
| `results/prioritas_tinjau.csv` | 4 KB |  |

### submission

(kosong)

### temp (BELUM TERBUKTI, jangan dianggap final)

| file | ukuran | deskripsi |
|---|---|---|
| `temp/eda_common.py` | 4 KB | Helper bersama untuk skrip EDA 2xx (laptop, CPU, tanpa inference model). |

- `log_exp.md`: log semua eksperimen
<!-- AUTO:END -->
