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
- **Jalur aktif (2026-11-02)**: [docs/rencana_7b.md](docs/rencana_7b.md), notebook Kaggle di `notebook/` (utama `dsc_gelarrasa_2026_dinov3.ipynb` dan `konversi_dinov3_7b_fp16.ipynb`, belum dijalankan; builder di `temp/nb_7b/` dan `temp/exp_014_dinov3_7b/`), EDA `temp/eda_201` s.d. `eda_209`, preprocessing `temp/prep_301_verifikasi/`. Label v4 sampai v6 dan t1 tidak dipakai karena dipengaruhi skor anotasi test. EDA lokal memakai `.venv` (uv, Python 3.12).
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
| `docs/HANDOFF.md` | 3 KB | Handoff eksperimen — DSC Gelar Rasa 2026 |
| `docs/constraint.md` | 4 KB | Constraint — DSC Gelar Rasa 2026 |
| `docs/context.md` | 9 KB | Data Science Competition Gelar Rasa 2026 |
| `docs/data.md` | 3 KB | Data |
| `docs/environment.md` | 3 KB | > Snapshot komputer sumber per 2026-10-31. Path conda, cache model, dan GPU di bawah ini hanya referensi; sesuaikan di p |
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
| `scripts/utils/detect_env.py` | 5 KB | Deteksi lingkungan kerja: conda env, kaggle credential, GPU, cache model, disk. |
| `scripts/utils/log_exp.py` | 4 KB | Tambah / update baris log_exp.md dengan format baku. |
| `scripts/utils/model_loader.py` | 3 KB | Resolve model dari cache lokal dulu, fallback ke Hugging Face Hub. |
| `scripts/utils/promote.py` | 2 KB | Promosikan hasil dari temp/ ke folder final, lalu update log_exp dan katalog. |
| `scripts/utils/seed_utils.py` | 3 KB | Seed strict dan deterministik untuk semua eksperimen ngedata. |
| `scripts/utils/update_katalog.py` | 4 KB | Regenerasi bagian AUTO di katalog.md. Bagian MANUAL tidak disentuh. |
| `scripts/utils/vram_monitor.py` | 5 KB | Pantau VRAM dedicated dan shared GPU memory (deteksi bocor ke RAM di Windows). |

### notebook

| file | ukuran | deskripsi |
|---|---|---|
| `notebook/dsc_gelarrasa_2026_dinov3.ipynb` | 61 KB | DSC Gelar Rasa 2026: Klasifikasi Citra Sampah |
| `notebook/konversi_dinov3_7b_fp16.ipynb` | 7 KB | DSC Gelar Rasa 2026: Konversi Bobot DINOv3 ViT-7B ke fp16 |

### notebooks

(kosong)

### results

(kosong)

### submission

(kosong)

### temp (BELUM TERBUKTI, jangan dianggap final)

| file | ukuran | deskripsi |
|---|---|---|
| `temp/eda_common.py` | 3 KB | Helper bersama untuk skrip EDA 2xx (laptop, CPU, tanpa inference model). |

- `log_exp.md`: log semua eksperimen
<!-- AUTO:END -->
