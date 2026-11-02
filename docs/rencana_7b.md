# Rencana DINOv3 ViT-7B untuk clustering train

Status: 2026-11-02. Notebook sudah ditulis tetapi belum dijalankan (`notebook/dsc_gelarrasa_2026_dinov3.ipynb`, builder `temp/nb_7b/build_nb.py`). Angka waktu dan VRAM di bawah adalah **perkiraan**, kecuali yang ditandai "terukur". Sumber kebenaran tetap [constraint.md](constraint.md).

## Ringkasan keputusan

| Pertanyaan | Keputusan |
|---|---|
| Muat di T4? | Ya, bobot fp16 **dibagi ke dua T4** (20 + 20 blok, sekitar 6,3 GiB per GPU). Satu GPU hanya menyisakan sekitar 1,7 GiB, jadi tidak dipakai. |
| fp16 atau int8? | fp16 dengan **residual stream fp32** dan matmul fp16 lewat autocast. int8 tidak dipakai. |
| Risiko overflow | Ditangani di tiga lapis: residual fp32, fallback blok ke fp32 bila keluaran non-finite, dan `cek_presisi` per blok sebelum ekstraksi. |
| Sumber bobot | `timm/vit_7b_patch16_dinov3.lvd1689m` revisi `b40f6c8`. **Tidak gated** (versi `facebook/` gated manual). Satu berkas fp32 26,86 GB (terukur dari API HF). |
| Fitur yang di-cache | CLS dan rata-rata patch lapis akhir, plus CLS lapis 20 dan 30, untuk pandangan asli dan flip. fp16 `.npy` dengan manifest. |
| Pindah ke 7B? | Hanya bila lolos aturan beku: `ari_boot` naik minimal 0,03, `knn_top1` tidak turun, dan jumlah cluster rapuh tidak bertambah. |
| Pelabelan cluster | **Tim yang menamai** dari montage. Top-5 kelas ImageNet mentah hanya bukti pendukung, tanpa pemetaan otomatis. |
| Classifier final | **ConvNeXt-L DINOv3 fine-tune**. Head di atas embedding 7B tidak dipakai. Akibatnya test **tidak** perlu diekstrak dengan 7B. |

## 1. Memori, presisi, dan waktu

Hitungan parameter dari konfigurasi timm 1.0.29 (dim 4096, 40 blok, SwiGLU dengan rasio 2, 32 head, 4 register): sekitar 168 juta per blok, total 6,71 miliar. Angka ini cocok dengan ukuran berkas 26,86 GB / 4 byte.

| Mode | Bobot | Sisa di T4 (sekitar 14,6 GiB bisa dipakai) | Catatan |
|---|---|---|---|
| fp32 satu GPU | 25 GiB | tidak muat | |
| fp32 dua GPU | 12,5 GiB per GPU | sekitar 2 GiB | lambat (fp32 di T4 sekitar 8 TFLOPS), batch 1–2 |
| **fp16 dua GPU** | **6,3 GiB per GPU** | **sekitar 8 GiB** | **dipakai**, batch 16 di 384 px |
| fp16 satu GPU | 12,5 GiB | sekitar 1,7 GiB | mungkin jalan di batch kecil, rawan OOM |
| int8 (bitsandbytes) | sekitar 6,3 GiB | sekitar 8 GiB | aktivasi tetap fp16 (risiko overflow sama), matmul int8 lambat di T4, fitur ikut terdistorsi |

**Kenapa residual fp32.** DINOv3 dilatih dengan bf16, yang rentang eksponennya sama dengan fp32. Di fp16 (maksimum 65.504), aktivasi besar di residual stream adalah titik overflow yang paling mungkin. Di notebook, `h` disimpan fp32, sedangkan attention dan MLP berjalan di fp16 lewat autocast. Penjumlahan residual `fp32 + fp16` menghasilkan fp32, jadi tambahan memorinya hanya sekitar 9,5 MB per citra. Bila keluaran blok fp16 tetap non-finite, blok itu dikonversi ke fp32 secara permanen (+0,34 GiB per blok). `cek_presisi` mencatat galat relatif fp16 terhadap fp32 per blok pada 4 citra ke `cache/presisi_7b.csv`.

**Resolusi.** Resolusinya 384 px, sama dengan ViT-L lama dan classifier, karena 88% citra punya sisi pendek minimal 384 (EDA 201). Hasilnya 576 patch ditambah 5 token prefix.

**Perkiraan waktu (belum terukur).** Satu forward 7B di 384 px membutuhkan sekitar 2 × 6,7e9 × 581 ≈ 7,8 TFLOP, ditambah attention sekitar 0,2 TFLOP. Dengan throughput fp16 efektif T4 sekitar 18 TFLOPS dan dua GPU yang bergantian (pipeline naif), estimasinya:

| Tahap | Estimasi |
|---|---|
| Unduh fp32 dari HF (bila tanpa dataset fp16) | 3–8 menit |
| Baca, konversi, dan kirim ke GPU | 3–6 menit |
| Train 1.920 × 2 pandangan | sekitar 30 menit |
| ViT-L (pembanding) | sekitar 2 menit |
| Head ConvNeXtV2-H IN1k | sekitar 2 menit |

Manifest `cache/manifest_7b.json` mencatat waktu dan VRAM terukur setelah run pertama. Perbarui tabel ini dengan angka tersebut.

## 2. Akses bobot dan disk

- `timm/vit_7b_patch16_dinov3.lvd1689m` tidak gated, tetapi lisensinya tetap **DINOv3 License**. Simpan teks lisensinya bila bobot diredistribusi.
- **Jalur A (bawaan notebook bila tidak ada dataset):** notebook mengunduh fp32 26,9 GB ke `/tmp/hf` dengan revisi terkunci. Sebelumnya notebook memeriksa ruang kosong minimal 30 GiB dan berhenti dengan pesan jelas bila kurang. Batas `/kaggle/working` 20 GB tidak terpakai karena unduhan masuk ke `/tmp`. Ruang `/tmp` di sesi Kaggle **belum diverifikasi**, dan cek disk di notebook ada untuk itu.
- **Jalur B (disarankan untuk reproducibility):** jalankan `notebook/konversi_dinov3_7b_fp16.ipynb` sekali di Kaggle (CPU, Internet ON), lalu simpan keluarannya (sekitar 13,4 GB, shard 4 GiB, manifest sha256) sebagai **dataset Kaggle privat**. Notebook otomatis memakai pola `dinov3_7b_fp16*.safetensors` bila dataset itu dilampirkan. Dataset privat yang dilampirkan ke notebook harus ikut dibagikan ke juri saat notebook dikumpulkan.
- RAM Kaggle sekitar 29 GB. Loader tidak pernah memuat seluruh bobot ke RAM: tensor dibaca satu per satu dengan `safe_open`, dikonversi, lalu dikirim ke GPU (`init_empty_weights` + `set_module_tensor_to_device`).
- Model Kaggle yang sudah ada untuk 7B hanya varian satelit (`sat493m`), sehingga tidak relevan.

## 3. Cache dan reproducibility

Berkas di `cache/` (nama tetap, dicari di `/kaggle/working/cache` lalu di `/kaggle/input/**`):

| Berkas | Isi | Ukuran perkiraan |
|---|---|---|
| `emb_7b_train_{asli,flip}_{cls,patch,l20,l30}.npy` | 1.920 × 4.096 fp16 | 8 × 15,7 MB = 126 MB |
| `emb_vitl_train_{asli,flip}_{cls,patch}.npy` | 1.920 × 1.024 fp16 | 16 MB |
| `head_in1k_train.npy` | 1.920 × 1.000 fp16 | 3,8 MB |
| `manifest_{7b,vitl}.json`, `presisi_{7b,vitl}.csv` | revisi, waktu, VRAM, sha256, blok fp32 | kecil |
| `cluster_train.csv`, `boot_train.npy` | partisi acuan, stabilitas, pemetaan 20 bootstrap | kecil |
| `cv_oof.npz`, `prob_test.npy` | OOF CV dan probabilitas test | kecil |

Alur kerja:

1. **Run 1** (Save Version, GPU T4 x2, Internet ON): EDA, ekstraksi, perbandingan ruang, clustering, lalu montage per cluster. Karena `NAMA_CLUSTER` masih kosong, `LABEL_SIAP = False` dan bagian 8 sampai 11 dilewati dengan pesan. Run tetap selesai normal sehingga output tersimpan.
2. Unduh `/kaggle/working/cache` dan `gambar/`, lalu buat dataset Kaggle `gelarrasa-cache` dari folder `cache`.
3. Tim melihat `gambar/cluster_*.png` dan `bukti_cluster.csv`, lalu mengisi `PECAH` (bila perlu) dan `NAMA_CLUSTER` di notebook.
4. **Run 2** dengan dataset cache dilampirkan dan `PAKAI_CACHE = True`: ekstraksi dan clustering dimuat dari cache. Partisi dimuat dari `cluster_train.csv`, bukan dihitung ulang, sehingga nama tetap menunjuk citra yang sama. Notebook juga mencetak ARI hitung ulang sebagai cek. Run lalu lanjut ke CV, model final, dan submission.
5. Untuk reproduksi penuh dari nol, set `PAKAI_CACHE = False`. Sel ekstraksi tetap ada di notebook.

## 4. Apakah 7B lebih baik? (diukur, bukan diasumsikan)

Baseline lokal ViT-L dan ConvNeXt-L dari cache lama (`temp/eda_208_cluster`, resep CLS satu pandangan, train-only):

| K | Silhouette ViT-L | `ari_boot` ViT-L | Cluster rapuh ViT-L | `ari_boot` CNN | Cluster rapuh CNN |
|---|---|---|---|---|---|
| 8 | 0,234 | 0,808 | 0 | 0,761 | 0 |
| 32 | 0,225 | 0,619 | 15 | 0,643 | 14 |
| 48 | 0,246 | 0,638 | 18 | 0,688 | 18 |
| 64 | 0,268 | 0,676 | 32 | 0,738 | 18 |

Metrik di notebook (fungsi `metrik_ruang`, identik dengan EDA 208 kecuali referensi kNN):

- `ari_boot` dan `rapuh`: stabilitas partisi terhadap subsample 80%. Ini metrik utama.
- `knn_top1`: porsi 10 tetangga yang berbagi kelas top-1 ImageNet dari head ConvNeXtV2. Referensi ini independen dari kedua embedding dan tidak memakai pemetaan kelas buatan LLM.
- `ari_seed` dan silhouette hanya dilaporkan. Silhouette tidak sah dibandingkan antar ruang karena bergantung geometri.
- Kesepakatan dengan label lama hanya dihitung bila `label_lama.csv` dilampirkan. Untuk ViT-L metrik ini melingkar (label lama lahir di ruang ViT-L), jadi tidak masuk aturan.

Aturan keputusan ditetapkan sebelum run (lihat Ringkasan). Ekspektasi realistis: makalah DINOv3 melaporkan selisih ViT-L hasil distilasi dan 7B pada tolok ukur global hanya beberapa poin, sehingga 7B **belum tentu** lolos. Bila gagal, notebook memakai ViT-L dan ekstraksi 7B tetap tercatat sebagai bukti.

## 5. Penamaan cluster dan kepatuhan

- Constraint §3 mengizinkan penamaan manual. Pemetaan otomatis lama (`PETA` 231 kelas ImageNet ke 8 kategori) **ditulis oleh LLM**, sehingga rawan dianggap LLM ikut mengklasifikasi (Rule 9). Pemetaan itu tidak dipakai lagi.
- Alur baru: montage 20 citra inti dan 6 citra pinggir per cluster, top-5 nama kelas ImageNet mentah, ukuran, dan stabilitas. Tim mengisi `NAMA_CLUSTER = {'0': 'textile', ...}` atau `campur`. Cluster campuran bisa dipecah dengan `PECAH`.
- Claude boleh melihat grid preprocessing untuk QA (rule.md 2026-11-02), tetapi tidak membuka montage cluster untuk menamai dan tidak mengusulkan nama.
- Label v4 sampai v6 (baterai, foil) dan pemilihan varian t1 dipengaruhi agregat skor anotasi test lokal, jadi semuanya tidak dipakai. Label dibangun ulang dari nol.

## 6. Classifier final

| Opsi | Kelebihan | Kekurangan |
|---|---|---|
| Head linear atau MLP di embedding 7B | training detik-an, CV murah | mereproduksi batas cluster beserta kesalahannya, skor CV melingkar, test **wajib** diekstrak 7B (+3–4 menit dan 13 GB bobot di notebook final) |
| **ConvNeXt-L DINOv3 fine-tune** | pandangan berbeda dari ruang clustering, uji lama (uji 9) menunjukkan fine-tune menambah informasi di atas head linear, test tanpa 7B | CV 5 fold sekitar 1 jam di 2 × T4 (perkiraan) |

Submission berasal dari satu model yang dilatih ulang pada seluruh label dengan 8 epoch. Model fold tidak dirata-rata, dan TTA flip juga tidak dipakai karena merata-rata beberapa prediksi bisa dibaca sebagai blending. Keputusan TTA ini tetap ada di tangan tim.

## 7. Risiko terbuka

| Risiko | Mitigasi |
|---|---|
| Ruang `/tmp` Kaggle < 30 GiB | Jalur B (dataset fp16). Notebook berhenti dengan pesan bila disk kurang. |
| fp16 overflow di dalam MLP | Fallback per blok ke fp32, tercatat di `manifest_7b.json` (`blok_fp32`). |
| KMeans berbeda antar versi sklearn | Partisi disimpan dan dimuat dari cache. ARI hitung ulang hanya dicetak. |
| Nama cluster salah | Bagian 9 menampilkan cluster yang kesepakatan OOF-nya < 0,9 untuk ditinjau ulang lewat montage. |
| Imbalance antar cluster (rasio 5,5 di K=48) | Bobot kelas akar-invers pada loss, stratifikasi pada SGKF. |
