# Riset klasifikasi sampah tanpa label dan tanpa VLM

Status: tahap riset dan tahap uji pada embedding selesai, hasil uji ada di akhir dokumen. Tanggal: 2026-10-31. Bahan mentah: `temp/research_unsup/`. Sinyal material: `docs/research_material.md`.

Tidak ada citra kompetisi yang dilihat, `data/temuan/` tidak dibaca, tidak ada submission, dan tidak ada model yang diunduh. Skrip `temp/research_uji_crossfit/ekstrak_convnext.py` mengubah citra menjadi vektor di CPU, dan hanya vektornya yang dibaca.

## Koreksi setelah eksperimen tim (dibaca dari `temp/eda_110_fusi/notes.md` dan `temp/exp_009_t1_dinov3/notes.md`)

- Jalur yang berhasil adalah normalisasi prior plus bagging KMeans (EXP-009, 458 dari 480). Di laporan saya gagasan prior hanya peringkat 4, dan metode material peringkat 1 belum teruji. Peringkat di bawah adalah prediksi sebelum eksperimen, bukan hasil.
- Label model EM yang saya tempatkan di peringkat 2 sudah dicoba tim dan **gagal**: varian satu pandangan runtuh (metal 0), kesepakatan tiga pandangan 37%.
- Target sekarang 0,98, yaitu minimal 471 dari 480, dengan paling banyak 2 backbone: DINOv3 ViT-L dan ConvNeXt-L DINOv3.
- Hasil uji pada data ada di bagian "Hasil uji pada embedding" di akhir dokumen.

## Jawaban singkat

Tidak ada paper yang menjawab kasus ini secara persis. Pencarian extended tidak menemukan metode yang memetakan kelas ImageNet ke label target tanpa label target, tanpa teks, dan tanpa penamaan manual. Metode yang mendekati mengandalkan salah satu dari tiga hal yang dilarang: CLIP, sedikit label, atau pelabelan cluster manual.

Jawaban yang paling kuat adalah gabungan dari komponen yang terverifikasi: **pisahkan bukti objek dan bukti material, lalu gabungkan di tingkat micro-cluster dengan label model EM**. Head ImageNet-1k tidak punya kelas `paper` atau `plastic` generik. Model material MINC dan DMS punya, keduanya non-VLM, keduanya tidak dilatih pada data sampah, dan bobotnya bisa diunduh.

Target 0,95 belum terbukti bisa dicapai. Perkiraan saya untuk pipeline penuh adalah 0,90 sampai 0,95, dan ini perkiraan, bukan hasil uji. Bila dataset ini serupa RealWaste, supervised pun hanya 89,19% (Inception V3, Single dkk. 2023), jadi plafon bisa di bawah target.

## Diagnosis yang memandu riset

- EXP-006: accuracy 0,8542, yaitu 410 dari 480. Target 0,95 berarti 456 benar, jadi kesalahan harus turun dari 70 ke paling banyak 24, atau minimal 46 kesalahan diperbaiki.
- 39 dari 70 kesalahan adalah `paper` (24) dan `glass` (15) yang menjadi `plastic`. Kelas lain ber-F1 0,91 ke atas.
- Train EXP-005 punya 624 `plastic` dari 1.920, dengan sumber utama `packet` 147 dan `tray` 51. Dua kelas itu tidak punya bahan yang pasti.
- Kesalahan terstruktur dan berasal dari nama cluster. Metode noisy label tidak bisa membalik nama cluster yang salah, karena noise ini konsisten dalam ruang fitur.
- Kesimpulan: urutan kerja harus **tambah informasi nama, baru agregasi, baru pembersihan**. Memulai dari deep clustering atau DivideMix hanya mengatur ulang sinyal yang sama.

## Peringkat metode

Metode 1 sampai 4 membentuk label, metode 5 sampai 7 membentuk model akhir, metode 8 opsional. Peringkat di bawah adalah peluang kontribusi terhadap 0,95. Perkiraan kenaikan adalah perkiraan saya, bukan hasil uji.

### 1. Fusi objek dan material dengan himpunan kandidat

**Algoritma**. Head objek memberi himpunan kandidat, model material memilih di antaranya. `bottle` menjadi {glass, plastic}, `packet` menjadi {plastic, paper, metal}, `tray` menjadi {plastic, metal, paper, foodwaste}. Posterior per citra: `p(y|x) ∝ [Σ_s p_obj(s|x) C(s,y)] × [Σ_m p_mat(m|x) D(m,y)]`, dengan `C` matriks kandidat kelas objek dan `D` matriks material ke target. Kelas yang tidak bermaterial tunggal seperti `electronic` hanya diambil dari faktor objek, lewat mixture.

**Angka kasus serupa**. Akurasi patch MINC: Paper 81,8%, Glass 78,5%, Metal 77,7%, Plastic 70,9%, Food 90,4%. DMS per piksel: Food 0,868, Fabric 0,801, Paper 0,729, Glass 0,653, Metal 0,575, Cardboard 0,570, Non-clear plastic 0,519, Clear plastic 0,360. Tidak ada paper yang menguji kombinasi ini pada sampah.

**Sumber**. arXiv 1412.0623, ECCV 2022 Upchurch dan Niu (https://github.com/apple/ml-dms-dataset), https://hangzhang.org/PyTorch-Encoding/tutorials/texture.html.

**Langkah**:

1. Tulis `C` dan `D` sebagai berkas versi, dibekukan sebelum dijalankan, dari WordNet dan taksonomi. Jangan diubah berdasarkan temuan.
2. Jalankan Deep TEN MINC dengan sliding crop 3 skala, ambil rata-rata probabilitas.
3. Jalankan DMS46 pada 512 px, hitung porsi piksel tiap material pada area objek. Area objek dari SAM2 atau bobot Gaussian tengah. Buang kelas latar seperti wood, tile, concrete, carpet, sky, foliage, skin.
4. Pakai ulang `temp/exp_005_imagenet_map/output/imagenet_train.npy` untuk faktor objek.

**Model**. Cache: ConvNeXtV2 huge, ConvNeXtV2 large 384, Swin-L 384, EfficientNetV2-L, SAM2. Unduh: DMS46 173,7 MB, Deep TEN MINC 94,2 MB.

**Risiko**. Skala citra sampah beda dari data latih. DMS menandai layar sebagai `glass`, jadi `electronic` jangan dibaca dari DMS. `glass` lawan `plastic` tetap sulit.

**Status**. Boleh. Rule 6 mengizinkan pretrained yang dijalankan sendiri. Data MINC dan DMS tidak dipakai untuk melatih, jadi rule 4 tidak terlanggar. Keduanya tidak khusus sampah.

**Perkiraan**: paling berpengaruh pada 24 kesalahan `paper`.

### 2. Label model EM di tingkat micro-cluster, pemetaan dipelajari dari data

**Algoritma**. Gabungan Dawid-Skene dan Naive Bayes semi-supervised. Item adalah micro-cluster `g`, sumber `m` adalah keluarga head, dan pemetaan `θ^m(s|y)` dipelajari, bukan ditetapkan.

```
E: log q_g(y) = log π_y + (1/|g|) Σ_{i∈g} Σ_m w_m Σ_s r_i^m(s) log θ^m(s|y)
M: θ^m(s|y) ∝ α·C^m(s,y) + ε + Σ_g q_g(y) Σ_{i∈g} r_i^m(s)
```

`C` hanya menjadi prior pseudo-count, sehingga `packet` bisa jatuh ke `paper` bila kontennya mendukung. Faktor `1/|g|` mencegah overconfidence karena citra dalam satu cluster berkorelasi. `ε` kecil mengizinkan label di luar kandidat, sejalan dengan PiCO+. Beberapa head ImageNet dihitung satu sumber karena tidak independen, sebab asumsi independen bersyarat Dawid-Skene memang sering gagal (arXiv 1510.05830).

**Angka kasus serupa**. EM plus Naive Bayes menurunkan error hingga 30% (Nigam dkk. 2000). MeTaL naik rata-rata 9,5 poin di atas labeling function pada dua tugas (arXiv 1810.02840). Benchmark 17 metode menyimpulkan tidak ada metode truth inference yang selalu menang (Zheng dkk. PVLDB 2017), jadi model sederhana sudah wajar. BLM (arXiv 2410.24018) membuktikan pemetaan probabilistik lebih baik dari satu-ke-satu, tetapi butuh label, jadi di sini kuantitas Bayes-nya diganti posterior EM. Saya tidak menemukan uji pada classifier pretrained tanpa label.

**Langkah**:

1. Pakai embedding DINOv3 ViT-L dari EXP-006, PCA 128.
2. Buat micro-cluster dengan KMeans K 128 dan 256, Agglomerative, dan BIRCH. Ambil konsensus, citra yang tidak disepakati masuk cadangan. Ide voting ini dari DECMCV.
3. Paksa 192 group citra mirip (cos >= 0,9) tetap satu.
4. Jalankan EM 30 iterasi dengan implementasi numpy sekitar 60 baris. Paket `snorkel` dan `cleanlab` belum terpasang, dan tidak wajib.
5. Citra train saja, test tidak ikut clustering (sesuai EXP-006).

**Model**. Cache: DINOv3 ViT-L, opsional H+ 3,2 GB. Tidak perlu unduhan baru.

**Risiko**. Hasil EM bisa terjebak bila `C` lemah. Micro-cluster yang tidak murni akan dipaksa satu nama, jadi kelompok berentropi tinggi harus dipecah lagi.

**Status**. Boleh. Tanpa teks, tanpa label, otomatis.

### 3. Head bernama dari Open Images dan ImageNet-12k/22k

**Algoritma**. Tambah kosakata sumber. Head 1k tidak punya `paper`, `tin can`, `beer can`, `soda can`, `tinfoil`, `paper bag`, `paper cup`, `printed circuit`, atau `fabric`. Head 12k punya semua itu. Open Images V2 punya `Paper`, `Cardboard`, `Glass bottle`, `Plastic bottle`, `Aluminum can`, `Tin can`, `Electronics`, `Waste`. YOLOv8 OIV7 memberi `Tin can`, `Plastic bag`, `Mobile phone`, `Laptop` tingkat kotak. Semuanya masuk ke metode 2 sebagai sumber baru.

**Angka kasus serupa**. Audit lokal menunjukkan semua lemma di atas ada di 12k. Hitungan regex per kategori di `audit_21k_kandidat.tsv` memuat salah cocok seperti `skyscraper` dan `dumpling`, jadi hanya dipakai sebagai daftar kandidat, bukan ukuran cakupan. Kekurangan: `plastic`, `cardboard`, dan `metal` generik tidak ada sebagai lemma di 21k. TAGLETS (arXiv 2111.04798) memakai ConceptNet plus ImageNet-21k dan mencapai 70,9% lawan 49 sampai 60% pada OfficeHome Product 1-shot, tetapi butuh label dan data 21k, jadi hanya gagasannya yang bisa dipakai.

**Sumber**. `temp/research_unsup/output/audit_21k_kandidat.tsv`, `docs/research_material.md`.

**Langkah**:

1. Unduh `convnext_large.fb_in22k`, `swin_large_patch4_window12_384.ms_in22k`, `tf_efficientnetv2_l.in21k`, `convnext_small.in12k`.
2. Jangan menyusun indeks kelas sendiri. Urutan berbeda per model: `imagenet-22k`, `imagenet-22k-ms`, `imagenet-21k-goog`, `imagenet-12k`. Pakai `timm.data.infer_imagenet_subset` dan `ImageNetInfo`.
3. Pasang korpus WordNet NLTK untuk closure hipernim, karena belum tersedia.
4. Pakai gloss hanya untuk menangkap jebakan nama seperti `plate` dan `cup`, bukan pemetaan utama.

**Model**. Unduhan: ConvNeXt-L 22k 919 MB, Swin-L 22k 929 MB, EfficientNetV2-L 21k 583 MB, ConvNeXt-S 12k 234 MB, YOLOv8x OIV7 138 MB, OIV2 194 MB. Cache hanya punya varian `ft_in1k`. Tidak ada env dengan TensorFlow, jadi OIV2 prioritas terakhir.

**Risiko**. Bobot OI V6 ASL 403 dan kelas `Plastic` generik hanya ada di V6. Keanggotaan MID dicek lewat pembacaan model ringkas, jadi ulangi dengan script. Kelas 21k halus bisa saja lemah pada citra sampah. Head dari keluarga ImageNet berkorelasi dan tidak boleh dihitung sebagai suara independen.

**Status**. Boleh. ImageNet 12k/22k jelas diizinkan user. Open Images dan YOLO lewat rule 6.

### 4. Optimal transport berimbang lemah untuk memutus limpahan `plastic`

**Algoritma**. Ganti argmax dengan transportasi massa dari micro-cluster ke kelas, dengan marginal kelas `π` yang dikendalikan KL, bukan persamaan keras. SeLa memakai equipartition keras, P2OT melonggarkannya untuk data imbalanced.

**Angka kasus serupa**. SeLa dan P2OT (arXiv 1911.05371, 2401.09266), tanpa angka pembanding yang relevan terbaca. Pada label shift, likelihood konkaf dan EM mencapai maksimum global, dan kalibrasi berbias per kelas mengungguli BBSL dan RLLS (arXiv 1901.06852).

**Langkah**:

1. Marginal sumber adalah ukuran micro-cluster, cost adalah `-log q_g(y)`.
2. Sinkhorn tak seimbang dengan penalti KL pada marginal kelas.
3. Prior `π` diduga dengan EM label shift tanpa label, lalu dibandingkan dengan seragam.
4. BCTS butuh label validasi, jadi diganti koreksi bias per kelas dari agreement antar sumber.

**Model**. Tanpa model baru.

**Risiko**. Bukti seragam hanya tidak langsung: SampleSubmission berisi tepat 60 per kelas, dan 2.400 = 8 × 300, tetapi host menyebut ada imbalance. Prior harus didasarkan pada SampleSubmission dan EM. Catatan EXP-007 juga memuat distribusi temuan, dan angka itu tidak boleh menjadi dasar, sesuai `docs/lock.md`.

**Status**. Boleh. Kekuatan penalti KL perlu keputusan user karena menyangkut asumsi prior.

### 5. Partial label learning untuk fine-tune DINOv3

**Algoritma**. PRODEN atau PiCO. Label latih berupa himpunan kandidat berbobot, dan model memperbarui bobot selama training. Mengganti label keras EXP-006 yang memaksakan nama pada citra ragu-ragu.

**Angka kasus serupa**. PiCO 93,58% pada CIFAR-10 q = 0,5, supervised 94,91%, dan tanpa disambiguasi 84,50%. PRODEN turun sekitar 10,6% bila 20% kandidat tidak memuat label benar, jadi kandidat dari metode 2 harus longgar atau memakai PiCO+. Label acak CIFAR berbeda dari kasus ini, jadi angka ini batas atas yang optimistis.

**Sumber**. arXiv 2201.08984, Lv dkk. ICML 2020.

**Langkah**:

1. Target awal `q_g(y)` dari metode 2 dan 4, nol di luar 2 sampai 3 kelas teratas.
2. Pakai ulang `temp/exp_006_cluster_dinov3/train_dino.py` dengan loss soft dan pembaruan target gaya PRODEN.
3. Validasi hold-out group yang sudah terkunci.

**Model**. Cache DINOv3 ViT-L, 10,91 GiB reserved menurut EXP-006.

**Risiko**. Skor hold-out hanya mengukur kesepakatan, bukan kebenaran.

**Status**. Boleh. Train saja.

### 6. SHOT-IM dari head 8 kelas hasil fusi

**Algoritma**. Bekukan head 8 kelas, sesuaikan fitur dengan minimasi entropi per citra dan maksimasi keragaman marginal, lalu centroid pseudo-label dengan β = 0,3.

**Angka kasus serupa**. ImageNet ke Caltech 1.000 ke 84 kelas: source only 69,7%, SHOT-IM 81,7%, SHOT 83,3%, ETN yang memakai data sumber 83,2%. Paling mirip struktur masalah kita, tetapi target Caltech adalah objek ImageNet. Office-Home partial 79,3%.

**Sumber**. arXiv 2002.08546.

**Langkah**. Awali dari model metode 5. `L_div` dipertahankan karena kita 8 kelas tertutup, berbeda dari partial-set di paper. Jalankan pada train saja.

**Risiko**. Entropi bisa mengunci kesalahan yang sudah ada. Ini memperkuat, bukan mengoreksi nama.

**Status**. Boleh di train. Varian pada test (SHOT, T3A, LAME) berarti memakai test saat adaptasi dan bertentangan dengan izin lock saat ini.

### 7. Label spreading dan regularisasi noisy label

**Algoritma**. Seed dari micro-cluster yang sangat yakin dan disepakati banyak sumber, lalu label spreading pada graf kNN DINOv3, logistic cross-fit untuk membuang label meragukan, kemudian ELR saat fine-tune.

**Angka kasus serupa**. Noise simetris CIFAR-10: ELR 95,8 sampai 93,3 dan DivideMix 95,0 sampai 92,4 pada 20 sampai 80%. Instance-dependent 50% CIFAR-100: DivideMix 57,00%. USB dengan ViT pretrained: FlexMatch error 7,07% pada EuroSAT 20 label, FixMatch 12,48%. Label propagation paling membantu saat label sedikit (arXiv 1904.04717).

**Sumber**. arXiv 2007.00151, 2412.04898, 2308.06861, 2208.07204.

**Langkah**. Seed `q >= 0,9` dengan minimal 3 sumber setuju, k = 10. Cross-fit 5 fold dengan group. Ambang dipilih dari stabilitas, bukan temuan.

**Risiko**. Noise bergantung cluster belum punya benchmark. DivideMix dan co-teaching tidak disarankan sebagai langkah utama, karena small-loss tidak memisahkan noise yang konsisten secara visual.

**Status**. Boleh. `cleanlab` perlu instal bila dipakai.

### 8. TURTLE tanpa CLIP sebagai suara partisi independen

**Algoritma**. Cari labeling yang linear-separable dengan margin besar di beberapa ruang representasi. Nama dari Hungarian atas matriks massa weak label, bukan ground truth. Masuk metode 2 sebagai sumber tambahan.

**Angka kasus serupa**. TURTLE 1 ruang DINOv2: CIFAR10 99,3, Flowers 99,0, CIFAR100 87,1, Food101 78,9, DTD 55,2. Supervised DINOv2: 99,5, 99,7, 93,9, 94,9, 85,0. Kelas tekstur jatuh jauh. Hungarian TURTLE memakai ground truth, jadi angka itu kualitas partisi saja. SCP (arXiv 2502.04226) adalah alternatif tanpa teks yang tidak sempat diverifikasi angkanya.

**Langkah**. Ruang: DINOv3 ViT-L, H+, ConvNeXt-L, ConvNeXtV2. C = 8 dan 16. M = 10, T = 6.000, γ = 10. Pilih dari 50 konfigurasi dengan CV pada pseudo label.

**Risiko**. Material berperilaku seperti DTD, partisi bisa mengikuti bentuk dan latar.

**Status**. Boleh hanya dengan ruang non-CLIP. Konfigurasi 2 ruang bawaan repo memakai CLIP dan TIDAK BOLEH.

## Pipeline yang disarankan

```
Sumber bukti (1, 3)      Micro-cluster DINOv3      Label model EM (2)
objek: IN-1k, 12k/22k  ->   konsensus + group   ->  pemetaan dipelajari
material: MINC, DMS46                               prior dari C dan D
                                                         |
Model akhir (5, 6, 7) <- seed dan label kandidat <- OT berimbang lemah (4)
```

Urutan uji murah ke mahal, tiap tahap satu eksperimen:

1. Unduh DMS46 dan Deep TEN MINC, ekstrak probabilitas train dan test (ResNet-50 dengan VRAM kecil).
2. Label model pada train, ukur distribusi dan agreement antar sumber.
3. Tambah OT, lalu latih DINOv3 hold-out seperti EXP-006.
4. Bila membaik, tambah head 22k dan YOLO.
5. Baru setelah itu PLL dan SHOT.

Validasi tanpa label, supaya temuan tetap hanya untuk skor akhir:

- Leave-one-source-out: sembunyikan satu sumber, ukur kecocokan hasil gabungan dengan sumber itu pada kelas kuatnya.
- Stabilitas: ARI hasil label antar seed clustering dan antar K.
- Generalisasi: akurasi logistic cross-fit group pada label yang dihasilkan.

## Hasil verifikasi kepatuhan model cache

| Model | Bobot | Status | Alasan |
|---|---|---|---|
| DINOv3 ViT-S/B/L/H+/7B, ConvNeXt S/L | ada | Boleh | self-supervised, citra saja |
| `vit_large_patch16_dinov3.sat493m` | ada | Boleh, tidak berguna | citra satelit |
| ConvNeXtV2 huge/large/base, Swin-L, EfficientNetV2 | ada, head 1k | Boleh | FCMAE atau supervised ImageNet |
| DINOv2 large/small | ada | Boleh | self-supervised |
| SAM2 Hiera-L | ada | Boleh | tanpa text encoder |
| EoMT DINOv3 ADE/COCO | ada | Boleh | segmentasi semantik citra saja |
| LingBot-Vision | ada `.pt` | Boleh, laporan teknis belum dibaca | SSL gaya DINO dari inisialisasi acak |
| `eva02_large_patch14_448.mim_*` | tidak lengkap | **TIDAK BOLEH** | nama tanpa `clip`, tetapi target MIM adalah fitur EVA-CLIP (arXiv 2303.11331, kartu model timm) |
| C-RADIOv4-H/SO400M | ada | **TIDAK BOLEH** | distilasi dari SigLIP2 (arXiv 2601.17237) |
| AIMv2 | ada | **TIDAK BOLEH** | decoder ikut merekonstruksi caption (arXiv 2411.14402) |
| SAM3 | ada | **TIDAK BOLEH** | model dengan prompt teks, tidak diverifikasi lebih lanjut |
| SteerViT | ada | **TIDAK BOLEH** | cross-attention ke text encoder RoBERTa (arXiv 2604.02327) |
| SigLIP2, semua `*_clip_*`, `waste_siglip_clean`, Augmented-Waste-Classifier-SigLIP2 | ada | **TIDAK BOLEH** | VLM, yang terakhir dilatih khusus sampah |

Filter nama `clip` di `docs/rule.md` tidak cukup: EVA-02 MIM, C-RADIO, AIMv2, dan SteerViT lolos filter itu.

## Metode yang dicoret

| Metode | Alasan |
|---|---|
| TAC, SIC, TeDeSC, clustering berbantuan teks | butuh text encoder citra-teks. TIDAK BOLEH |
| TURTLE 2 ruang CLIP + DINOv2 | ruang CLIP dilarang |
| RAM, Tag2Text, Detic, OWL-ViT, Grounding DINO, SAM3, dino.txt | berbasis teks. TIDAK BOLEH |
| EVA-02, C-RADIO, AIMv2, SteerViT | backbone turunan CLIP, SigLIP, atau teks. TIDAK BOLEH |
| FMMC (arXiv 2603.17390) versi penuh | CLIP plus GPT-4V. TIDAK BOLEH |
| `prithivMLmods/Minc-Materials-23` | classifier SigLIP. TIDAK BOLEH |
| DECMCV utuh | bagian 3.1 menyatakan "the clustered data undergoes manual labeling". Angka 93,78% bukan hasil tanpa label |
| BLM utuh | butuh label target |
| TAGLETS utuh | butuh label dan data ImageNet-21k |
| GCN-ZSL dan DGP WordNet | top-1 26,6% pada 2-hops, dan butuh word embedding. Tidak perlu karena head 12k/22k sudah memberi kelas langsung |
| SCAN, ProPos, SPICE, RUC | melatih encoder dari nol pada 1.920 citra, angka CIFAR-10 88 sampai 94 dengan Hungarian ground truth. Tidak menyelesaikan penamaan |
| TEMI | naik 6,1 dan 12,2 poin di atas KMeans, tetapi hanya memperbaiki partisi |
| DivideMix sebagai langkah utama | noise kita konsisten per cluster |
| Melatih model pada FMD, MINC, DMS, atau data sampah | rule 4 |
| Augmentasi Stable Diffusion | generative AI, rule 9 |
| Bobot TResNet Open Images V6 | 403, bucket `miil-public-eu` tidak bisa diakses |
| Tencent ML-Images | Google Drive, TF1, Python 2.7, biaya tinggi. `cardboard` tidak terlihat pada bagian daftar yang terbaca (sekitar 60% berkas) |
| Pemetaan otomatis murni dari gloss | 72 dari 1.000 kelas saja, gloss botol tidak menyebut bahan |

## Hasil uji pemetaan otomatis dari gloss WordNet

Script: `temp/research_unsup/audit_gloss_map.py`, hasil `temp/research_unsup/output/gloss_map.tsv`.

- Head 1k: 72 dari 1.000 kelas dapat kandidat. Head 12k: 1.067 dari 11.821.
- Dari 232 entri PETA EXP-005: 51 cocok, 177 tanpa kandidat, 4 berbeda.
- Gloss `beer bottle` adalah "a bottle that holds beer". WordNet tidak mencatat bahan botol, jadi `glass` lawan `plastic` tidak bisa diturunkan dari gloss.
- Gloss berguna untuk jebakan nama: `plate` ternyata hidangan, `cup` ternyata minuman punch.
- Regex yang dipakai sederhana, sehingga cakupan bisa dinaikkan, tetapi masalah botol tetap ada.

## Asal data

Tidak ditemukan dataset publik dengan 8 kelas persis ini. Query yang dicoba: kombinasi 8 nama kelas di web umum, Roboflow Universe, dan Kaggle, serta daftar `AgaMiko/waste-datasets-review`.

| Kandidat | Kemiripan | Angka |
|---|---|---|
| RealWaste | Cardboard, Food Organics, Glass, Metal, Paper, Plastic, Textile Trash, plus 2 kelas lain. 7 dari 8 nama cocok, tanpa elektronik | 4.752 citra 524×524, Inception V3 89,19% |
| TrashBox | E-Waste, Plastic, Paper, Metal, Glass, Cardboard, Medical | 17.785 citra hasil scraping |
| Garbage V2 | Metal, Glass, Biological, Paper, Battery, Trash, Cardboard, Shoes, Clothes, Plastic | 19.762 citra |
| Roboflow Garbage Classification 3 | PAPER, PLASTIC, GLASS, METAL, CARDBOARD, CLOTH, BIODEGRADABLE | tidak dicek |

Hipotesis EDA, belum diuji:

1. Median test 416 dan kuartil atas 640 mengarah ke ekspor Roboflow. Cukup diuji dari metadata ukuran.
2. Data gabungan dari beberapa sumber. Ukuran citra, tabel kuantisasi JPEG, dan subsampling bisa membentuk kelompok sumber. Hanya metadata, tanpa membuka citra.
3. 784 citra mirip dalam 192 group, terbesar 60, kemungkinan augmentasi atau duplikasi sumber.

Data ini hanya untuk memahami karakter, tidak boleh dipakai untuk training.

## Pertanyaan terbuka untuk user

1. Izin unduh: DMS46 dan Deep TEN MINC (total sekitar 268 MB), lalu head 22k dan YOLO.
2. Apakah bobot Open Images dan YOLO OIV7 diterima, karena allow-list eksplisit hanya menyebut ImageNet dan DINOv3.
3. Apakah prior seragam dari SampleSubmission boleh dipakai sebagai asumsi lunak.
4. Apakah metadata JPEG dan ukuran boleh dipakai sebagai sinyal, karena bisa dianggap memanfaatkan artefak penggabungan data.
5. Apakah `C` dan `D` boleh ditulis sebagai taksonomi kelas sumber oleh asisten, selama dibekukan sebelum dijalankan. Tabel ini memetakan nama kelas, bukan data kompetisi, tetapi tetap perlu persetujuan.

## Sumber

| Topik | Rujukan |
|---|---|
| TURTLE | arXiv 2406.07236, https://github.com/mlbio-epfl/turtle |
| TEMI | arXiv 2303.17896 |
| SCP | arXiv 2502.04226 |
| TAC | arXiv 2310.11989 |
| SeLa, P2OT | arXiv 1911.05371, 2401.09266 |
| SHOT | arXiv 2002.08546 |
| Label shift EM | arXiv 1901.06852 |
| BLM | arXiv 2410.24018 |
| Dawid-Skene dependen | arXiv 1510.05830, 2503.07119 |
| Snorkel MeTaL | arXiv 1810.02840 |
| Truth inference | PVLDB 10, 541-552, https://www.vldb.org/pvldb/vol10/p541-zheng.pdf |
| NB-EM | Nigam dkk., Machine Learning 39, 2000 |
| PiCO, PiCO+ | arXiv 2201.08984 |
| ELR, noisy label | arXiv 2007.00151, 2412.04898, 2308.06861 |
| Label propagation | arXiv 1904.04717 |
| USB | arXiv 2208.07204 |
| LAME | arXiv 2201.05718 |
| TAGLETS | arXiv 2111.04798 |
| DGP | arXiv 1805.11724 |
| DECMCV | arXiv 2503.02241 |
| Ridnik 21K | arXiv 2104.10972 |
| RealWaste | doi 10.3390/info14120633 |
| Kepatuhan model | arXiv 2303.11331, 2601.17237, 2411.14402, 2604.02327 |

## Hasil uji pada embedding

Folder: `temp/research_uji_crossfit/`. CPU saja, 4 thread, tanpa `data/temuan/`, tanpa unduhan, test hanya diprediksi.

### Uji 1: cross-fit berkelompok satu pandangan (5 kali 5 fold, 1.626 label)

| Metode | Setuju label | Pinggir (suara < 0,95, n = 381) | Setuju EoMT | Stabil antar split |
|---|---|---|---|---|
| LR keras | 0,9903 | 0,9690 | 0,8598 | 0,9953 |
| kNN | 0,9895 | 0,9591 | 0,8550 | 0,9914 |
| Label spreading | 0,9814 | 0,9234 | 0,8545 | 0,9945 |
| LR soft label | 0,9872 | 0,9559 | 0,8587 | 0,9947 |
| LR soft + 294 sisa | 0,9859 | 0,9549 | 0,8604 | 0,9937 |
| PRODEN + 294 sisa | 0,9817 | 0,9354 | 0,8584 | 0,9925 |
| Pandangan EoMT | 0,8605 | 0,7554 | sama dengan dirinya | 0,9296 |

Selisih terhadap LR keras pada pinggir, CI 95% bootstrap group: label spreading -0,0457 (-0,0780 sampai -0,0152), PRODEN -0,0336 (-0,0537 sampai -0,0164), LR soft -0,0131 (-0,0247 sampai -0,0025), kNN -0,0100 (-0,0317 sampai 0,0115, tidak signifikan).

Metrik ini melingkar dan condong ke LR keras, sebab label lahir dari ruang yang sama. Soft label memang tidak berusaha meniru label keras. Kesimpulan metode diambil dari uji injeksi, bukan dari tabel ini.

### Uji 2: injeksi noise, ukuran utama yang tidak melingkar

2 ulangan kali 5 fold per sel. Noise hanya di bagian latih, hold-out 20% group dinilai terhadap label EXP-009. Head linear di atas embedding beku, bukan fine-tune.

| Noise | ViT | CNN | ViT+CNN | Saring | Silang | Soft OOF | SGD dasar | Co-teach 0,1 | Co-teach 0,2 |
|---|---|---|---|---|---|---|---|---|---|
| bersih | 0,9896 | 0,9739 | 0,9853 | 0,9846 | 0,9853 | 0,9819 | 0,9884 | 0,9806 | 0,9737 |
| acak 5% | 0,9855 | 0,9708 | 0,9844 | 0,9842 | 0,9842 | 0,9801 | 0,9731 | 0,9845 | 0,9792 |
| acak 15% | 0,9827 | 0,9577 | 0,9815 | 0,9796 | 0,9790 | 0,9798 | 0,9398 | 0,9691 | 0,9803 |
| bingung 5% | 0,9858 | 0,9723 | 0,9835 | 0,9835 | 0,9829 | 0,9814 | 0,9703 | 0,9849 | 0,9807 |
| bingung 15% | 0,9812 | 0,9629 | 0,9814 | 0,9832 | 0,9782 | 0,9814 | 0,9359 | 0,9619 | 0,9855 |
| cluster 5% | 0,9445 | 0,9295 | 0,9433 | 0,9438 | 0,9452 | 0,9473 | 0,9296 | 0,9548 | **0,9615** |
| cluster 15% | 0,8821 | 0,8682 | 0,8754 | 0,8819 | 0,8816 | 0,8866 | 0,8554 | 0,8889 | **0,9191** |

Pemulihan kesepakatan silang, ambang 0,8 per ronde, maksimal 3 ronde:

| Noise | Kotor | Diubah | Pulih | Merusak | Recall | Presisi |
|---|---|---|---|---|---|---|
| acak 5% | 64,5 | 44,2 | 44,2 | 0,0 | 0,685 | 1,000 |
| bingung 5% | 64,5 | 42,2 | 42,1 | 0,1 | 0,653 | 0,998 |
| cluster 5% | 71,0 | 8,7 | 8,6 | 0,1 | 0,121 | 0,989 |
| cluster 15% | 199,5 | 22,6 | 22,4 | 0,2 | 0,112 | 0,991 |

Pelajaran:

- Noise per citra hampir tidak merugikan LR, sebab LR sudah menahannya sendiri. Masalah sesungguhnya adalah noise cluster: 15% noise cluster menurunkan akurasi 0,9896 ke 0,8821.
- Noise cluster cuma pulih 11 sampai 12% oleh kesepakatan silang. Ini bukti eksperimen untuk dugaan bahwa cross-fit tidak bisa membalik nama cluster yang salah.
- Rata-rata ViT dan CNN tidak menambah ketahanan noise cluster (0,8754 lawan ViT 0,8821), walau noise dibuat di ruang ViT.
- Co-teaching lupa 0,2 satu-satunya yang tahan, +4,4 poin atas ViT+CNN pada cluster 15%, tetapi -1,2 poin pada data bersih. Laju lupa harus cocok dengan noise asli yang tidak diketahui. Lupa 0,1 pada noise 15% lebih buruk daripada LR.

### Uji 3: fusi ViT dan CNN (5 kali 5 fold)

| Fusi | Setuju label | Pinggir | NLL | Setuju EoMT |
|---|---|---|---|---|
| ViT | 0,9903 | 0,9690 | 0,1542 | 0,8598 |
| CNN | 0,9724 | 0,9160 | 0,1956 | 0,8541 |
| Concat | 0,9843 | 0,9454 | 0,1529 | 0,8594 |
| Rata-rata prob | 0,9849 | 0,9480 | 0,1671 | 0,8595 |
| Rata-rata logit | 0,9846 | 0,9470 | 0,1544 | 0,8595 |
| Rata-rata prob + suhu | 0,9860 | 0,9522 | 0,0475 | 0,8597 |
| Geometrik + suhu | 0,9860 | 0,9522 | **0,0414** | 0,8597 |
| Stacking bersarang | **0,9882** | 0,9638 | 0,0589 | 0,8598 |

Suhu optimal ViT sekitar 0,33 dan CNN 0,47, jadi LR C = 3 kurang yakin dan CNN lebih tidak yakin lagi. Rata-rata tanpa suhu memberi bobot tidak adil. Kesepakatan dengan label condong ke ViT karena label lahir di ruang ViT, jadi keunggulan ViT di sini sebagian artefak. Rata-rata 35,6 citra OOF berbeda antara ViT dan CNN: 31,6 ViT mengikuti label, 2,6 CNN. Saat berbeda, EoMT mendukung ViT 19,4 dan CNN 10,2. Pada test, ViT dan CNN berbeda di 34 citra (setuju 0,9292).

### Uji 4: mengisi 294 citra dan OT

Saat 381 citra pinggir disembunyikan dan model dilatih hanya pada label yakin, kesepakatan pada pinggir turun ke 0,7743 sampai 0,8373. Angka 0,969 di uji 1 karena itu terutama berasal dari tetangga pada KMeans yang sama.

| Metode | Ambang | Cakupan pinggir | Setuju pinggir | Terisi dari 294 | Setuju fine-tune |
|---|---|---|---|---|---|
| ViT + CNN rata-rata | 0,8 | 0,415 | 0,9430 | 36 | 1,0000 |
| ViT + CNN rata-rata | 0,7 | 0,549 | 0,9139 | 60 | 1,0000 |
| Dua pandangan sepakat | 0,8 | 0,307 | 0,9573 | 15 | 1,0000 |
| Dua pandangan sepakat | 0,5 | 0,646 | 0,8902 | 89 | 0,9551 |
| PRODEN | 0,8 | 0,417 | 0,9623 | 57 | 0,9649 |
| PRODEN | 0,7 | 0,585 | 0,9507 | 91 | 0,8681 |
| Label spreading | 0,8 | 0,299 | 0,9298 | 44 | 0,6364 |

Stabilitas prediksi 294 antar 10 bootstrap group: ViT+CNN 0,9640, PRODEN 0,9478, CNN 0,9389, ViT 0,9355, LR soft 0,9255, label spreading 0,8922. Test: ViT+CNN 0,9859, ViT 0,9823, CNN 0,9820.

OT versi pertama saya salah: semua `rho` konvergen ke keseimbangan penuh, tabelnya dibuang (`output/uji4_ot_BUG_dibuang.csv`). Versi benar di `uji4b_ot.py`. Pada `rho` 5, 9 prediksi test berubah, 5 mendekati fine-tune dan 2 menjauh. Pada keseimbangan penuh, 17 berubah, 6 mendekat dan 8 menjauh. Plastic test turun dari 52 ke 47, menjauhi 60, kemungkinan karena marginal train memuat 294 sisa yang tidak seimbang, tetapi penyebab ini belum diuji. OT tidak terbukti memperbaiki.

### Uji 5: ensemble partisi

| Varian | Label >= 0,75 | ARI antar blok seed | Setuju antar blok | Citra baru setuju fine-tune |
|---|---|---|---|---|
| ViT (replika t1) | 1.612 | 0,9367 | 0,9734 | 0,8276 |
| CNN | 1.727 | 0,9437 | 0,9755 | 0,8125 |
| ViT + CNN | 1.609 | **0,9838** | **0,9932** | **0,9314** |
| Consensus K=64 | 1.920 | 0,9168 | 0,9646 | 0,6224 |
| Consensus K=96 | 1.920 | 0,9255 | 0,9672 | 0,6497 |

Bagging gabungan ViT dan CNN adalah perbaikan paling jelas pada stabilitas label. Replika ViT dengan seed yang sama menghasilkan 1.612 label, bukan 1.626, dan seluruh argmax sama. Ambang 0,75 sensitif terhadap ketidakdeterministikan KMeans lintas thread, dan himpunan label antar blok hanya sama 0,9406. Agglomerative atas co-association lebih buruk daripada voting.

### Uji 6: co-teaching sebagai detektor

Skema: noise pada 1.626 label, OOF berkelompok 5 fold, koreksi bila dua head sepakat pada kelas lain dengan p >= tau. 3 ulangan.

| Noise | Detektor | tau | Kotor | Diubah | Pulih | Presisi | Recall |
|---|---|---|---|---|---|---|---|
| cluster 5% | LR | 0,8 | 91,0 | 3,0 | 3,0 | 1,00 | 0,03 |
| cluster 5% | co-teach 0,1 | 0,9 | 91,0 | 44,3 | 41,3 | 0,93 | 0,45 |
| cluster 5% | co-teach 0,2 | 0,9 | 91,0 | 59,7 | 57,3 | 0,96 | 0,63 |
| cluster 15% | LR | 0,8 | 257,7 | 16,0 | 16,0 | 1,00 | 0,06 |
| cluster 15% | co-teach 0,2 | 0,9 | 257,7 | 125,7 | 111,0 | 0,88 | 0,43 |
| bingung 5% | LR | 0,8 | 81,0 | 49,0 | 48,7 | 0,99 | 0,60 |
| bingung 5% | co-teach 0,2 | 0,9 | 81,0 | 75,7 | 72,7 | 0,96 | 0,90 |

Ensemble 3 seed tidak lebih baik dari satu seed pada tau 0,9. Probabilitas LR kurang yakin (suhu 0,33), jadi tau tidak setara antar detektor, tetapi LR pada tau 0,5 pun hanya recall 0,24 dengan presisi 0,84 pada cluster 5%.

### Uji 7: validasi aturan beku

Kontrol, satu model, pengoptimal dan epoch sama: recall cluster 5% adalah 0,084 pada lupa 0 dan 0,589 pada lupa 0,2. Cluster 15%: 0,083 dan 0,450. Mekanisme lupa memang penyebabnya.

Aturan utuh, `deteksi.py` (lupa 0,2, R 5, S 3, TOLAK 0,9, YAKIN 0,9), seed noise baru, 2 ulangan:

| Noise | Kotor | Koreksi | Pulih | Presisi koreksi | Recall koreksi | Buang | Presisi buang |
|---|---|---|---|---|---|---|---|
| asli tanpa injeksi | 0 | 4,5 | tidak diketahui | tidak diketahui | - | 13,0 | tidak diketahui |
| acak 5% | 81 | 73,0 | 71,0 | 0,973 | 0,877 | 22,5 | 0,444 |
| bingung 5% | 81 | 75,0 | 71,5 | 0,953 | 0,883 | 18,5 | 0,405 |
| cluster 3% | 51 | 36,0 | 34,5 | 0,958 | 0,676 | 26,0 | 0,423 |
| cluster 5% | 104 | 55,5 | 52,5 | 0,946 | 0,505 | 19,5 | 0,436 |
| cluster 10% | 165 | 84,0 | 81,5 | 0,970 | 0,494 | 30,0 | 0,600 |

Efek pada hold-out berlabel asli, 5 fold, ViT LR + CNN LR, dengan R 2 dan S 2:

| Noise | ViT+CNN label kotor | ViT+CNN setelah aturan | Selisih |
|---|---|---|---|
| asli | 0,9859 | 0,9836 | -0,0023 |
| bingung 5% | 0,9836 | 0,9830 | -0,0006 |
| cluster 5% | 0,9667 | 0,9730 | +0,0063 |
| cluster 10% | 0,9260 | 0,9744 | +0,0484 |

Presisi buang dan presisi koreksi pada data sintetis adalah batas bawah, karena "kena bersih" bisa saja kesalahan asli pada label EXP-009.

### Hasil aturan pada label asli

17 dari 1.626 label ditandai, 3 koreksi dan 14 buang. Berkas: `temp/research_uji_crossfit/output/daftar_label_diragukan.csv`. Kode beku `deteksi.py` SHA-1 `eaafc4012ec1bc6bca7696d4752648b4473e97fb`, cocok dengan hash sebelum daftar dibuat.

- Koreksi: 1737 glass ke plastic, 1798 glass ke plastic, 1920 paper ke cardboard.
- Buang: 0088, 0143, 0550 (glass). 0164, 0264, 0478, 0971 (paper). 0536, 0577, 0680, 1027, 1206, 1626 (metal). 1530 (plastic).
- Tidak ada label `electronic`, `cardboard`, `textile`, `foodwaste` yang ditandai.
- Stabilitas: dua paruh model OOF menandai 18 dan 17 citra, irisan 16, Jaccard 0,8421.

Audit yang tidak mengubah daftar:

| Audit | Hasil |
|---|---|
| Wasit 22k + DMS pada 3 koreksi | 3 mendukung label baru, 0 label lama |
| Wasit pada 14 buang | setuju label lama 0,1429, sedangkan rata-rata semua label 0,7263 |
| EoMT pada 3 koreksi | 3 setuju label lama, bertentangan dengan wasit |
| EoMT pada 14 buang | setuju label lama 0,50, rata-rata semua label 0,8665 |
| Keluarga berkas pada 3 koreksi | semua label baru lebih umum di keluarganya |
| Citra mirip | 2 koreksi dalam group, keduanya menuju mayoritas group |

EoMT dilatih pada label EXP-009, jadi condong membenarkannya. Jumlah 3 koreksi terlalu kecil untuk dijadikan bukti kuat.

### Uji 8: pengisian 294 dengan head co-teaching

| Ambang | Cakupan pinggir | Setuju pinggir | Terisi dari 294 | Setuju fine-tune |
|---|---|---|---|---|
| 0,7 | 0,664 | 0,8972 | 162 | 0,9691 |
| 0,8 | 0,625 | 0,9034 | 133 | 0,9850 |
| 0,9 | 0,567 | 0,9120 | 103 | 0,9903 |
| 0,95 | 0,493 | 0,9149 | 74 | 1,0000 |

Dibanding LR dua pandangan pada ambang 0,8, yang hanya mengisi 15 citra. Stabilitas antar seed 0,9667 pada sisa dan 0,9938 pada test. Sekitar 10% isian diperkirakan berbeda dari label bila perilaku pinggir berlaku pada 294.

### Uji 9: efek varian label pada prediksi test

Proksi: LR ViT dan CNN dengan fusi geometrik terkalibrasi. Test hanya diprediksi. Ini batas dampak pada head linear, bukan skor.

| Varian | Latih | Test berubah | Wasit dukung baru | Wasit dukung lama | Ke fine-tune | Dari fine-tune |
|---|---|---|---|---|---|---|
| L1 bersih | 1.612 | 3 | 2 | 1 | 2 | 0 |
| L3 bersih + isi 0,95 | 1.686 | 4 | 2 | 2 | 3 | 0 |
| L2 bersih + isi 0,9 | 1.715 | 8 | 6 | 2 | 6 | 1 |
| L4 bersih + isi 0,7 | 1.774 | 11 | **11** | **0** | 9 | 2 |
| L5 bersih + bagging gabung | 1.714 | 12 | 7 | 5 | 5 | 7 |

Uji tanda L4 p = 0,00098, tetapi 5 varian dibandingkan, jadi setelah koreksi Bonferroni p sekitar 0,005. Margin wasit 0,5 log-odds. Wasit sendiri hanya setuju 0,7263 dengan label, jadi ini bukti arah, bukan akurasi.

Pada test, LR ViT, LR CNN, dan fine-tune EXP-009 ketiganya sepakat pada 440 citra dan tidak sepakat pada 40. Saat fine-tune EXP-009 berbeda dari LR ViT, wasit mendukung fine-tune 18 lawan 5 (p = 0,011), jadi fine-tune memang menambah informasi di atas head linear.

### Uji 10: berkas label v3

`temp/research_uji_crossfit/output/label_folds_v3.csv`: 1.774 baris, 1.609 lama, 3 koreksi, 162 isian, 14 dibuang, 132 sisa tanpa label. Hold-out fold 0 berisi 384 baris lama saja. Kelas: glass 192, paper 224, electronic 183, metal 234, plastic 272, cardboard 224, textile 211, foodwaste 234.

Aturan ronde 2 tidak menolak satu isian pun. Isian berasal dari keluarga head yang sama, jadi pemeriksaan ini lemah dan bukan bukti isian benar. Ronde 2 menandai 7 baris lama, 4 di antaranya paper ke cardboard, dicatat di `output/ronde2_lama_ditandai.csv` dan sengaja tidak diubah.

## Putusan metode

| Metode | Putusan | Bukti utama |
|---|---|---|
| Co-teaching sebagai detektor cross-fit | **Memberi perbaikan terukur** | recall cluster 5% 0,63 lawan LR 0,03, hold-out +0,048 pada cluster 10% |
| Isian 294 dari head co-teaching | **Memberi perbaikan terukur, tertinggi pada test** | 11 perubahan, wasit 11 lawan 0 |
| Fusi geometrik dengan suhu | **Memberi perbaikan** pada kalibrasi | NLL 0,167 ke 0,041, kesepakatan 0,9849 ke 0,9860 |
| Bagging KMeans ViT + CNN | Stabil, manfaat label belum terbukti | ARI 0,9367 ke 0,9838, tetapi wasit test 7 lawan 5 |
| Stacking | Bagus pada metrik melingkar saja | 0,9882, condong ke ViT |
| Kesepakatan silang LR | Aman tetapi lemah | presisi 0,99, recall noise cluster 0,12 |
| Co-teaching sebagai model akhir | Tidak disarankan | -1,2 poin pada data asli, butuh laju lupa yang tidak diketahui |
| Soft label dari OOF | Tidak ada perbaikan berarti | -0,003 pada asli, +0,011 pada cluster 15% |
| Soft label dari suara, PRODEN pada cross-fit | Tidak ada perbaikan | -0,013 dan -0,034 pada pinggir |
| Label spreading | **Lebih buruk** | -0,046 pada pinggir, stabilitas 0,8922 |
| OT tak seimbang | Tidak terbukti | keseimbangan penuh: 6 mendekat, 8 menjauh |
| Consensus agglomerative | **Lebih buruk** dari voting | ARI 0,92, setuju fine-tune 0,62 |
| Rata-rata ViT dan CNN untuk ketahanan noise | Tidak menambah | cluster 15%: 0,8754 lawan ViT 0,8821 |

## Batas pengujian

- Semua hasil pada head linear di atas embedding beku. Fine-tune bisa berperilaku lain, terutama co-teaching.
- Noise cluster sintetis dibuat di ruang ViT dari satu KMeans K=96. Kesalahan asli bisa berbeda bentuk.
- Tidak ada uji di sini yang mengukur accuracy sebenarnya. Dampak pada skor harus dinilai agen utama lewat eval temuan.
- Ambang isian 0,7 dipilih setelah melihat uji 9.
- Target 0,98 belum terdukung. Label asli hanya menandai 17 citra dan L4 hanya menggeser 11 prediksi head linear, sedangkan target butuh 13 perbaikan bersih di atas EXP-009.
