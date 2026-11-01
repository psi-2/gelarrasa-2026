# Constraint — DSC Gelar Rasa 2026

Batasan wajib untuk seluruh pengerjaan. Sumber: aturan resmi di [context.md](context.md), jawaban resmi panitia di QnA Kaggle (1 Nov 2026), dan batasan internal tim. Panitia **memeriksa notebook dan laporan** di akhir lomba, dan pelanggaran berujung diskualifikasi.

## 1. Batasan internal tim (lebih ketat dari aturan panitia)

| Dilarang | Maksudnya |
|---|---|
| **Blending / ensembling prediksi** | Tidak boleh merata-rata, voting, atau stacking output beberapa model untuk submission. Submission final berasal dari **satu model**. (Rule 7 membolehkan beberapa model dalam *pipeline*, misalnya backbone + clustering + classifier, tapi bukan blending prediksi.) |
| **Semi-supervised learning** | Tidak boleh memakai citra **test** (tanpa label) dalam proses training dalam bentuk apa pun: consistency training, MixMatch/FixMatch, domain adaptation ke test, dan sejenisnya. |
| **Pseudo-labeling** | Tidak boleh memakai prediksi model pada data **test** sebagai label untuk training ulang, termasuk self-training atau iterasi relabel. |
| **Probing** | Tidak boleh memakai skor public leaderboard sebagai sinyal tuning (LB probing): menebak label test lewat submission, memilih hyperparameter atau threshold berdasarkan skor LB, atau submit varian untuk "mengintip" distribusi test. Semua keputusan model diambil dari validasi lokal. |

## 2. Data

- Hanya dataset dari panitia. **Dilarang memakai dataset eksternal** (Rule 4).
- Data train: 1.920 citra **tanpa label/ground truth**, semuanya dalam satu folder. Data test: 480 citra tanpa label.
- Data test **hanya untuk inference** akhir, tidak untuk training, clustering, normalisasi statistik, atau seleksi model.

## 3. Pelabelan train (QnA resmi)

- Label train **wajib diperoleh lewat clustering** dengan model yang dikembangkan sendiri oleh tim.
- **Penamaan cluster secara manual diperbolehkan** (misalnya "cluster 0 = textile") berdasarkan karakteristik citra.
- **Tidak boleh melabeli train secara manual tanpa clustering.** Tim yang sudah terlanjur melakukannya disarankan panitia untuk memodelkan ulang.
- Clustering hanya dilakukan pada data **train**.

## 4. Model

- **Dilarang semua VLM** (CLIP, SigLIP, BLIP, LLaVA, dll.), **termasuk hanya untuk feature extraction** atau tahap lain (QnA resmi + Rule 9).
- **Dilarang** AutoML, LLM, generative AI, atau layanan AI pihak ketiga untuk membuat model, mengklasifikasi, atau menghasilkan prediksi (Rule 9).
- **Dilarang** API/layanan pihak ketiga yang melakukan inference, klasifikasi, deteksi, atau prediksi pada data kompetisi (Rule 8).
- **Boleh** memakai pretrained weights non-VLM (misalnya ResNet/EfficientNet dari ImageNet) untuk feature extraction atau transfer learning, **asal dijalankan dan dilatih/di-fine-tune sendiri** di lingkungan pengerjaan tim (Rule 6 + QnA resmi).
- Boleh memakai library, framework, dan tools untuk preprocessing, augmentasi, analisis, training, dan evaluasi (Rules 5, 10).

## 5. Submission

- Format CSV: `image_id,waste_category`. Label valid: `glass, paper, electronic, metal, plastic, cardboard, textile, foodwaste`.
- Metrik: **Accuracy**.
- Maksimal **3 submission per hari**. Kuota direset pukul 07.00.
- Public LB ≈ 30% data test (~145 citra), private ≈ 70%. Skor public bukan acuan (lihat larangan probing).

## 6. Administrasi

- Tim berisi 2–3 orang. Nama tim dan nama akun Kaggle harus sesuai registrasi. Merge team dilakukan sebelum submit (Rules 1–3).
- Dilarang kerja sama antartim (Rule 12).
- Semua komponen (notebook, laporan, submission) dikumpulkan tepat waktu. Terlambat berarti diskualifikasi (Rules 13–14).
- Panitia dapat menambah aturan sewaktu-waktu (Rule 15). Pantau QnA.

## 7. Konsekuensi untuk notebook

Notebook harus **reproducible end-to-end** dan menunjukkan alur: EDA → feature extraction (non-VLM) → clustering train → penamaan cluster manual (dengan bukti visual per cluster) → training classifier pada train → validasi lokal → inference test → submission.
