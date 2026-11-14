# Constraint DSC Gelar Rasa 2026

Sumber: Padlet QnA resmi panitia (folder `Padlet - QnA Data Science Competition Gelar Rasa 2026/`, 25 pos, jawaban akun "Gelar Rasa 2026"). Aturan resmi lomba (Rules 1-15, format submission, metrik) ada di [context.md](context.md). Angka dalam kurung adalah nomor pos Padlet. Satu batasan internal tim ditambahkan di bagian 1.

## 1. Batasan internal tim

| Dilarang | Maksudnya |
|---|---|
| **Blending / ensembling prediksi** | Tidak boleh merata-rata, voting, atau stacking output beberapa model untuk submission. Submission berasal dari **satu model**. Beberapa model dalam satu *pipeline* (misalnya backbone untuk clustering lalu classifier) tetap boleh sesuai Rule 7. |

## 2. Data

- Folder train tidak punya ground truth atau label, seluruh citra latih ada di satu folder (#2).
- Dataset eksternal **tidak boleh**, termasuk TrashNet, RealWaste, maupun dataset publik lain (#22.6).
- Data test **hanya untuk prediksi akhir**. Test tidak boleh dipakai untuk pelatihan maupun fitting model, termasuk clustering dan normalisasi. Parameter preprocessing diperoleh dari train lalu diterapkan ke test (#24).
- Tidak wajib memakai seluruh data train. Penanganan citra noise diserahkan ke strategi tim (#15). Menghapus data ambigu atau noise saat data cleaning boleh (#18).
- Test-time augmentation ditanyakan di #24 tetapi tidak dijawab secara eksplisit.

## 3. Pelabelan train

- **Penamaan cluster secara manual boleh**, berdasarkan karakteristik citra, selama clustering memakai model yang dikembangkan sendiri oleh peserta (#1, #2, #14). Nama folder hasil clustering boleh dipakai sebagai penanda atau identitas cluster (#22.2). Peserta dianjurkan mengembangkan model clustering sendiri (#4).
- Hasil clustering yang dilabeli lewat metode atau program boleh menjadi dasar model klasifikasi supervised (CNN, transfer learning). Proses ini tidak termasuk pelabelan manual atau pseudo-labeling manual (#12).
- **Clustering ulang** pada cluster yang anggotanya campur boleh. Jumlah cluster bebas sesuai strategi tim (#19.2).
- Pemeriksaan hasil clustering boleh dilakukan dengan program atau metode (#10, #12).
- **Pelabelan atau koreksi manual per gambar:**
  - Koreksi atau pemindahan gambar antarcluster secara manual satu per satu tidak boleh (#10, #12, #13, #14). Pelabelan manual pada train tidak disarankan (#20).
  - Koreksi manual pada sebagian kecil data train boleh (#19.1).
  - Batas hand labeling **5 gambar per kelas, maksimal 40 gambar**. Label boleh ditulis langsung di notebook (#22.1).
  - Tim yang terlanjur melabeli manual tanpa clustering disarankan memodelkan ulang (#10). Bila hasil clustering belum optimal, peserta disarankan mencoba teknik pelabelan lain (#10).
- **Pelabelan otomatis boleh**: pseudo-labeling, kNN, label propagation, dan self-training, selama hanya memakai data train panitia, tanpa data test, tanpa dataset eksternal, dan tanpa model atau layanan AI yang dilarang (#22.3).
- Membuang gambar yang terindikasi salah cluster secara otomatis dengan aturan program (misalnya jarak ke pusat cluster atau kemiripan tetangga) boleh, tanpa memeriksa gambar satu per satu (#19.3).
- **Perbaikan label train otomatis dari prediksi model:** jawaban panitia bertentangan.
  - #19.4 membolehkannya pada data train tanpa pengecekan manual per gambar.
  - #21 menyatakan penggantian label individual berdasarkan confidence model tidak boleh.
- Bantuan AI dalam anotasi maupun pelabelan data **tidak boleh** (#16, #20).

## 4. Model

- **VLM tidak boleh** (misalnya SigLIP dan CLIP), termasuk untuk feature extraction maupun tahapan lain (#4, #22.4). Larangan ini berlaku juga bila hanya memakai image encoder tanpa teks (#23).
- Zero-shot CLIP dengan prompt teks nama kelas tidak boleh (#22.5).
- Pretrained model computer vision non-VLM (ResNet, EfficientNet, DINOv2) **boleh** untuk feature extraction maupun fine-tuning, selama dijalankan dan dilatih atau di-fine-tune sendiri oleh peserta di lingkungan pengerjaan (#5, #22.4).
- Model pretrained dari Hugging Face boleh **hanya sebagai base model** dan wajib di-fine-tune dengan data train panitia. Model yang sudah di-fine-tune menyeluruh oleh pihak lain tidak boleh. Model jadi **dilarang keras** dipakai untuk langsung menebak data test tanpa pelatihan pada dataset kompetisi (#17).

## 5. Submission dan penilaian

- Kuota submission harian direset setiap pukul 07.00 WIB (#11).
- Penilaian tidak hanya dari skor Kaggle, tetapi juga notebook dan laporan analisis (#6, #7).
- Panitia memeriksa model, notebook, dan laporan secara menyeluruh di akhir kompetisi. Pelanggaran berujung diskualifikasi (#3, #7, #8).
- Laporan mengikuti struktur subbab template laporan yang disediakan (#25).
