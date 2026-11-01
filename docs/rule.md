# Aturan

Sumber: teks halaman Kaggle yang ditempel user (2026-10-31).

## Aturan resmi

1. Setiap tim wajib terdiri dari 2 sampai 3 orang anggota.
2. Nama tim pada kompetisi Kaggle harus sesuai dengan nama tim yang terdaftar pada saat registrasi. Setiap tim wajib melakukan merge team sebelum melakukan submission.
3. Nama akun peserta Kaggle harus sesuai dengan nama yang didaftarkan pada perlombaan.
4. Peserta hanya diperbolehkan menggunakan dataset yang disediakan oleh panitia dan dilarang menggunakan dataset eksternal dalam pengerjaan proyek.
5. Peserta diperbolehkan menggunakan library, framework, dan tools untuk keperluan preprocessing, data augmentation, analisis data, pengembangan, pelatihan, dan evaluasi model.
6. Peserta diperbolehkan menggunakan pretrained model selama model tersebut dijalankan oleh peserta dalam lingkungan pengerjaan.
7. Peserta diperbolehkan menggunakan lebih dari satu model machine learning dalam pengerjaan proyek.
8. Peserta dilarang menggunakan layanan atau API pihak ketiga yang melakukan inference, klasifikasi, deteksi, atau prediksi terhadap data kompetisi, termasuk layanan computer vision, image recognition, maupun layanan AI lainnya.
9. Peserta dilarang menggunakan AutoML/LLM, VLM, generative AI, atau layanan AI pihak ketiga lainnya untuk membuat model, melakukan klasifikasi atau menghasilkan prediksi terhadap data kompetisi.
10. Penggunaan layanan eksternal yang hanya berfungsi sebagai library, framework, atau tools pendukung dan tidak melakukan inference terhadap data kompetisi tetap diperbolehkan.
11. Setiap tim diberikan kesempatan maksimal 3 kali submission per hari selama preliminary round.
12. Kerja sama antartim dilarang dalam bentuk apa pun yang dapat memberikan keuntungan tidak adil dalam kompetisi.
13. Peserta wajib mengumpulkan seluruh komponen proyek sesuai dengan format dan batas waktu yang telah ditentukan oleh panitia.
14. Peserta yang mengumpulkan proyek melebihi tenggat waktu yang telah ditentukan akan didiskualifikasi.
15. Aturan tambahan dapat ditetapkan oleh panitia apabila diperlukan selama pelaksanaan kompetisi.

## Ringkasan untuk pipeline

| Item | Keputusan |
|---|---|
| Data eksternal | Dilarang (rule 4) |
| Pretrained model | Boleh, asal dijalankan sendiri di lingkungan pengerjaan (rule 6) |
| Ensemble | Boleh (rule 7) |
| Inference lewat API pihak ketiga | Dilarang (rule 8) |
| AutoML, LLM, VLM, generative AI | Dilarang untuk membuat model, klasifikasi, atau prediksi (rule 9) |
| Submission | Maksimal 3 per hari (rule 11) |
| Tim | 2 sampai 3 orang, wajib merge sebelum submission (rule 1, 2) |
| Lisensi | Tidak disebutkan pada teks yang ditempel |

## Catatan kepatuhan (bukan aturan resmi, interpretasi proyek ini)

**Claude tidak boleh melihat citra kompetisi.** Membaca citra train/test lewat tool `Read` atau menampilkan citra ke Claude berarti inference VLM pihak ketiga terhadap data kompetisi (rule 8, 9). Claude hanya boleh membaca angka, tabel, dan statistik hasil script. Plot yang memuat citra asli (montage, grid sampel, galeri salah prediksi) hanya untuk dilihat manusia. Pelabelan train oleh Claude juga masuk larangan ini.

**Keputusan user (2026-10-31):** rule 9 dimaknai melarang LLM melakukan klasifikasi manual pada data kompetisi. Asisten LLM boleh menulis kode.

**Constraint proyek (ditetapkan user 2026-10-31): NO VLM.** Berlaku mutlak, termasuk untuk model yang dijalankan lokal.

- Dilarang: semua model bahasa-visi atau yang menerima teks untuk mengklasifikasi citra (SigLIP, CLIP, BLIP, LLaVA, Qwen vision, dan sejenisnya), zero-shot dengan prompt teks, serta backbone hasil pelatihan CLIP (nama timm berisi `clip`, misalnya `vit_large_patch14_clip_336.openai`, `eva02_large_patch14_clip_336.merged2b`). Termasuk model yang namanya tidak berisi `clip` tetapi dilatih dari model bahasa-visi: `eva02_*.mim_*` (target MIM EVA-CLIP), C-RADIOv4 (distilasi SigLIP2), AIMv2 (decoder merekonstruksi caption), dan SteerViT (text encoder RoBERTa).
- Larangan mencakup semua pemakaian: label, pseudo label, fitur, diagnosis, dan pembanding.
- Boleh: model yang hanya menerima citra, misalnya DINOv3 dan ConvNeXt, serta classifier ImageNet tanpa text encoder.
- Bila ada permintaan yang bertentangan dengan constraint ini atau aturan resmi, Claude wajib menolak dulu dan menyebut aturannya, walau permintaan itu datang dari user.

**Masih belum jelas:**

- Label train tidak tersedia. Label train saat ini berasal dari cluster DINOv3 yang dinamai ConvNeXtV2 (detail di `docs/lock.md`). Pseudo label pada test belum diizinkan.
