# Data Science Competition Gelar Rasa 2026

Preliminary Round Data Science Competition Gelar Rasa 2026

## Overview

Selamat datang di GELAR RASA 2026 cabang lomba Data Science Competition! Pada cabang lomba ini, peserta ditantang untuk mengidentifikasi dan mengklasifikasikan kategori sampah (Waste Category) berdasarkan data citra menggunakan pendekatan Data Science dan teknik Computer Vision.

Pengelolaan sampah masih menjadi tantangan permasalahan lingkungan yang mendesak hingga saat ini, karena peningkatan volume sampah yang tidak diimbangi dengan pengelolaan yang tepat dapat menyebabkan pencemaran tanah, air, kerusakan ekosistem, penyumbatan saluran drainase yang memicu banjir, serta mengancam kesehatan masyarakat akibat penumpukan sampah. Selain itu, sampah yang tercampur dapat menyulitkan proses daur ulang dan menyebabkan material yang masih bernilai guna dapat terbuang sia-sia. Oleh karena itu, identifikasi dan pemilahan sampah berdasarkan kategorinya menjadi langkah penting untuk mendukung pengelolaan sampah yang lebih efektif. Namun, tantangannya sampah memiliki wujud yang sangat beragam dari segi bentuk, warna, tekstur, dan kondisi, sehingga pemilahan manual menjadi lambat, melelahkan, dan rawan keliru, sehingga teknik Computer Vision dapat dimanfaatkan untuk membantu mengenali dan mengklasifikasikan sampah berdasarkan data citra. Itulah tantangan yang dihadirkan pada Data Science Competition (DSC) Gelar Rasa 2026. Pada preliminary round, peserta diajak membangun model Image Classification berbasis Computer Vision untuk mengklasifikasikan sampah ke dalam delapan kategori, yaitu glass, paper, electronic, metal, plastic, cardboard, textile, dan food waste. Namun, proses yang dijalani tidak hanya sebatas melatih model klasifikasi. Dataset yang digunakan juga menghadirkan berbagai kondisi yang umum ditemui pada data dunia nyata, seperti adanya noise pada citra, variasi latar belakang yang beragam, serta distribusi data yang tidak merata (class imbalance) antar kategori.

Melalui tantangan tersebut, peserta perlu menerapkan tahapan data science secara menyeluruh, mulai dari eksplorasi data, preprocessing, pemilihan metode pemodelan, hingga evaluasi performa model pada setiap kategori. Rangkaian proses ini diharapkan dapat memberikan gambaran mengenai bagaimana data dan teknologi dapat dimanfaatkan untuk membantu proses identifikasi serta pemilahan sampah berdasarkan kategori materialnya. Dengan demikian, fokus kompetisi tidak hanya berada pada hasil prediksi yang diperoleh, tetapi juga pada pendekatan dan pemahaman yang digunakan dalam mengolah data untuk menjawab permasalahan yang diberikan.

## Objektif Kompetisi

Kompetisi ini berfokus pada pengembangan model Image Classification yang mampu mengklasifikasikan citra sampah ke dalam 8 kategori secara akurat, serta menghasilkan insight berdasarkan pola yang ditemukan dari data. Adapun 8 kategori sampah yang digunakan dalam kompetisi adalah:

1. glass
2. paper
3. electronic
4. metal
5. plastic
6. cardboard
7. textile
8. foodwaste

## Tujuan

Dalam kompetisi Data Science Competition (DSC) Gelar Rasa 2026 ini, peserta ditantang untuk menerapkan tahapan Data Science secara menyeluruh dalam menyelesaikan permasalahan klasifikasi citra sampah. Peserta diharapkan mampu mengolah dan memahami karakteristik data, mengembangkan model berbasis Computer Vision, serta menganalisis hasil yang diperoleh untuk menghasilkan solusi dan insight yang relevan terhadap permasalahan yang diberikan. Adapun tujuan yang ingin dicapai dalam kompetisi ini adalah:

1. Melakukan preprocessing dan data cleaning pada citra.
2. Membangun model klasifikasi berbasis Computer Vision.
3. Memprediksi kategori sampah pada data yang diberikan.
4. Menganalisis hasil model dan menghasilkan insight berdasarkan pola pada data.

Dataset yang digunakan dalam kompetisi terdiri dari 1.920 data train dan 480 data test tanpa label. Dataset tersebut memiliki kondisi yang beragam, seperti noise, latar belakang yang berbeda, dan ketidakseimbangan jumlah data antarkelas. Kondisi tersebut diharapkan dapat mendorong peserta untuk menerapkan strategi pengolahan data dan pemodelan yang sesuai, sehingga model yang dikembangkan mampu menghasilkan prediksi yang baik serta memberikan hasil analisis yang dapat mendukung pengembangan solusi lebih lanjut.

### Evaluation

Submission akan dievaluasi menggunakan **Accuracy**, yaitu proporsi jumlah prediksi yang benar terhadap seluruh data yang dievaluasi.

| Accuracy = \frac{TP + TN}{TP + TN + FP + FN} |
| -------------------------------------------- |

dengan:

- **TP (True Positive):** Jumlah data positif yang diprediksi dengan benar.
- **TN (True Negative):** Jumlah data negatif yang diprediksi dengan benar.
- **FP (False Positive):** Jumlah data negatif yang salah diprediksi sebagai positif.
- **FN (False Negative):** Jumlah data positif yang salah diprediksi sebagai negatif.

Accuracy digunakan untuk mengukur seberapa banyak data yang berhasil diklasifikasikan dengan benar oleh model dibandingkan dengan seluruh data yang dievaluasi.

Semakin tinggi nilai Accuracy, semakin besar proporsi prediksi yang benar, sehingga menunjukkan performa model yang lebih baik dalam mengklasifikasikan kategori sampah.

## Submission File

Untuk setiap data pada test set, peserta harus memprediksi kategori sampah menggunakan model yang telah dikembangkan.

Format file submission yang digunakan adalah CSV dengan dua kolom sebagai berikut:
```autohotkey
image_id,waste_category
001,glass
002,plastic
003,metal
004,paper
005,textile
006,foodwaste
007,cardboard
008,electronic
```

## Dataset Description

Dataset pada kompetisi ini terdiri dari citra limbah yang dikelompokkan ke dalam delapan kategori utama sebagai berikut:

1. **glass**: Merepresentasikan limbah berbahan kaca, seperti botol, wadah, atau pecahan kaca.
2. **paper**: Mencakup berbagai jenis limbah berbahan kertas, seperti lembaran kertas, koran, dan bahan berbasis kertas lainnya.
3. **electronic**: Menggambarkan limbah peralatan atau komponen elektronik yang sudah tidak digunakan.
4. **metal**: Merepresentasikan limbah berbahan logam, seperti kaleng, wadah logam, dan benda logam lainnya.
5. **plastic**: Mencakup berbagai jenis limbah berbahan plastik, seperti botol, kemasan, dan wadah plastik.
6. **cardboard**: Merepresentasikan limbah berbahan kardus, seperti kotak dan berbagai jenis kemasan berbahan cardboard.
7. **textile**: Mencakup limbah berbahan tekstil, seperti pakaian, kain, dan material tekstil lainnya.
8. **foodwaste**: Menggambarkan limbah yang berasal dari sisa makanan atau bahan pangan yang sudah tidak digunakan.

## Competition Rules

Setiap tim peserta Data Science Competition Gelar Rasa 2026 wajib mematuhi peraturan-peraturan berikut:

1. Setiap tim wajib terdiri dari 2 sampai 3 orang anggota.
2. Nama tim pada kompetisi Kaggle harus sesuai dengan nama tim yang terdaftar pada saat registrasi. Setiap tim wajib melakukan *merge team* sebelum melakukan *submission*.
3. Nama akun peserta Kaggle harus sesuai dengan nama yang didaftarkan pada perlombaan.
4. Peserta hanya diperbolehkan menggunakan dataset yang disediakan oleh panitia dan dilarang menggunakan dataset eksternal dalam pengerjaan proyek.
5. Peserta diperbolehkan menggunakan *library*, *framework*, dan *tools* untuk keperluan *preprocessing*, data *augmentation*, analisis data, pengembangan, pelatihan, dan evaluasi model.
6. Peserta diperbolehkan menggunakan *pretrained* model selama model tersebut dijalankan oleh peserta dalam lingkungan pengerjaan.
7. Peserta diperbolehkan menggunakan lebih dari satu model *machine learning* dalam pengerjaan proyek.
8. Peserta dilarang menggunakan layanan atau API pihak ketiga yang melakukan *inference*, klasifikasi, deteksi, atau prediksi terhadap data kompetisi, termasuk layanan *computer vision*, *image recognition*, maupun layanan AI lainnya.
9. Peserta dilarang menggunakan *AutoML*/*LLM*, *VLM*, g*enerative AI*, atau layanan AI pihak ketiga lainnya untuk membuat model, melakukan klasifikasi atau menghasilkan prediksi terhadap data kompetisi.
10. Penggunaan layanan eksternal yang hanya berfungsi sebagai *library*, *framework*, atau *tools* pendukung dan tidak melakukan *inference* terhadap data kompetisi tetap diperbolehkan.
11. Setiap tim diberikan kesempatan maksimal 3 kali *submission* per hari selama *preliminary round*.
12. Kerja sama antartim dilarang dalam bentuk apa pun yang dapat memberikan keuntungan tidak adil dalam kompetisi.
13. Peserta wajib mengumpulkan seluruh komponen proyek sesuai dengan format dan batas waktu yang telah ditentukan oleh panitia.
14. Peserta yang mengumpulkan proyek melebihi tenggat waktu yang telah ditentukan akan didiskualifikasi.
15. Aturan tambahan dapat ditetapkan oleh panitia apabila diperlukan selama pelaksanaan kompetisi.
