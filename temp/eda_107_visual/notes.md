# EDA 107: fitur visual dan forensik (Hipotesis E, F)

**Metode**: `fitur_visual.py` menghitung warna, tekstur, tepi, padding, anisotropi, periode grid. `forensik.py` mencari sub-sumber. `pemisah.py` menguji pemisahan pasangan sulit. Keluaran `visual.csv`, `gabung.csv`.

**PERINGATAN KEPATUHAN**: `pemisah.py` versi pertama memakai kamus `JANGKAR` berisi nomor cluster yang dipilih dan dinamai sendiri oleh agen. Ini penamaan cluster manual oleh LLM dan melanggar aturan. Semua AUC dari versi itu TIDAK SAH dan sudah dibuang dari catatan ini. Skrip diganti dengan aturan otomatis, lihat bagian hasil di bawah bila sudah terisi.

**Hasil Hipotesis E** (versi otomatis): acuan adalah citra train dengan porsi suara >= 0,9 dari `label_train_bag.csv`. Fitur non-deep tidak dipakai saat membuat label, jadi ini cek independen, tetapi tetap bisa terbantu perbedaan sumber.

| Pasangan | n | AUC GBM | Dalam satu keluarga | Fitur terbaik |
|---|---|---|---|---|
| paper vs cardboard | 395 | 0,990 | F_416 0,948 | pusat_rendah_sat 0,90, coklat 0,86 |
| plastic vs paper | 332 | 0,977 | - | hue01 0,82, pusat_spec 0,79 |
| metal vs paper | 302 | 0,957 | F_640 0,970 | orient_entropi 0,76 |
| glass vs paper | 320 | 0,949 | - | hue01 0,77 |
| plastic vs glass | 282 | 0,939 | F_416 0,934 | lbp_entropi 0,80, lok_halus 0,79 |
| metal vs plastic | 264 | 0,919 | - | lbp_entropi 0,76 |
| metal vs glass | 252 | 0,895 | F_416 0,925 | orient_entropi 0,77 |

**Hasil Hipotesis F** (tidak bergantung pada jangkar):

| Kriteria | Train | Test |
|---|---|---|
| kontras < 0,25 | 1,61% | 2,08% |
| buram lap_var < 0,0005 | 2,14% | 1,67% |
| noise > 0,03 | 2,45% | 2,50% |
| sisi terpendek < 160 | 0,63% | 0,21% |

Noise bukan sumber kesalahan utama. Noise tinggi menumpuk di F_lain_web (59 dari 120).

**Hasil forensik**:

- Padding hitam 182 citra, 178 di F_640. Dari nama cluster: electronic 142, foodwaste 29.
- Periode grid di F_640 mengisyaratkan ukuran asli beragam (1280, 1030x2050, 1790), jadi F_640 berasal dari banyak sumber yang diresize ke 640.
- GMM forensik dalam keluarga: AMI ke cluster DINO 0,18 di F_416 dan 0,31 di F_640. Sub-sumber ada tetapi tidak tajam.
- Latar putih empat sisi 284 citra (train 226, test 58).
