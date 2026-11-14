# Notebook final DINOv3 ViT-7B QLoRA untuk Vast.ai (EXP-019, 2026-11-14, belum dijalankan)

Builder `build_nb.py` menghasilkan `notebook/dsc_gelarrasa_2026_dinov3_7b_qlora.ipynb` (78 sel). Target: Jupyter Vast.ai dengan satu GPU Blackwell. RTX 5090 32 GB memakai NF4, RTX PRO 6000 96 GB memakai bf16 (otomatis dari VRAM, ambang 60 GiB). Keluaran: satu `submission.csv` dari satu model.

## Isi notebook

- Label: `results/label_train.csv` dari notebook v2 (Kaggle versi 4), hash kanonik kolom `image_id, nama, berlabel, grup` dikunci di Settings. Clustering dan penamaan tidak diulang, label tidak diganti.
- Bobot: timm `vit_7b_patch16_dinov3.lvd1689m` revisi `b40f6c82...` (fp32 26,9 GB, tidak gated). Kerangka `init_empty_weights`, 200 lapis linear blok menjadi `Linear4bit` NF4 (double quant) atau bf16, parameter kecil fp32. Token HF dibaca dari env atau `.env`, tidak dicetak.
- LoRA r 16, alpha 32, dropout 0,05 lewat PEFT (regex `blocks.*`): 200 lapis, 39.321.600 parameter, ditambah head 8192 -> 8 (CLS + rata-rata patch).
- Uji 1 langkah: 384 px dengan batch mikro 16/8/4/2, lalu 320/256/224. Syarat VRAM puncak <= 85% dan perkiraan model final <= `BATAS_FINAL_JAM` 1,5 jam. Batch efektif tetap 16.
- Fitur beku sekali untuk linear probe SGKF5 (gerbang rata-rata >= 0,8) dan inisialisasi head.
- Training 4 epoch: AdamW LR LoRA 1e-4, head 1e-3, wd 0,05, warmup 10% lalu cosine, label smoothing 0,1, bobot kelas akar-invers, clip 1,0, autocast bf16, checkpoint non-reentrant per blok. Checkpoint per epoch untuk resume.
- Urutan: model final, adapter (hanya LoRA + head), test, `submission.csv`, CV (`BATAS_CV_JAM` 2 jam, fold disimpan), pseudo-label opsional (default mati, ambang 0,85, `submission_pseudo.csv`), `ringkasan_run.json`.

## Uji lokal (tanpa model asli, tanpa GPU)

- Semua sel kode lolos `ast.parse` dan pyflakes. Tidak ada em dash, divider, atau karakter non-ASCII di sel kode.
- `uji_logika.py nf4|bf16 <folder>` mengeksekusi semua sel kode berurutan dengan Eva mini (arsitektur 7B, dim 64, 2 blok, bobot acak), stub sel GPU, dan autocast CPU bf16 sebagai tiruan autocast CUDA. Lulus di kedua mode: format `submission.csv` sama dengan sample (header, urutan, ID tiga digit), adapter tanpa bobot base, CV 5 fold tanpa grup terbelah, resume checkpoint final memulihkan prediksi identik, fold CV dimuat dari disk, tangga fallback OOM dan waktu, OOM saat training menurunkan batch mikro, `kepala_lp` sama dengan `decision_function` sklearn, dan `muat_citra` identik dengan `prep_302/pipa.py`.
- Tanpa autocast, mode bf16 gagal di CPU (bobot bf16 bertemu aktivasi fp32). Di GPU semua forward notebook berada di dalam autocast CUDA bf16, jadi jalur ini aman selama autocast tidak dilepas.
- PEFT pada 7B di meta device: 200 modul LoRA, 39.321.600 parameter, scaling 2,0, patch embedding tidak tersentuh. Karena model timm tidak punya `is_loaded_in_4bit`, PEFT membungkus `Linear4bit` dengan `lora.layer.Linear` generik (bukan varian bnb). Jalur generik ini yang diuji di CPU dengan kuantisasi NF4 asli dan backward.

## Review jalur GPU (subagen, baca kode saja)

Tidak ada cacat yang pasti crash di GPU sehat. Lima risiko bersyarat ditutup:

1. Worker DataLoader dipaksa `fork`, karena Python 3.14 memakai forkserver dan worker tidak bisa memuat kelas buatan notebook.
2. Worker DataLoader 0 bila `/dev/shm` bebas < 2 GiB, karena bawaan Docker 64 MB memicu bus error. Biayanya beberapa persen waktu per langkah.
3. `PYTORCH_CUDA_ALLOC_CONF` dipasang sebelum `import torch`, karena `import bitsandbytes` sudah menginisialisasi CUDA.
4. Checkpoint dan hasil fold ditulis ke `.tmp` lalu `os.replace`, supaya kernel mati saat menulis tidak merusak resume.
5. timm dipasang `--no-deps`, supaya pip tidak mengganti torch bila image tidak punya torchvision.

Jalur `fork` tidak bisa diuji di Windows. Kernel sm_120 di wheel Linux bitsandbytes, versi Python image, dan ukuran `/dev/shm` di Vast belum terverifikasi.

## QA preprocessing (`qa_montage.py`, `output/`)

Montage seluruh 1.920 citra train per kelas (16 halaman, tanda T trim, A aspek ekstrem, K sisi < 200) dan 2 grid sebelum/sesudah augmentasi, memakai fungsi `pipa.py`. Diperiksa visual oleh Claude untuk preprocessing saja, setelah nama cluster final. Hasil penglihatan tidak dipakai untuk melabeli, memindah, atau membuang citra (QnA #16, #20).

- Orientasi EXIF benar. 123 citra hasil trim tidak menyisakan pita (contoh letterbox sumber 700, 1763, 1182 bersih). Hitam yang tersisa pada 1511, 1527, 1568 adalah latar studio di dalam bingkai, bukan pita.
- 95 citra beraspek ekstrem setelah trim memanjang karena stretch, objek tetap utuh. 105 citra sisi pendek < 200 px buram tetapi terbaca.
- Watermark stok foto, bingkai bawaan sumber (misalnya 1834, 1871), dan sudut gelap fisheye pada foto nampan foodwaste adalah sifat sumber. Trim tidak memotong sudut fisheye (baris tidak hitam seragam), dan memang tidak seharusnya.
- Augmentasi (RRC 0,5, flip, rot90, jitter) menjaga objek terlihat di semua contoh. Tidak ada perubahan preprocessing.

## Belum terbukti

Semua jalur GPU: kernel sm_120, unduhan 26,9 GB, kuantisasi NF4 di GPU, VRAM dan waktu nyata, hasil training dan skor. Perkiraan waktu (sekitar 24 TFLOP per citra per langkah training di 384 px, sekitar 40 menit untuk model final di RTX 5090) dihitung dari FLOP, bukan diukur. Angka nyata dicetak oleh sel uji 1 langkah.
