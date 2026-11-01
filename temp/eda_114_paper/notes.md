# EDA 114: paper, cardboard, glass, dan plastic

**Konteks**: agregat dari koordinator (tanpa ID): 466 dari 480 pada v5, target 471. Dari 14 kesalahan, prediksi paper menyerap 6 (metal 2, cardboard 2, textile 1, glass 1), plastic ke glass 2, sisanya masing-masing 1. Agregat ini berasal dari temuan dan mengarahkan fokus, tetapi tidak masuk kode.

**Batasan**: CPU saja, maksimal 4 thread. Bila bukti tidak kuat, tidak ada koreksi yang dipaksakan.

## Hasil per kasus

| Kasus | Bukti | Keputusan |
|---|---|---|
| Foil di paper | Bimodal, histogram train [1906, 5, 2, 0, 0, 1, 3, 3], tinfoil 0,498 | Rumus R1 |
| Kain di paper | Train memuat packet, DMS Paper 0,37. Test memuat diaper, DMS Fabric 0,624. Beda populasi | Tidak ada koreksi |
| Cardboard lawan paper | Tidak bimodal. Citra paper dengan kardus: DMS Paper 0,44, Cardboard 0,374 | Tidak ada koreksi |
| Glass lawan plastic | DMS menurun bertahap, dua sumber sepakat hanya 1 citra tiap arah, LR tidak setuju | Tidak ada koreksi |
| Kardus berlabel paper | Dua sumber 0 citra | Tidak ada koreksi |
| Cardboard dengan bukti kertas dua sumber | 3 citra, belum ada rumus beku | Hanya dicatat |

Foil per keluarga: train F_640 6, F_q75_4x3 1. Test F_416 2, jadi test berasal dari keluarga berbeda.

DMS tidak mengonfirmasi foil sebagai metal (median d_Metal 0,0). Bukti foil hanya dari kosakata 3 model 22k.

## Aturan beku (`label_v6.py`)

R1: median massa foil >= 0,20 menjadi metal.
R2 isi kosong: ViT dan CNN v5 sepakat, rata-rata p >= 0,80, LR setuju, dan massa kosakata >= 0,20 atau material DMS >= 0,30.

Hash `label_v6.py`: a1726047222ec46d160aebcc44b2e0dffb19cc6bec480101c2ffba9acec24392.
Hash `kosakata_material.py`: f0eaf1f6157d1b5daacc3a7f4e9600f4d62f27406d9de206abc571171bd532e4.
Beku 2026-10-31 18:01:05, dijalankan sekali.

## Hasil v6

| Kelas | v5 | v6 | Selisih |
|---|---|---|---|
| paper | 238 | 235 | -3 |
| metal | 230 | 238 | +8 |
| plastic | 283 | 287 | +4 |
| kelas lain | tetap | tetap | 0 |

Total 1.852 ke 1.861. 5 paper ke metal, 2 kosong ke metal (R1). 7 kosong terisi: plastic 4, paper 2, metal 1 (R2). 59 tetap kosong. Fold 0 tetap 384, tidak ada perubahan di fold 0.

R2 saringan: 66 kandidat, sepakat 58, p>=0,8 34, LR setuju 45, bukti luar 15, diterima 7.

## Uji tanpa temuan (`uji_v6.py`)

- Recall foil leave-group-out LR: 6 dari 7 pada bobot 1, 7 dari 7 pada bobot 3.
- Test LR v5 ke v6: berubah 3, paper ke metal. Dua citra foil, satu citra non-foil.
- p_metal citra foil test pada bobot 1: 0,415 dan 0,581, margin tipis.

## Batasan

- Hanya 7 contoh foil, dan foil test ada di keluarga berbeda dari train.
- Foil = metal belum terverifikasi. Ukuran 2 citra cocok dengan agregat, tetapi kecocokan ID tidak diketahui.
- Regex kosakata dibuat LLM dari nama, ide foil muncul dari profil train kosong dan test paper.
- Tidak ada bukti cukup untuk cardboard ke paper 2, textile ke paper 1, glass ke paper 1, dan plastic ke glass 2. Secara aritmetika, foil paling banyak memberi 468 dari 480.
