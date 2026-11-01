# EXP-005: ConvNeXtV2 ImageNet dengan pemetaan kelas

**Tujuan**: memprediksi 8 kategori tanpa model bahasa dan tanpa prompt teks.

**Metode**: `convnextv2_huge_clean` (ImageNet-1k, 384 px) dengan flip TTA. 231 dari 1.000 kelas dipetakan dari namanya: glass 11, paper 9, electronic 51, metal 39, plastic 12, cardboard 3, textile 59, foodwaste 47. Massa softmax dijumlahkan per kategori, kelas tak terpetakan diabaikan. Waktu 266 detik, VRAM puncak 4,53 GiB.

**Hasil**:

| Kelas | Test | Train |
|---|---|---|
| glass | 30 | 142 |
| paper | 39 | 148 |
| electronic | 71 | 219 |
| metal | 58 | 189 |
| plastic | 126 | 624 |
| cardboard | 47 | 202 |
| textile | 58 | 214 |
| foodwaste | 51 | 182 |

- Cakupan median 0,83 di test. 16,5% citra test punya top-1 di kelas tak terpetakan, terbanyak `ashcan`.

**Kesimpulan**:

- Skor temuan ada di EXP-007: macro F1 0,7113, accuracy 0,7104.
- `plastic` jadi kelas penampung: 126 di test, dengan `packet`, `tray`, dan `water bottle` sebagai sumber utama. Pemetaan `packet` dan `tray` kemungkinan terlalu kasar.
- `can opener` ternyata pintu masuk utama kelas `metal`, karena ImageNet tidak punya kelas kaleng.
- Draft `output/draft_submission_convnextv2.csv` lolos validasi format.
