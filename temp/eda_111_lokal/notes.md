# EDA 111: fitur lokal patch token (Hipotesis D)

**Pertanyaan**: apakah pasangan plastic-glass-paper-metal terpisah lebih baik pada level patch dibanding CLS?

**Rancangan**: `patch.py` mengekstrak patch token DINOv3 ViT-L 384, PCA 64, float16. `bow.py` membuat bag of visual words, kosakata dari train saja. Target uji tidak berasal dari DINOv3: citra train yang pandangan vocabnya sepakat tanpa kNN dan tanpa cluster, supaya CLS tidak diuntungkan oleh cara label dibuat.

**Hardware**: satu job GPU, batas 5 GiB, puncak 1,96 GiB, 50 detik.

**Hasil** (AUC 5 fold ber-group, `output/cls_vs_bow.csv`):

| Pasangan | n pos | n neg | CLS | BoW | Gabungan |
|---|---|---|---|---|---|
| plastic-glass | 263 | 67 | 0,9895 | 0,9776 | 0,9892 |
| plastic-paper | 493 | 463 | 0,9995 | 0,9993 | 0,9996 |
| glass-paper | 540 | 586 | 0,9986 | 0,9988 | 0,9992 |
| metal-plastic | 451 | 431 | 0,9800 | 0,9872 | 0,9902 |
| metal-glass | 713 | 228 | 0,9888 | 0,9892 | 0,9913 |
| metal-paper | 649 | 292 | 0,9936 | 0,9918 | 0,9944 |
| paper-cardboard | 519 | 157 | 0,9966 | 0,9964 | 0,9978 |

**Kesimpulan**: pasangan sulit sudah terpisah pada CLS. Fitur lokal menambah paling banyak 0,0102. Hambatan utama bukan fitur, melainkan nama.

**Batasan**: target hanya citra yang sepakat antar kosakata dengan margin, jadi ini citra mudah dan AUC di atas lebih tinggi dari kinerja pada citra sulit. Jumlah n pos dan neg tiap pasangan juga tidak mewakili jumlah kelas sebenarnya.
