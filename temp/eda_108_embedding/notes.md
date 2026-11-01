# EDA 108: struktur embedding (Hipotesis D)

**Metode**: `struktur.py`, PCA 128 lalu KMeans, stabilitas 5 seed, adversarial train vs test. PCA pada skrip ini memakai train dan test, jadi hanya diagnosis dan tidak untuk pembuatan label.

**Hasil**:

| k | silhouette | ARI antar seed |
|---|---|---|
| 8 | 0,220 | 0,778 |
| 12 | 0,229 | 0,766 |
| 32 | 0,223 | 0,645 |
| 64 | 0,258 | 0,660 |
| 96 | 0,274 | 0,658 |
| 192 | 0,271 | 0,640 |

- Tidak ada k alami. Silhouette mendatar sekitar k=96. ARI hanya sekitar 0,65 untuk k >= 32, jadi batas cluster tidak stabil dan satu run KMeans tidak layak jadi dasar label.
- AUC adversarial: DINOv3 0,6375, forensik GBM 0,7121. Di dalam keluarga naik: F_416 0,7458, F_640 0,7401. Ini cocok dengan kelas penyusun sumber yang berbeda antar split.
- Rata-rata ESS importance weight train 1.731,9 dari 1.920.

**Fitur lokal**: lihat `temp/eda_111_lokal/notes.md`. BoW patch hampir tidak menambah di atas CLS.
