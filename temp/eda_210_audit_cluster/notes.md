# EDA 210: audit partisi Run 1 (ViT-L, K=48)

Skrip `audit_cluster.py`, keluaran `output/audit_cluster.csv`, `output/subcluster_train.csv`, `output/audit_cluster.png`. Train-only, tanpa nama kelas.

| Temuan | Implikasi teori | Tindakan |
|---|---|---|
| KMeans ulang lokal (sklearn 1.9.1) mereproduksi partisi Kaggle dengan ARI 1,000 | PCA dan KMeans deterministik lintas lingkungan pada data ini | Usulan pecah k=2 boleh dihitung lokal lalu dimuat di notebook (`subcluster_train.csv`); notebook tetap mencetak kesepakatan dengan hitung ulang |
| 78,0% citra stab >= 0,7. 16 dari 48 cluster rapuh (Jaccard bootstrap < 0,6) | Batas cluster bergeser antar subsample, citra pinggir rawan salah nama | Label hanya untuk `stab_nama` >= 0,7 (level nama, bukan level cluster) |
| Pasangan yang paling sering bertukar anggota: 12-24 (0,742), 7-29 (0,512), 8-17, 6-46, 6-18, 34-40, 18-47, 3-20 | Bila sepasang cluster diberi nama sama, pertukaran tidak menambah noise. Bila berbeda, citra perbatasan menjadi noise | Tim menamai pasangan ini dengan membandingkan kedua montage berdampingan |
| Calon pecah (jarak sub-centroid >= jarak ke tetangga, sub >= 5, tetangga sub berbeda): cluster 5, 6, 47 | Cluster bisa memuat dua kelompok visual yang masing-masing dekat ke cluster lain | Tim boleh menamai `5a`/`5b` terpisah. Keputusan tetap pada tim |
| Spearman stab_mean vs sebaran -0,54, vs rasio pisah -0,51 | Cluster yang menyebar dan bercabang adalah cluster yang tidak stabil | Prioritas tinjau digabung dari Jaccard, rasio pisah, dan sinyal 7B (EDA 214) |

Catatan: t-SNE di skrip ini lambat di laptop (lebih dari 10 menit), hanya untuk visual.
