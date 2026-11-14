# EDA 214: bahan penamaan cluster untuk tim

Skrip `montage_penamaan.py`. Keluaran untuk tim (bukan untuk asisten AI): `results/gambar_v2/cluster_<k>.png` (sampai 40 citra 150 px per cluster, dipisah sub-cluster a/b, judul merah = stab < 0,7), `results/prioritas_tinjau.csv`, `results/nama_cluster_templat.csv`.

Montage tidak memuat nama kelas, skor ImageNet, atau saran nama. Asisten AI tidak membuka montage ini (docs/rule.md: penamaan cluster oleh tim).

Prioritas tinjau = (1 - Jaccard) + rasio pisah/1,5 + (1 - porsi utuh di 7B) + (1 - kNN 7B). Urutan teratas: 47, 6, 7, 5, 18, 10, 24, 44, 39, 23, 8, 12.
