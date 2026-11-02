# EDA 208: imbalance antar-cluster dan baseline kualitas cluster

Skrip `cluster_imbalance.py`, keluaran `output/metrik_baseline.csv`, `cluster_imbalance.png`. Fungsi `metrik_ruang` dipakai juga di notebook untuk membandingkan 7B.

| Temuan | Implikasi teori | Tindakan |
|---|---|---|
| ViT-L: `ari_boot` 0,808 (K=8), 0,619 (K=32), 0,638 (K=48), 0,676 (K=64). Cluster rapuh (Jaccard < 0,6): 15, 18, dan 32 | Batas cluster pada K ≥ 32 tidak stabil. Satu run KMeans bukan dasar label yang aman | Stabilitas per citra dari 20 bootstrap, label hanya untuk `stab_nama` ≥ 0,7 |
| ConvNeXt-L: `ari_boot` lebih tinggi di K ≥ 48 (0,688 dan 0,738), `knn_ref` 0,737 lawan 0,721 | Ruang CNN tidak kalah stabil dari ViT-L. Silhouette CNN lebih tinggi, tetapi silhouette tidak sah dibandingkan antar ruang | 7B dibandingkan dengan aturan beku pada `ari_boot`, `rapuh`, dan `knn_top1` |
| Ukuran cluster ViT-L K=48: 15 sampai 83 (rasio 5,5). Label lama per kelas: rasio 1,49 | Imbalance antar cluster jauh lebih besar daripada antar kelas. Kelas besar tersusun dari banyak cluster | Syarat ukuran cluster minimal 10 saat memilih K. Bobot kelas akar-invers pada loss |
| Kemurnian label lama naik dengan K: 0,854 (32), 0,899 (48), 0,943 (64). Angka ini melingkar untuk ViT-L | Overclustering membuat tiap cluster lebih mudah diberi satu nama | K dipilih dari {32, 48, 64} |
