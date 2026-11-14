# EDA 213: kesepakatan ruang ViT-L (Run 1) dan 7B

Skrip `dua_ruang.py`, keluaran `output/dua_ruang.png`, `output/dua_ruang_cluster.csv`, `output/dua_ruang_citra.csv`.

| Temuan | Implikasi teori | Tindakan |
|---|---|---|
| ARI partisi K=48 ViT-L vs 7B 0,644. Median 83% anggota cluster ViT-L tetap bersama di 7B; 11 cluster < 60% (terendah 18, 44, 39, 6, 5, 10, 47) | Dua ruang DINOv3 hanya sebagian sepakat. 7B adalah pandangan kedua yang cukup independen | 7B tidak dipakai untuk clustering (aturan beku Run 1), tetapi dipakai untuk aturan kemiripan tetangga |
| kNN-10 dengan cluster sendiri (tanpa tetangga satu grup duplikat): ViT-L 0,813, 7B 0,787. Citra < 0,5: ViT-L 12,2%, 7B 15,5%, keduanya 9,9% | Sekitar satu dari sepuluh citra lebih mirip cluster lain di kedua ruang | Aturan buang: kNN-10 nama di ruang 7B >= 0,5. Di level nama angkanya lebih kecil karena cluster tetangga sering bernama sama |
| Spearman stab bootstrap vs kNN 7B 0,50 | Dua sinyal keraguan berkorelasi tetapi tidak redundan | Keduanya dipakai: stab_nama (ruang ViT-L) dan kNN nama (ruang 7B) |
