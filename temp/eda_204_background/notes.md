# EDA 204: latar dan foreground

Skrip `background.py`, keluaran `output/latar.csv` (dipakai prep_301), `background.png`, montage per tipe latar.

| Temuan | Implikasi teori | Tindakan |
|---|---|---|
| Bertekstur 76,5% di kedua split. Putih polos: train 16,8%, test 18,8% | Distribusi tipe latar train dan test sama | Tidak perlu augmentasi penggantian latar |
| Putih polos per label lama: metal 33%, cardboard 30%, glass 28%, paper 4%, textile 2%. Cluster K=8 bernama glass 52% berlatar putih | Latar berkorelasi dengan cluster. Embedding dan CNN bisa memakai latar sebagai jalan pintas | Crop acak (scale ≥ 0,4) mengurangi porsi latar. Tim memeriksa montage untuk cluster yang terbentuk karena latar |
| Box foreground menyentuh tepi pada 78,4% citra. Objek kecil (box < 25% luas) 5,4% | Sebagian besar objek memenuhi bingkai, atau mask tidak andal pada latar bertekstur | Verifikasi crop hanya memakai citra berlatar polos (mask andal) |
