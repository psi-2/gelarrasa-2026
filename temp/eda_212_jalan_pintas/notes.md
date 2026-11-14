# EDA 212: atribut non-isi per cluster (jalan pintas)

Skrip `jalan_pintas.py`, keluaran `output/jalan_pintas.png`, `output/cluster_atribut_dominan.csv`, `output/atribut_train.csv`.

| Temuan | Implikasi teori | Tindakan |
|---|---|---|
| Cramer's V cluster vs atribut: qhash 0,772, pita hitam 0,747, sumber berkas 0,725, ukuran 0,674, tipe latar 0,458, ukuran objek 0,440 | Ruang DINOv3 ikut mengelompokkan gaya sumber foto, bukan hanya material | Montage cluster yang didominasi satu sumber diperiksa paling teliti saat penamaan |
| Cluster 1, 2, 34, 40 berisi 97-100% sumber `F_q2527_400x` (JPEG q70, 400 px). EDA 202: sumber ini 9,2% train dan 0% test | Bila model memakai isyarat sumber ini, isyarat itu tidak pernah muncul di test sehingga tidak memicu prediksi salah; risikonya hanya kelas tersebut kurang belajar isi dari sumber lain | Tidak ada augmentasi JPEG. Dicatat sebagai risiko generalisasi kelas terkait |
| Cluster 16 dan 30 didominasi sisi pendek < 384 (82%, 81%), cluster 29 didominasi > 640 (88%), cluster 44 100% sumber `F_q75_4x3` | Resolusi ikut membentuk cluster. Sebaran resolusi train dan test mirip (EDA 201), jadi bias ini dimiliki kedua split | Tanpa augmentasi resolusi, karena bias yang dimiliki train dan test tidak merugikan skor test |
