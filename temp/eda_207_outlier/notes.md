# EDA 207: outlier

Skrip `outlier.py`, keluaran `output/outlier.csv`, `outlier.png`, `montage_outlier_emb.png`.

| Temuan | Implikasi teori | Tindakan |
|---|---|---|
| Outlier embedding (z robust > 3,5 pada jarak kNN-10 atau LOF): 65 citra (3,4%). Tidak ada irisan dengan outlier kualitas (28) maupun metadata (8) | Outlier embedding bukan citra rusak, melainkan isi yang jarang | Citra tidak dibuang. Outlier akan jatuh ke cluster kecil atau pinggir dan tersaring oleh `stab_nama` |
| 25 dari 65 berlabel lama foodwaste | Foodwaste paling beragam secara visual (banyak jenis makanan) | Wajar bila foodwaste terpecah ke beberapa cluster. Tim bisa memberi nama sama ke beberapa cluster |
| Spearman jarak kNN lawan keyakinan label lama -0,27 | Citra terpencil cenderung berlabel ragu | Mendukung penyaringan stabilitas per citra |
