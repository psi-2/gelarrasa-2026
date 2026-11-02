# EDA 205: distribusi warna

Skrip `warna.py`, keluaran `output/warna.csv` (dipakai prep_301), `warna.png`.

| Temuan | Implikasi teori | Tindakan |
|---|---|---|
| KS Lab/saturation train lawan test ≤ 0,072 | Tidak ada shift warna berarti | Tidak perlu normalisasi warna khusus |
| Cardboard: b* 20,8, saturasi 0,40. Paper: b* 7,2, saturasi 0,18. Jarak centroid 14,2 dE, hampir seluruhnya di sumbu chroma | Chroma adalah pembeda utama cardboard dan paper. Augmentasi yang menurunkan chroma (saturation, contrast) menggeser cardboard ke arah paper | Saturation dan hue jitter ditiadakan, contrast dibatasi (prep_301) |
| Glass, metal, paper, dan plastic hanya berjarak 3,6 sampai 7,1 dE. Std alami dalam kelas L 15,7, a 9,3, b 11,1 | Warna rata-rata tidak membedakan keempatnya. Pembedanya tekstur, specular, dan bentuk | Augmentasi tidak boleh menghaluskan tekstur (tanpa denoise, tanpa blur) |
