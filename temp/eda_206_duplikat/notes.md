# EDA 206: duplikat dan near-duplicate

Skrip `duplikat.py`, keluaran `output/grup_train.csv`, `duplikat.png`, `montage_grup_terbesar.png`.

| Temuan | Implikasi teori | Tindakan |
|---|---|---|
| Train-train pHash dihedral Hamming ≤ 4: 100 pasangan, kesepakatan label lama 0,99. Hamming ≤ 10: 272 pasangan, kesepakatan 0,64 | Ambang 10 merantai citra berlatar polos yang berbeda isi | Ambang Hamming 4 |
| Cos ViT-L ≥ 0,92: 1.329 pasangan, kesepakatan 0,998. Angka ini melingkar karena label lama lahir di ruang ViT-L | Cosine menangkap salinan hasil crop atau warna yang lolos dari pHash | Grup = Hamming ≤ 4 atau cos ≥ 0,92: 1.471 grup, terbesar 52, 628 citra dalam grup > 1 |
| Test dengan pasangan train Hamming ≤ 4: 7,3%. NN cos ≥ 0,95: 11,0% | Sebagian test adalah salinan train. Validasi acak akan terlalu optimistis dibanding skor test | StratifiedGroupKFold dengan grup ini. Test tidak ikut membentuk grup |
| Notebook memakai kuantil kepadatan pasangan (0,0721%), bukan cos 0,92 mentah | Nilai cosine tidak setara antar ruang embedding (ViT-L lawan 7B) | Ambang cosine diturunkan dari kuantil di ruang yang dipakai |
