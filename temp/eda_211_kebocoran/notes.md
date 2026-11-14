# EDA 211: near-duplicate, kebocoran validasi, konflik label dalam grup

Skrip `kebocoran.py`, keluaran `output/grup_train.csv`, `output/kebocoran.png`, `output/montage_grup_lintas_cluster.png`.

| Temuan | Implikasi teori | Tindakan |
|---|---|---|
| Pasangan Hamming 0: 54, <= 2: 85, <= 4: 100. Salinan hampir identik (H <= 2): 169 citra di 84 grup | Salinan flip/rot90 sama persis ada di train | Grup near-duplicate wajib untuk validasi |
| Aturan notebook (H <= 4 atau cos >= 0,8573, kuantil 0,0721%): 1.396 grup, terbesar 50, 732 citra (38,1%) dalam grup > 1 | Hampir 2 dari 5 citra punya kembaran | StratifiedGroupKFold dengan grup ini |
| Simulasi kNN-1 dengan target cluster Run 1: piksel 24x24 acak 0,264 vs grup 0,193 (+7,0 poin); embedding 7B acak 0,871 vs grup 0,840 (+3,1 poin). Pada split acak 33% NN validasi berasal dari grup duplikatnya sendiri | Validasi acak terlalu optimis karena salinan, bahkan untuk fitur semantik | Angka CV hanya dilaporkan dari split grup |
| 8 grup (18 citra) tersebar di lebih dari satu cluster. 5,9% pasangan H <= 2 jatuh di cluster berbeda. Stab citra di grup lintas cluster 0,60 vs 0,815 | Citra yang identik bisa mendapat nama berbeda setelah penamaan, yaitu label yang saling bertentangan | Aturan pembersihan: grup duplikat dengan lebih dari satu nama dibuang seluruhnya (drop, bukan relabel) |
