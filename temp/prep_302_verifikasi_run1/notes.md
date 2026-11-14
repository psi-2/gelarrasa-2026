# PREP 302: verifikasi visual pipeline notebook per cluster Run 1

`pipa.py` berisi salinan persis fungsi notebook (trim, load, transform torchvision v2). `verifikasi_run1.py [rrc_min]` membuat grid: 1 citra per cluster Run 1 (4 halaman) dan kasus tepi (trim terbesar, objek kecil, aspek ekstrem, sangat kecil). `iterasi_rrc.py` mengukur sisa foreground per crop. Parameter final: `output/prep_final.json` (dibaca builder notebook). Hanya transform, tanpa model.

| Langkah | Hasil | Keputusan |
|---|---|---|
| Trim pita hitam | 123 citra train, median 30,3% luas dibuang, maks 77,6%. Identik dengan prep_301. Visual: pita sumber 640x640 terpotong bersih, isi tengah utuh. Terkumpul di cluster 18 (42), 36 (23), 46 (19), 6 (15), 42 (10) | Tetap |
| RRC iterasi 1 (0,4) | Visual: objek tipis/kecil di latar polos hilang (kipas 1194: 3 dari 4 crop tanpa objek; kaleng 1617 tinggal potongan). Ukur: p(sisa < 50%) objek kecil 0,1005, latar polos 0,0103 | Gagal kriteria pra-ukur (<= 0,10 dan <= 0,02) |
| RRC iterasi 2 (0,5) | p(sisa < 50%) 0,0728 dan 0,0030, luas crop rata-rata 0,694. Visual: kipas 1194 tampil di 4 dari 4 crop | Dipakai |
| Stretch aspek ekstrem | 49 train beraspek > 1,6 atau < 0,62 (6 di luar [0,5; 2]). Bentuk memanjang sampai 2-2,85x (kaleng 1373, deretan botol 42), tekstur tetap | Tetap stretch. Porsi aspek ekstrem mirip di test, preprocessing identik |
| Citra sangat kecil | 105 train sisi pendek < 200 px, buram setelah upscale ke 512 tetapi isi terbaca. Watermark stok foto (Can Stock Photo, dreamstime, gettyimages) terlihat | Tanpa tindakan. Watermark tidak bisa dibuang tanpa model, dan sumber stok ada di kedua split |
| Augmentasi JPEG/resolusi | Tidak diuji visual, ditolak dari EDA 212 (bias dimiliki kedua split, sumber khusus train tidak muncul di test) | Tidak dipakai |
