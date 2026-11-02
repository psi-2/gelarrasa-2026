# EDA 209: distribution shift train lawan test (diagnostik saja)

Skrip `shift.py`, gambar `output/shift.png`. Tidak dipakai untuk tuning, pembobotan, atau seleksi.

| Temuan | Implikasi teori | Tindakan |
|---|---|---|
| AUC adversarial: metadata 0,661, visual 0,686, embedding ViT-L 0,654 | Shift sedang. eda_102 menunjukkan penyebab utamanya komposisi sumber berkas per kelas yang bertukar antar split | Validasi ber-grup. Std antar fold dilaporkan karena skor test bisa bergeser |
| KS terbesar: tinggi 0,160, q_ijg 0,125, lbp_entropi 0,122 | Perbedaan didominasi ukuran berkas dan tekstur mikro (jejak encoder), bukan warna | Augmentasi tidak diarahkan ke warna |
| Tidak ada citra train yang terlihat seperti test (p > 0,5: 0%). ESS bobot importance 1.635 dari 1.920 | Train masih menutup sebagian besar distribusi test | Importance weighting tidak dipakai (juga akan memakai test untuk training) |
