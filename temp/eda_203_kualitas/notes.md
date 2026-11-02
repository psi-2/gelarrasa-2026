# EDA 203: noise, blur, kontras, blockiness

Skrip `noise_blur.py`, gambar `output/noise_blur.png` dan montage terburuk (untuk manusia).

| Temuan | Implikasi teori | Tindakan |
|---|---|---|
| Noise > 0,03: train 2,4%, test 2,5%. 30 dari 47 citra bising train ada di keluarga web | Noise adalah ciri sumber unduhan web, bukan kondisi umum dataset | Tanpa denoise global. prep_301: NLM h=6 hanya menurunkan noise 1% pada citra bising, tetapi menghapus 17% tekstur pada citra bersih |
| Buram (var Laplacian < 5e-4): 2,1% dan 1,7%. KS noise 0,034, KS blur 0,042 | Kualitas train dan test hampir sama | Tidak perlu augmentasi blur atau noise untuk menutup shift |
| Kontras rendah (< 0,10) 10,5%. Blockiness > 1,5: train 16,4%, test 10,0% | Kontras rendah banyak muncul pada latar putih. Blockiness adalah jejak resize dari JPEG lama | Contrast jitter dibatasi 0,1 (prep_301) |
| Median noise per cluster K=8 berkisar 0,0016 sampai 0,0063 | Noise berkorelasi dengan cluster lewat sumber, sehingga cluster bisa sebagian menangkap sumber | Montage penamaan dicek untuk cluster dengan satu sumber dominan |

**Koreksi setelah inspeksi visual (2026-11-02):** montage dan grid denoise menunjukkan citra dengan noise tertinggi adalah tumpukan pakaian atau sampah bertekstur padat, beberapa dengan artefak blok upscale dan watermark stok foto. Estimator Immerkaer di data ini lebih banyak mengukur tekstur daripada noise sensor. Keputusan tanpa denoise tetap berlaku, dengan alasan lebih kuat.
