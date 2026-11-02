# EDA 201: resolusi dan aspect ratio

Skrip `resolusi.py`, gambar `output/resolusi.png`. Sumber: `meta.csv` eda_101.

| Temuan | Implikasi teori | Tindakan |
|---|---|---|
| Median sisi pendek 416 (train dan test). Sisi pendek < 384: train 11,7%, test 9,4%. Sisi pendek < 448: train 52,9%, test 61,5% | Input 448 meng-upsample lebih dari separuh citra. Upsampling tidak menambah informasi dan memperbesar artefak JPEG q75 | Input 384 untuk 7B, ViT-L, dan ConvNeXt. Ini konsisten dengan catatan lama bahwa 448 lebih buruk |
| Persegi: train 76,6%, test 81,7%. Aspect di luar [0,67; 1,5]: 5,4% dan 3,7% | Distorsi stretch kecil pada mayoritas citra, sedangkan center crop membuang tepi objek pada citra 4:3 | Resize stretch bicubic ke persegi, tanpa center crop (diverifikasi di prep_301) |
| KS sisi pendek D=0,117, megapiksel D=0,169. Keluarga 1599×1599, 400×533, dan 320×320 hanya ada di train | Shift ukuran berasal dari sumber berkas yang hanya ada di train, bukan dari isi citra | Hanya diagnostik. Ukuran berkas tidak dipakai sebagai fitur |
