# EXP-008: inventaris classifier dan sumber label tambahan

**Tujuan**: menyiapkan sumber label baru bila normalisasi prior tidak cukup.

**Hasil**:

- Cache tidak punya head ImageNet-22k dengan bobot lengkap.
- Diunduh ke `D:\cache_moved\huggingface\hub`: `timm/convnext_large.fb_in22k`, `timm/swinv2_large_window12_192.ms_in22k`, `timm/beitv2_large_patch16_224.in1k_ft_in22k`.
- Diunduh ke `D:\cache_moved\dms46`: DMS46 dari Apple, 46 kelas material.
- `eva02_large_patch14_448.mim_m38m_ft_in22k` sempat diunduh, lalu dipindahkan ke Recycle Bin karena target MIM-nya EVA-CLIP dan melanggar NO VLM.

**Kesimpulan**: `infer22k.py` dan `infer_dms.py` belum dijalankan, karena EXP-009 sudah melewati target. Keduanya jadi langkah berikutnya bila skor Kaggle lebih rendah dari skor temuan.
