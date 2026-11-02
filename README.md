# DSC Gelar Rasa 2026

Ringkasan kompetisi ada di [docs/context.md](docs/context.md), batasan kerja terbaru di [docs/constraint.md](docs/constraint.md), dan instruksi melanjutkan eksperimen di [docs/HANDOFF.md](docs/HANDOFF.md).

Data dan sample submission sudah tersimpan di `data/`. Riwayat skrip, hasil, cache embedding terpilih, dan notebook tersimpan di `temp/`. Lihat [log_exp.md](log_exp.md) untuk indeks riwayat.

Catatan: notebook dan skor ensemble lama hanya arsip. Ikuti batasan terbaru di `docs/constraint.md`.

## Progres yang dibagikan — 3 November 2026

- Notebook terbaru: [notebook/dsc_gelarrasa_2026_dinov3.ipynb](notebook/dsc_gelarrasa_2026_dinov3.ipynb).
- Run 1 Kaggle sudah menghasilkan embedding, bukti cluster, dan gambar evaluasi di [results/](results/). DINOv3 7B gagal memenuhi tiga syarat seleksi yang dibekukan; ViT-L dipilih dengan K=48. Rincian metrik ada di [log_exp.md](log_exp.md), baris EXP-014. Hasil ini belum membuktikan training classifier final atau submission final selesai.
- Referensi notebook dan template ada di [ref/](ref/).
- Data, skrip, catatan EDA, notebook lama, dan hasil eksperimen historis tetap tersedia di `data/` dan `temp/`.

Cache hasil run dan environment lokal disimpan sebagai ZIP yang dibagi menjadi bagian maksimal 40 MiB di [archives/](archives/README.md). Petunjuk penggabungan, ekstraksi, serta pemeriksaan checksum ada di sana. Daftar paket lokal tersedia di [requirements-snapshot.txt](requirements-snapshot.txt); kebutuhan GPU dijelaskan di [docs/environment.md](docs/environment.md).

Snapshot mencakup berkas proyek yang ada di direktori kerja, termasuk cache bytecode lokal dalam arsip. Metadata Git, direktori konfigurasi/kredensial lokal (`.agents/`, `.codex/`, `.aws/`), serta berkas rahasia tidak disertakan. Bobot atau cache eksternal yang tidak ada di direktori ini tidak termasuk.
