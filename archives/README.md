# Arsip snapshot — 3 November 2026

Berkas ZIP besar dibagi per 40 MiB agar setiap bagian dapat disimpan di GitHub. Semua bagian diperlukan; bagian `.zip.part001` bukan ZIP mandiri. Berkas sumber lokal tetap berada di tempat semula.

| Arsip | Isi |
| --- | --- |
| `cache.zip.part*` | Seluruh `results/cache/`, serta cache bytecode lokal di `temp/__pycache__/` dan `temp/eda_204_background/__pycache__/`. |
| `environment.zip.part*` | Snapshot `.venv/`, termasuk symlink. |

Cache yang sebelumnya sudah dilacak Git di `temp/` tetap tersedia langsung pada jalur aslinya. `manifest.json` mencatat ukuran, SHA-256, serta daftar berkas dan symlink untuk setiap ZIP.

## Mengembalikan cache

Jalankan dari root repo pada Linux/macOS setelah clone. Gunakan direktori clone baru agar ekstraksi tidak menimpa perubahan lokal.

```sh
sha256sum -c archives/SHA256SUMS
cat archives/cache.zip.part* > /tmp/gelarrasa-cache.zip
unzip -t /tmp/gelarrasa-cache.zip
unzip /tmp/gelarrasa-cache.zip -d .
```

Urutan bagian mengikuti nama berkas (`part001`, `part002`, dan seterusnya). Pada Windows, gabungkan bagian sebagai data biner sesuai urutan tersebut, lalu buka ZIP gabungannya dengan aplikasi ZIP.

## Environment lokal

Arsip environment ditujukan untuk menyimpan keadaan mesin sumber. Symlink interpreter dan skrip aktivasi dapat menunjuk ke jalur absolut mesin sumber; environment ini tidak dijamin langsung berjalan di komputer lain. Untuk menjalankan proyek, buat environment Python 3.12 baru dengan paket dalam `requirements-snapshot.txt`, kemudian ikuti kebutuhan GPU di `docs/environment.md`.

Untuk memeriksa arsip environment, ekstrak ke direktori terpisah:

```sh
cat archives/environment.zip.part* > /tmp/gelarrasa-environment.zip
unzip -t /tmp/gelarrasa-environment.zip
mkdir -p /tmp/gelarrasa-environment-snapshot
unzip /tmp/gelarrasa-environment.zip -d /tmp/gelarrasa-environment-snapshot
```

Kredensial, konfigurasi lokal di `.agents/`, `.codex/`, `.aws/`, dan metadata `.git/` tidak diarsipkan. Bobot model dan cache di luar direktori repo juga tidak termasuk.
