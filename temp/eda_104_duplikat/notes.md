# EDA 104: duplikat dan near-duplicate (Hipotesis C)

**Metode**: `hash_dup.py` pHash 63 bit pada 8 transformasi dihedral. `emb_dup.py` tetangga cos DINOv3 dalam dan antar split.

**Hasil**:

- Pasangan Hamming <= 10: 400 (train-train 272, train-test 105, test-test 23). Hamming 0: 95 pasangan.
- Komponen unik pHash: 2.055 dari 2.400. 560 citra ada dalam 215 keluarga, terbesar 53.
- 82 citra test punya keluarga hash yang memuat citra train (17,1%).
- Transformasi terbaik tersebar merata di 8 dihedral (113, 52, 38, 49, 45, 35, 41, 27), jadi ada keluarga augmentasi flip dan rotasi.
- Cos test ke train: >= 0,99 4,6%, >= 0,95 11,0%, >= 0,90 23,5%, >= 0,85 48,3%.
- Citra unik per ambang cos: 0,99 2.359 komponen, 0,95 2.102, 0,90 1.657, 0,85 1.051 (komponen terbesar 435, sudah merantai).
- NN test ke train: keluarga sumber sama 59,4%, nama cluster sama 91,7%.

**Implikasi**: split validasi wajib ber-group. Group berbasis cos 0,9 sudah dipakai. pHash dihedral perlu ditambahkan karena flip dan rotasi bisa lolos dari cos. Hold-out dari satu sumber saja akan terlalu optimistis.

## Koreksi (`konsistensi.py`)

Ambang Hamming <= 10 terlalu longgar. Keluarga 53 dan 15 anggota adalah rantai positif palsu dari citra berlatar polos: isinya campur sumber dan nama, `lok_halus` median 0,749 dan 0,816. Konsistensi label otomatis pada pasangan: Hamming <= 4 sekitar 0,98, Hamming <= 10 hanya 0,63.

Akibatnya:

- Angka "2.055 komponen unik", "82 citra test (17,1%)", dan "keluarga terbesar 53" di atas terlalu dipengaruhi rantai palsu. Jangan dikutip.
- Angka yang layak: pasangan Hamming <= 4 adalah train-train 100, train-test 39, test-test 19.
- Group validasi memakai Hamming <= 4. Dengan ambang 10 group terbesar 213, dengan ambang 4 group terbesar 60 (1.318 group).
