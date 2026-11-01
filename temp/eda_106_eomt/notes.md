# EDA 106: vektor area kelas EoMT

**Tujuan**: kosakata pelengkap ImageNet-1k, memuat `paper-merged`, `cardboard`, `food-other-merged` (COCO) dan `glass`, `bottle`, `bag`, `box` (ADE).

**Kepatuhan**: EoMT hanya menerima citra, tanpa teks, dilatih pada COCO dan ADE20K, bukan dataset sampah. Rule 6 mengizinkan pretrained. Larangan koordinator menyebut ImageNet pretrained boleh dan tidak menyebut COCO atau ADE, jadi pemakaian ini perlu dikonfirmasi koordinator.

**Metode**: `eomt_area.py` resize persegi (640 dan 512), semantik dari kelas query kali mask, simpan area per kelas seluruh citra dan area tengah 50%. Satu job GPU, batas 5 GiB, puncak 1,02 GiB pada uji bs=2. `profil.py` mencetak profil per cluster dan keluarga.

**Keluaran**: `output/{coco,ade}_{train,test}.npy` (N x C x 2), utuh setelah crash.

**Hasil**: kelas per cluster bertentangan dengan nama EXP-006 pada beberapa cluster bernama plastic. Contoh: cluster 1, 43, 37 dominan `paper-merged` dan `book`. Cluster 55 dominan `handbag`, `backpack`, `blanket`. Cluster 3 dominan `oven`, `bowl`, `sink`. Keputusan nama hanya boleh keluar dari aturan otomatis, lihat eda_110.
