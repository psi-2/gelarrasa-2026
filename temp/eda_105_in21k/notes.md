# EDA 105: ketersediaan kosakata lebih kaya

**Pertanyaan**: adakah model non-VLM di cache dengan kosakata lebih cocok dari ImageNet-1k?

**Hasil** (`cek_bobot.py`, tanpa memuat citra):

- Head ImageNet-21k TIDAK tersedia lengkap. `timm/convnext_large.fb_in22k` hanya blob `.incomplete` 34,5 MB. `microsoft/swinv2-large-patch4-window12-192-22k`, `facebook/convnext-base-224-22k` hanya config. `vit_small_patch16_224.augreg_in21k` tidak ada bobot.
- Tersedia lengkap: EoMT DINOv3 COCO panoptic base 369,7 MB (133 kelas), EoMT DINOv3 ADE large 1.259,8 MB (150 kelas).
- Model terlarang dilewati oleh filter nama: clip, siglip, siglino, sam3, radio, steervit, serta model waste.
- Definisi dan lemma synset ImageNet-21k tersedia offline di paket timm (`imagenet_synset_to_definition.txt`, `imagenet_synset_to_lemma.txt`).

**Usulan**: unduh bobot `timm/convnext_large.fb_in22k`. Butuh persetujuan karena unduhan berkas, belum dilakukan.
