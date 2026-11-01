# Riset pembedaan material untuk penamaan kelas sampah

Status: selesai untuk tahap riset, belum diuji pada data. Tanggal: 2026-10-31.

Dokumen ini menjawab satu pertanyaan: sumber sinyal apa, selain head ImageNet-1k, yang bisa membedakan `paper`, `plastic`, `glass`, dan `metal` tanpa VLM dan tanpa label. Alasannya datang dari EXP-007: 39 dari 70 kesalahan EXP-006 adalah `paper` (24) dan `glass` (15) yang menjadi `plastic`. Head ImageNet-1k memang tidak punya kelas `paper`, `cardboard`, `plastic`, atau kaleng generik, jadi perbaikan algoritma di atas sinyal yang sama tidak bisa menambah informasi ini.

## Ringkasan kandidat

| Sumber sinyal | Kelas relevan | Ukuran | Status bobot | Cache lokal | Status aturan |
|---|---|---|---|---|---|
| DMS46 (Apple, ECCV 2022) | paper, cardboard, plastic, clear plastic, glass, metal, fabric, food | zip 173,7 MB (`Content-Length` 173.688.250) | HTTP 200 per 2026-10-31 | belum ada | Boleh: citra saja, data umum bukan sampah |
| Open Images V2 ResNet-101 (Google) | 5.000 kelas: Paper, Cardboard, Glass, Metal, Textile, Waste, Scrap, Aluminum can, Tin can, Plastic bottle, Plastic bag, Glass bottle, Carton, Electronics, Food, Clothing | 193,6 MB | HTTP 200, format TF-Slim | belum ada | Boleh: citra saja, sigmoid multi-label |
| Open Images V6 TResNet-L ASL (Alibaba) | 9.605 kelas, tambah Plastic, Recycling, Compost | tidak diketahui | **HTTP 403 pada 2026-10-31**, mirror belum ditemukan | belum ada | Boleh bila bobot didapat |
| Tencent ML-Images ResNet-101 | 11.166 kelas dengan hierarki WordNet: paper, litter, garbage heap, fabric, printed circuit, tin can, beer can, plastic bag, carton | tidak diketahui | Google Drive, TF1 + Python 2.7, belum diuji | belum ada | Boleh, biaya konversi tinggi |
| MINC Deep TEN ResNet-50 | paper, plastic, glass, metal, fabric, food, Other | 94,2 MB | HTTP 200, PyTorch | belum ada | Boleh: citra saja, Flickr dan Houzz |
| YOLOv8x Open Images V7 | Tin can, Plastic bag, Bottle, Box, Laptop, Mobile phone, Food, Clothing | 137,9 MB | HTTP 200, AGPL-3.0 | belum ada | Boleh: detektor citra saja |
| Head ImageNet-12k/21k/22k (timm) | paper, tin can, beer can, soda can, tinfoil, paper bag, paper cup, cellophane, plastic wrap, printed circuit, fabric, garbage | 234 MB sampai 2 GB | HTTP 200 di HF | **head belum ada**, hanya varian `ft_in1k` | Boleh |

## DMS46: kelas material hampir sama dengan label target

Sumber: Upchurch dan Niu, ECCV 2022, "A Dense Material Segmentation Dataset for Indoor and Outdoor Scene Parsing", https://github.com/apple/ml-dms-dataset. Taksonomi dibaca dari `taxonomy.json`, angka dari Tabel 4 dan Tabel 8 paper.

Model: ResNet-50 dilated dengan Pyramid Pooling Module, input 512, 46 material, TorchScript `DMS46_v1.pt`. Dilatih pada 24.255 citra dari total 44.560 citra indoor dan outdoor umum. Skor test: per-pixel accuracy 0,729, mean class accuracy 0,585, mIoU 0,420.

| Kelas DMS | Acc test | IoU test | Kesepakatan anotator | Target |
|---|---|---|---|---|
| Food | 0,868 | 0,668 | 0,84 | foodwaste |
| Fabric/cloth | 0,801 | 0,692 | 0,82 | textile |
| Paper | 0,729 | 0,508 | 0,76 | paper |
| Glass | 0,653 | 0,483 | 0,80 | glass |
| Metal | 0,575 | 0,303 | 0,65 | metal |
| Cardboard | 0,570 | 0,363 | 0,53 | cardboard |
| Non-clear plastic | 0,519 | 0,321 | 0,60 | plastic |
| Clear plastic | 0,360 | 0,222 | 0,50 | plastic |
| Clutter | 0,182 | 0,152 | tidak dilaporkan | abaikan |

Catatan:

- Angka di atas per piksel pada adegan penuh. Sinyal per citra sampah didapat dari agregasi piksel objek, jadi akurasi per citra bisa berbeda arah dan belum terukur.
- `electronic` tidak punya kelas material. Anotator DMS diminta melabeli layar elektronik sebagai `glass`, jadi citra elektronik bisa memberi sinyal `glass` palsu. Kelas ini harus diambil dari head objek, bukan DMS.
- `Clear plastic` lemah (0,360) dan inilah pembeda yang paling dibutuhkan untuk `glass` lawan `plastic`. DMS diperkirakan lebih membantu `paper` lawan `plastic` daripada `glass` lawan `plastic`.
- Lisensi kode dan model: lisensi permisif kustom Apple, tanpa batasan riset saja. Dataset CC-BY-NC, tetapi dataset tidak dipakai.
- Hanya bobot yang dipakai, sehingga rule 4 tidak terlanggar dan rule 6 mengizinkan pretrained model yang dijalankan sendiri.

SegFormer-B5 pada DMS mencapai mIoU 0,4572 (arXiv 2605.23747), tetapi paper itu hanya merilis indeks data dan kode, tanpa bobot. Melatih ulang berarti memakai data eksternal, jadi jalur ini dicoret.

## Open Images: label langsung bernama material

Verifikasi per 2026-10-31 dengan membaca `oidv6-class-descriptions.csv`, `oidv6-classes-trainable.txt` (95.985 byte), dan `2017_07/classes-trainable.txt` lewat WebFetch. Pemeriksaan keanggotaan MID memakai pembacaan model ringkas, jadi implementasi wajib mengulang cek deterministik dengan script sebelum memakainya.

| Nama | MID | V6 9.605 | V2 5.000 |
|---|---|---|---|
| Plastic | /m/05z87 | ada | **tidak ada** |
| Paper | /m/0641k | ada | ada |
| Cardboard | /m/03q7pgh | ada | ada |
| Glass | /m/039jq | ada | ada |
| Metal | /m/04t7l | ada | ada |
| Textile | /m/0dnr7 | ada | ada |
| Waste | /m/01fnck | ada | ada |
| Scrap | /m/043520 | ada | ada |
| Aluminum can | /m/0gjbv4m | ada | ada |
| Tin can | /m/02jnhm | ada | ada |
| Plastic bottle | /m/02rlncx | ada | ada |
| Plastic bag | /m/05gqfk | ada | ada |
| Glass bottle | /m/089mxq | ada | ada |
| Carton | /m/033rv2 | ada | ada |
| Electronics | /m/02mrp | ada | ada |
| Electronic device | /m/0bs7_0t | ada | ada |
| Food | /m/02wbm | ada | ada |
| Clothing | /m/09j2d | ada | ada |
| Newspaper | /m/05jnl | ada | ada |
| Litter | /m/03ns2k | ada | ada |
| Recycling | /m/0j6v7 | ada | tidak ada |
| Compost | /m/01ss0 | ada | tidak ada |

Risiko V6: isu ASL #26 memperkirakan hanya sekitar 5.500 dari 9.604 kelas yang punya data, jadi keberadaan kelas di head bukan jaminan kelas itu terlatih.

Risiko V2: `Plastic` generik tidak ada. Sinyal plastik harus berasal dari `Plastic bottle`, `Plastic bag`, dan sejenisnya, sehingga evidence menjadi asimetris dan perlu kalibrasi per kelas. Input VGG-style 299 px, keluaran sigmoid per kelas. Model memakai TF-Slim, jadi perlu TensorFlow di env terpisah atau konversi bobot ke PyTorch.

## Head ImageNet-12k/21k

Audit di `temp/research_unsup/audit_label_21k.py` memakai daftar kelas paket timm, hasil di `temp/research_unsup/output/audit_21k_kandidat.tsv`.

| Lemma | Ada di | Lemma | Ada di |
|---|---|---|---|
| paper (n14974264) | 12k | tin can | 12k |
| newspaper | 12k | beer can | 12k |
| fabric, cloth, material, textile | 12k | soda can | 12k |
| paper bag | 12k | tinfoil | 12k |
| paper cup | 12k | printed circuit | 12k |
| plastic wrap | 12k | battery | 12k |
| cellophane | 12k | garbage | 12k |
| jar, bottle | 12k | garbage heap | 12k |
| food | 12k | landfill, junk, litter | 12k |
| `cardboard`, `plastic`, `metal`, `compost`, `motherboard`, `smartphone` | tidak ada sebagai lemma | carton | 1k |

Kekurangan head 21k: `plastic` generik dan `cardboard` tidak ada sebagai kelas. Karena itu head 21k saja tidak menutup masalah `plastic`, dan perlu digabung dengan DMS atau Open Images.

## MINC: satu-satunya model material dengan kelas `plastic` generik yang kuat

Sumber: Bell dkk. CVPR 2015, arXiv 1412.0623. 23 kategori: Brick, Carpet, Ceramic, Fabric, Foliage, Food, Glass, Hair, Leather, Metal, Mirror, Other, Painted, Paper, Plastic, Polished stone, Skin, Sky, Stone, Tile, Wallpaper, Water, Wood. Sumber citra: Flickr (OpenSurfaces) dan Houzz, bukan data sampah.

| Kelas MINC | Akurasi patch GoogLeNet | Target |
|---|---|---|
| Food | 90,4% | foodwaste |
| Paper | 81,8% | paper, cardboard (MINC tidak punya cardboard) |
| Glass | 78,5% | glass |
| Fabric | 77,8% | textile |
| Metal | 77,7% | metal |
| Plastic | 70,9% | plastic |

Mean class accuracy 85,2% pada patch. Dua varian bobot:

| Varian | Ukuran | Status | Catatan |
|---|---|---|---|
| Deep TEN ResNet-50 (`deepten_resnet50_minc-1225f149.zip`, PyTorch-Encoding) | 94,2 MB | HTTP 200 di Wasabi, 2026-10-31 | top-1 MINC-2500 81,04%, top-5 95,62%, input 224. Disarankan |
| GoogLeNet, VGG-16, AlexNet Caffe (`minc-model.tar.gz`) | 748 MB | HTTP 200 | perlu konversi Caffe, 85,2% mean class pada GoogLeNet |

Catatan implementasi:

- Bell dkk. memakai patch 23,3% dari sisi terpendek. Deep TEN dilatih pada patch MINC-2500. Citra penuh yang di-resize ke 224 berbeda skala dari data latih, jadi jalankan sliding crop 3 skala, misalnya 25%, 50%, dan 100% sisi terpendek, lalu rata-ratakan probabilitas di area objek.
- `pip install torch-encoding` berisiko gagal pada torch 2.14. Lebih aman menyalin definisi `deepten.py` beserta layer Encoding ke script lokal, lalu memuat `state_dict`.
- Lisensi bobot tidak disebut pada halaman yang terbaca. Anotasi MINC CC BY 4.0. Lisensi repo PyTorch-Encoding perlu dicek sebelum dipakai.
- `prithivMLmods/Minc-Materials-23` adalah classifier SigLIP. TIDAK BOLEH.

## YOLOv8 Open Images V7 sebagai bukti objek tingkat kotak

`yolov8x-oiv7.pt`, 137,9 MB, HTTP 200, lisensi AGPL-3.0, mAP50-95 36,3, 601 kelas. Kelas relevan terverifikasi dari YAML Ultralytics: Bottle (57), Box (62), Clothing (115), Computer keyboard (127), Computer monitor (128), Drink (171), Envelope (180), Fast food (186), Food (199), Fruit (210), Jeans (283), Laptop (304), Mobile phone (339), Mug (345), Paper towel (368), Plastic bag (394), Remote control (420), Shirt (455), Television (527), Tin can (535), Toilet paper (539), Towel (545), Vegetable (566), Waste container (576), Wine glass (590). Tidak ada Carton atau Cardboard.

Nilai tambah utama adalah `Tin can`, yang menutup lubang kaleng di ImageNet-1k (EXP-005 terpaksa memakai `can opener`), dan `Plastic bag`. Detektor ini juga memberi kotak untuk cropping.

## FMD

FMD punya 10 kategori, termasuk fabric, glass, metal, paper, plastic. Tidak ditemukan bobot pretrained yang bisa langsung diunduh. Model vision-only DINOv2 pada arXiv 2603.17390 mencapai 0,79 mIoU dan 0,88 mAcc pada FMD, tetapi dilatih pada data sintetis 20 ribu citra dengan bobot yang tidak disebut dirilis, dan versi penuhnya memakai CLIP plus deskripsi GPT-4V. Versi penuh TIDAK BOLEH. Versi vision-only tidak bisa dipakai tanpa bobot. Melatih sendiri pada FMD melanggar rule 4.

## Peran tiap sumber terhadap kesalahan EXP-006

| Kesalahan | Sumber yang diharapkan membantu | Alasan | Keyakinan |
|---|---|---|---|
| `paper` ke `plastic` (24) | MINC Paper 81,8%, DMS Paper 0,729, OI `Paper` | ketiganya punya kelas kertas eksplisit, tidak ada di head 1k | sedang-tinggi |
| `glass` ke `plastic` (15) | OI `Glass bottle` lawan `Plastic bottle`, MINC Glass 78,5% lawan Plastic 70,9% | DMS `Clear plastic` lemah (0,360), botol PET dan kaca mirip, ini tetap kasus tersulit | sedang-rendah |
| `metal` lawan `plastic` | YOLO `Tin can`, OI `Aluminum can`, 12k `beer can` dan `soda can` | head 1k tidak punya kaleng | sedang |
| `paper` lawan `cardboard` | DMS Cardboard 0,570 dan Paper 0,729, head `carton` | hanya DMS memisahkan keduanya | sedang |

Fitur fisik buatan tangan seperti specular highlight atau transmisi cahaya tidak diprioritaskan. Tidak ditemukan bukti kuat bahwa fitur itu memisahkan botol PET dan kaca pada citra sampah beragam, dan model material di atas sudah mempelajari isyarat serupa.

## Kebutuhan unduhan bahan

| Berkas | Ukuran | Prioritas |
|---|---|---|
| `dms46_v1.zip` | 173,7 MB | 1 |
| `deepten_resnet50_minc-1225f149.zip` | 94,2 MB | 1 |
| `yolov8x-oiv7.pt` | 137,9 MB | 2 |
| `oidv2-resnet_v1_101.ckpt.tar.gz`, plus TensorFlow di env terpisah | 193,6 MB | 3 |

Tidak ada env yang punya TensorFlow saat ini (`kaggle-py311-d`, `kaggle-py311`, `he-llm` sudah dicek). Bobot TResNet Open Images V6 mengembalikan 403 pada 2026-10-31, begitu juga bobot lain di bucket `miil-public-eu`, dan pencarian mirror tidak menemukan apa pun.
