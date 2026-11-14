"""PREP 302: verifikasi visual pipeline final notebook pada strata cluster Run 1 (train-only, tanpa nama kelas).

prep_301 memakai strata label lama (sekarang tidak dipakai) dan reimplementasi PIL. Di sini kode trim, load, dan
transform torchvision disalin persis dari notebook, lalu dijalankan pada 1 citra per cluster Run 1 dan pada kasus
tepi: trim terbesar, objek kecil, aspek ekstrem, dan citra sangat kecil. Hanya transform, tanpa model.
Keluaran: output/grid_cluster_<hal>.png, output/grid_<kasus>.png, output/ringkas.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision.transforms import v2 as T

from eda_common import ROOT, TEMP, buka, out_dir, path_train
from pipa import PREP, aug, muat_citra, tf_eval_tampil, trim_hitam

OUT = out_dir(__file__)
RRC_SCALE = (float(sys.argv[1]) if len(sys.argv) > 1 else PREP["rrc_scale"][0], 1.0)
tf_tampil = T.Compose([T.ToImage(), *aug(RRC_SCALE)])
torch.manual_seed(42)


def tampil(x):
    return x.permute(1, 2, 0).numpy() if torch.is_tensor(x) else x


def grid(ids, judul, nama, n_aug=3, keterangan=None):
    kol = ["asli", "muat 512 (trim)", "eval 384"] + [f"latih {j + 1}" for j in range(n_aug)]
    fig, ax = plt.subplots(len(ids), len(kol), figsize=(len(kol) * 1.9, len(ids) * 1.95), squeeze=False)
    for r, i in enumerate(ids):
        x = muat_citra(path_train(i))
        panel = [buka(path_train(i), 300), x, tampil(tf_eval_tampil(x))] + [tampil(tf_tampil(x)) for _ in range(n_aug)]
        for c, (a, im) in enumerate(zip(ax[r], panel)):
            a.imshow(im)
            a.set_title(f"{i} {kol[c]}" + (f" {keterangan[r]}" if keterangan is not None and c == 0 else ""), fontsize=6)
            a.axis("off")
    fig.suptitle(judul, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"grid_{nama}.png", dpi=80)
    plt.close(fig)


cl = pd.read_csv(ROOT / "results/cache/cluster_train.csv")
satu = cl[cl.stab >= .7].groupby("cluster").sample(1, random_state=7).sort_values("cluster")
for h in range(4):
    d = satu.iloc[h * 12:(h + 1) * 12]
    grid(d.image_id.tolist(), f"pipeline final per cluster Run 1 (hal {h + 1}/4)", f"cluster_{h + 1}", keterangan=[f"c{k}" for k in d.cluster])

meta = pd.read_csv(TEMP / "eda_101_metadata/output/meta.csv").query("split == 'train'").astype({"image_id": int}).set_index("image_id")
lat = pd.read_csv(TEMP / "eda_204_background/output/latar.csv").query("split == 'train'").set_index("image_id")
tr = pd.read_csv(TEMP / "prep_301_verifikasi/output/trim_hitam.csv").query("split == 'train'").set_index("image_id")
kasus = {
    "trim_terbesar": tr[tr.trim].porsi_trim.nlargest(10).index.tolist(),
    "objek_kecil": lat[(lat.tipe != "bertekstur") & (lat.fg_frac < .1)].sample(10, random_state=1).index.tolist(),
    "aspek_ekstrem": meta.ar[(meta.ar > 1.6) | (meta.ar < .62)].sample(10, random_state=1).index.tolist(),
    "sangat_kecil": meta.index[meta[["w", "h"]].min(1) < 200].tolist()[:10],
}
ringkas = {"n_trim_train": int(tr.trim.sum()), "n_objek_kecil_polos": int(((lat.tipe != "bertekstur") & (lat.fg_frac < .1)).sum()),
           "n_aspek_ekstrem": int(((meta.ar > 1.6) | (meta.ar < .62)).sum()), "n_sisi_pendek_lt200": int((meta[["w", "h"]].min(1) < 200).sum())}
for nama, ids in kasus.items():
    ket = [f"ar{meta.ar[i]:.2f} {meta.w[i]}x{meta.h[i]}" for i in ids]
    grid(ids, f"kasus tepi: {nama}", nama, n_aug=4, keterangan=ket)

# ukuran setelah trim: trim tidak boleh memotong lebih dari porsi yang diperkirakan prep_301
ukur = []
for i in tr[tr.trim].index:
    with Image.open(path_train(i)) as im:
        a = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    b = trim_hitam(a)
    ukur.append(1 - b.shape[0] * b.shape[1] / (a.shape[0] * a.shape[1]))
ukur = np.array(ukur)
ringkas["porsi_luas_dibuang_trim"] = {"median": round(float(np.median(ukur)), 3), "maks": round(float(ukur.max()), 3),
                                     "sama_dengan_prep301": bool(np.allclose(np.sort(ukur), np.sort(tr[tr.trim].porsi_trim.values), atol=.02))}
ringkas["cluster_berisi_trim"] = cl.set_index("image_id").loc[tr[tr.trim].index].cluster.value_counts().head(6).to_dict()
print(json.dumps(ringkas, indent=1))
json.dump(ringkas, open(OUT / "ringkas.json", "w"), indent=1)
print("simpan grid_*.png di", OUT)
