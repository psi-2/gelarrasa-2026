"""Uji logika notebook 7B QLoRA di CPU, tanpa model asli dan tanpa GPU.

    <venv dengan timm 1.0.29, peft, bitsandbytes, torch CPU>/python temp/nb_7b_qlora/uji_logika.py nf4|bf16 <folder_kerja>

Sel kode notebook dieksekusi berurutan apa adanya, dengan pengganti berikut:
- arsitektur vit_7b_patch16_dinov3 diperkecil (dim 64, 2 blok, 4 head) dan bobot acaknya ditulis ke safetensors;
- hf_hub_download mengembalikan berkas itu (tanpa unduhan), sel pip dan nvidia-smi dilewati;
- sel pemeriksaan GPU diganti stub (DEV = cpu), fungsi torch.cuda khusus GPU dijadikan no-op;
- citra disimpan 96 px, input 64 px, 1 epoch, num_workers 0, batas waktu dilonggarkan, pseudo-label dinyalakan.
Setelah itu: uji lanjut-dari-checkpoint, tangga fallback OOM/waktu, OOM saat training, kesetaraan preprocessing
dengan pipa.py, matematika kepala_lp, dan format submission terhadap sample.
"""
import json
import os
import re
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
MODE, KERJA = sys.argv[1], Path(sys.argv[2])
AKAR = Path(__file__).resolve().parents[2]
KELUARAN = KERJA / f"keluaran_{MODE}"
shutil.rmtree(KELUARAN, ignore_errors=True)
os.environ.update(DSC_KELUARAN=str(KELUARAN), HF_HOME=str(KERJA / "hf"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import huggingface_hub  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

plt.show = lambda *a, **k: plt.close("all")
for nama in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
    setattr(torch.cuda, nama, lambda *a, **k: None)
torch.cuda.max_memory_allocated = lambda *a, **k: 0
torch.cuda.memory_allocated = lambda *a, **k: 0
_autocast = torch.autocast
# di build CPU autocast('cuda') mati diam-diam; GPU asli meng-cast semua matmul ke bf16, jadi tiru dengan autocast CPU
torch.autocast = lambda device_type, dtype=None, enabled=True, **k: _autocast("cpu", dtype=dtype, enabled=enabled)

_buat = timm.create_model


def buat_kecil(nama, *a, **k):
    if nama == "vit_7b_patch16_dinov3":
        k.update(embed_dim=64, depth=2, num_heads=4)
    return _buat(nama, *a, **k)


timm.create_model = buat_kecil
torch.manual_seed(0)
ref = buat_kecil("vit_7b_patch16_dinov3", pretrained=False, num_classes=0)
BOBOT = KERJA / "bobot_kecil.safetensors"
save_file({k: v.contiguous() for k, v in ref.state_dict().items()}, str(BOBOT))
panggilan = []


def unduh_palsu(repo, nama_berkas, revision=None, local_files_only=False, **k):
    panggilan.append((repo, nama_berkas, revision, local_files_only))
    if local_files_only:
        raise FileNotFoundError("belum ada di cache (simulasi)")
    return str(BOBOT)


huggingface_hub.hf_hub_download = unduh_palsu

nb = json.loads((AKAR / "notebook/dsc_gelarrasa_2026_dinov3_7b_qlora.ipynb").read_text(encoding="utf-8"))
sel = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
os.chdir(AKAR / "notebook")
ns = {}


def jalankan(src):
    exec(compile(src, "<sel>", "exec"), ns)


for i, src in enumerate(sel):
    if src.lstrip().startswith(("!", "%")):
        continue
    if "def uji_kernel" in src:
        fungsi = src[src.index("def uji_kernel"):src.index("try:")]
        jalankan("DEV = torch.device('cpu')\nVRAM_GIB = 96.0 if '%s' == 'bf16' else 32.0\nSM = 'sm_120'\n"
                 "INFO_GPU = {'gpu': 'stub CPU'}\n" % MODE + fungsi
                 + "assert uji_kernel()\nMUAT = CFG.MUAT if CFG.MUAT != 'auto' else ('bf16' if VRAM_GIB >= CFG.VRAM_BF16_GIB else 'nf4')\n"
                 + f"assert MUAT == '{MODE}', MUAT\nprint('stub GPU, mode', MUAT)")
        continue
    jalankan(src)
    if "class Settings" in src:
        C = ns["CFG"]
        C.UKURAN_SIMPAN, C.UKURAN, C.UKURAN_CADANGAN, C.EPOCH, C.NUM_WORKERS = 96, 64, (48, 32), 1, 0
        C.LP_MIN, C.BATAS_FINAL_JAM, C.BATAS_CV_JAM, C.JALANKAN_PSEUDO, C.AMBANG_PSEUDO = 0.0, 100, 100, True, 0.3
        assert C.OUTPUT_DIR == KELUARAN, C.OUTPUT_DIR
    print(f"[sel kode {i} OK]", flush=True)

g = ns
print("\n=== pemeriksaan pasca-run ===")
assert panggilan[0][3] and not panggilan[1][3] and panggilan[1][2] == g["CFG"].M7B[1], panggilan
assert g["n_modul"] == 10 and g["MUAT"] == MODE
tipe = type(g["m"].blocks[0].attn.qkv.base_layer).__name__
assert tipe == ("Linear4bit" if MODE == "nf4" else "Linear"), tipe
assert all(p.dtype == torch.float32 for p in g["LORA"].values())
contoh = pd.read_csv(AKAR / "data/SampleSubmission_DSC.csv", dtype=str, encoding="utf-8-sig")
sub = pd.read_csv(KELUARAN / "submission.csv", dtype=str)
assert list(sub.columns) == ["image_id", "waste_category"] and sub.image_id.tolist() == contoh.image_id.tolist()
assert (KELUARAN / "submission.csv").read_text(encoding="utf-8").startswith("image_id,waste_category\n001,")
assert (KELUARAN / "submission_pseudo.csv").exists() and g["INFO_PSEUDO"]["n"] > 0
ad = load_file(str(KELUARAN / "adapter_lora_final.safetensors"))
assert sum(k.startswith("lora.") for k in ad) == len(g["LORA"]) and {"kepala.weight", "kepala.bias"} <= set(ad)
assert not any("base_layer" in k for k in ad), "adapter memuat bobot base"
assert len(g["SELESAI"]) == 5 and np.isfinite(g["ACC_CV"]).all()
assert sorted(np.unique(g["FOLD"])) == [0, 1, 2, 3, 4]
assert (pd.Series(g["FOLD"]).groupby(g["GRUP"][g["L"]]).nunique() == 1).all()
rj = json.loads((KELUARAN / "ringkasan_run.json").read_text(encoding="utf-8"))
assert rj["cv"]["mean"] is not None and rj["mode"] == MODE
print("submission, adapter, CV 5 fold, ringkasan: OK | acc CV", np.round(g["ACC_CV"], 3).tolist())

print("\n=== lanjut dari checkpoint ===")
p_test_lama = g["P_TEST"].copy()
jalankan("RIW_ULANG, _ = latih_aman(L, Y, nama='final')\nP_ULANG = prediksi(X_test)")
assert g["RIW_ULANG"] == g["RIW_FINAL"], (g["RIW_ULANG"], g["RIW_FINAL"])
jalankan("RIW_FOLD3, _ = latih(L[FOLD != 3], Y[FOLD != 3], L[FOLD == 3], Y[FOLD == 3])")
jalankan("RIW_LANJUT, _ = latih_aman(L, Y, nama='final')\nP_LANJUT = prediksi(X_test)")
assert np.allclose(g["P_LANJUT"], p_test_lama, atol=1e-5), "checkpoint final tidak memulihkan model final"
sel_cv = next(s for s in sel if "TAG = " in s)
jalankan(sel_cv)
assert g["SELESAI"] == [0, 1, 2, 3, 4]
print("checkpoint final memulihkan LoRA + head identik, fold CV dimuat dari disk: OK")

print("\n=== batas waktu CV dengan MIN_FOLD ===")
for f in range(1, 5):
    next(KELUARAN.glob(f"cv_fold{f}_*.npz")).unlink()
g["CFG"].BATAS_CV_JAM = 0.0
jalankan(sel_cv)
assert g["SELESAI"] == [0, 1], g["SELESAI"]
print("batas 0 jam tetap menjalankan fold 1 sampai MIN_FOLD = 2, lalu berhenti: OK")

print("\n=== tangga fallback ===")
asli_uji = g["uji_langkah"]


def oom_besar(ukuran, bs):
    if ukuran == 64 and bs >= 8:
        raise torch.cuda.OutOfMemoryError("simulasi")
    return asli_uji(ukuran, bs)


g["uji_langkah"] = oom_besar
p = g["pilih_konfigurasi"]()
assert (p["ukuran"], p["bs_mikro"]) == (64, 4), p
g["uji_langkah"] = lambda ukuran, bs: (1e4, 1.0) if ukuran == 64 else asli_uji(ukuran, bs)
p = g["pilih_konfigurasi"]()
assert p["ukuran"] == 48, p
g["uji_langkah"] = lambda ukuran, bs: (0.1, 1e3)
try:
    g["pilih_konfigurasi"]()
    raise AssertionError("seharusnya gagal")
except RuntimeError:
    pass
g["uji_langkah"] = asli_uji
print("OOM -> batch mikro turun, waktu -> resolusi turun, VRAM penuh -> berhenti: OK")

print("\n=== OOM saat training ===")
asli_latih = g["latih"]
n_oom = [0]


def latih_oom(*a, **k):
    if n_oom[0] == 0:
        n_oom[0] += 1
        raise torch.cuda.OutOfMemoryError("simulasi")
    return asli_latih(*a, **k)


g["latih"] = latih_oom
g["BS_MIKRO"], g["AKUM"] = 16, 1
jalankan("kecil = np.concatenate([L[Y == k][:8] for k in range(len(CFG.KELAS))])\n"
         "RIW_OOM, _ = latih_aman(kecil, Y[np.searchsorted(L, kecil)])")
assert g["BS_MIKRO"] == 8 and g["AKUM"] == 2, (g["BS_MIKRO"], g["AKUM"])
g["latih"], g["BS_MIKRO"], g["AKUM"] = asli_latih, 16, 1
print("latih_aman menurunkan batch mikro ke 8 x akumulasi 2: OK")

print("\n=== kepala_lp dan preprocessing ===")
F_, y_ = g["F_TRAIN"][g["L"]], g["Y"]
W, c = g["kepala_lp"](F_, y_)
lr = g["LogisticRegression"](C=g["CFG"].LP_C, max_iter=3000).fit((F_ - F_.mean(0)) / F_.std(), y_)
assert np.allclose(F_ @ W.T + c, lr.decision_function((F_ - F_.mean(0)) / F_.std()), atol=1e-3)
sys.path.insert(0, str(AKAR / "temp/prep_302_verifikasi_run1"))
import pipa  # noqa: E402

g["CFG"].UKURAN_SIMPAN = 512
for i in range(0, 1920, 97):
    assert np.array_equal(g["muat_citra"](g["f_train"][i]), pipa.muat_citra(g["f_train"][i])), i
print("kepala_lp = decision_function regresi logistik, muat_citra identik dengan pipa.py: OK")
print(f"\nSEMUA UJI LOGIKA LULUS ({MODE})")
