"""Cek ketersediaan classifier ImageNet-22k resmi di HF Hub, ukuran bobot, dan kepala kelas.

    python temp/exp_008_in22k_map/cek_hub.py
"""
import json
from pathlib import Path

from huggingface_hub import HfApi

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
KANDIDAT = ["timm/convnext_large.fb_in22k", "timm/convnext_xlarge.fb_in22k", "timm/eva02_large_patch14_448.mim_m38m_ft_in22k",
            "timm/beitv2_large_patch16_224.in1k_ft_in22k", "timm/swinv2_large_window12_192.ms_in22k",
            "timm/tf_efficientnetv2_l.in21k", "timm/tf_efficientnetv2_xl.in21k", "timm/vit_large_patch16_224.augreg_in21k",
            "timm/maxvit_rmlp_base_rw_224.sw_in12k", "timm/convnextv2_large.fcmae_ft_in22k_in1k_384",
            "facebook/convnext-base-224-22k", "microsoft/swinv2-large-patch4-window12-192-22k"]
api = HfApi()
hasil = []
for m in KANDIDAT:
    try:
        info = api.model_info(m, files_metadata=True, timeout=30)
        berkas = {s.rfilename: s.size for s in info.siblings if s.rfilename.endswith((".safetensors", ".bin")) }
        hasil.append({"model": m, "lisensi": (info.card_data or {}).get("license") if info.card_data else None,
                      "berkas": {k: round(v / 1e9, 2) for k, v in berkas.items() if v}})
        print(m, hasil[-1]["lisensi"], hasil[-1]["berkas"])
    except Exception as e:
        print(m, "GAGAL", type(e).__name__, str(e)[:100])
(OUT / "cek_hub.json").write_text(json.dumps(hasil, indent=2), encoding="utf-8")
