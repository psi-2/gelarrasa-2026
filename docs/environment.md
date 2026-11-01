> Snapshot komputer sumber per 2026-10-31. Path conda, cache model, dan GPU di bawah ini hanya referensi; sesuaikan di perangkat lain.

# Environment (terdeteksi 2026-10-31)

## Python aktif
- executable: `E:\datsci\conda_envs\kaggle-py311-d\python.exe`
- versi: 3.11.15
- OS: Windows 10

## Conda env (petunjuk)
- `E:\datsci\conda_envs\kaggle-py311-d` ada=True python=`E:\datsci\conda_envs\kaggle-py311-d\python.exe`
- conda env list: `C:\Users\itsme\miniconda3`, `C:\Users\itsme\miniconda3\envs\gemini_env`, `C:\Users\itsme\miniconda3\envs\he-llm`, `C:\Users\itsme\miniconda3\envs\kaggle-py311`, `E:\datsci\conda_envs\kaggle-py311-d`

## Kaggle credential
- file ditemukan: ['C:\\Users\\itsme\\.kaggle\\kaggle.json']
- env var: {'KAGGLE_USERNAME': False, 'KAGGLE_KEY': False, 'KAGGLE_API_TOKEN': False}
- kaggle CLI: `None`
- catatan manual: `kaggle` tidak ada di PATH, tetapi `<python env> -m kaggle` berjalan dan credential valid (versi 2.2.3, `competitions list` sukses). `Scripts\kaggle.exe` keluar kode 1 tanpa pesan, jangan dipakai.

## GPU
- nvidia_smi: NVIDIA GeForce RTX 5060 Ti, 16311 MiB, 610.62
- torch: 2.14.0+cu130
- torch_cuda: 13.0
- cuda_available: True
- device: NVIDIA GeForce RTX 5060 Ti
- capability: 12.0
- bf16: True

## Cache model
- `D:\cache_moved` ada=True free=190.9 GB
  - isi: .agent_harnesses.json, AIMV2--Large, IJEPA-HUGE, New folder, aimv2_large_clean, bge-m3, birdnet3_preview31, breast-cancer-detector, convnextv2-huge, convnextv2_huge_clean, debertav3--large, hf_audio, hub, huggingface, kaggle_models, keantengswin-v2-large-breast-cancer-classification, lingbot--vision--giant, lingbot-vision-checkpoints, lingbot-vision-main.zip, lingbot-vision-src, lingbot-vision-vit-large, llama.cpp, mdeberta-v3-base, pip, pip_targets, prithivMLmodsAugmented-Waste-Classifier-SigLIP2, qwen3-4b, qwen3-4b-instruct-2507, siglip2--so400m--patch14--384, so400m_p14_clean, tabpfn, timm--convnext_large.dinov3_lvd1689m, timm--vit_large_patch16_dinov3.lvd1689m, timm--vit_large_patch16_dinov3.sat493m, torch, twitter-xlmr-base-sentiment, twitter-xlmr-large-2022, waste_siglip_clean, xet, xlm--roberta--large, xlm-roberta-large
- `D:\cache_moved\huggingface` ada=True free=190.9 GB
  - isi: .agent_harnesses.json, .check_for_skill_update_done, .check_for_update_done, .locks, CACHEDIR.TAG, datasets, hub, models--Qwen--Qwen3-4B-Instruct-2507, models--nvidia--llama-nemotron-rerank-1b-v2, modules, sentence_transformers, siglino_parallel_parts, stored_tokens, token, xet

- Diunduh 2026-10-31 untuk EXP-008: `timm/convnext_large.fb_in22k`, `timm/swinv2_large_window12_192.ms_in22k`, `timm/beitv2_large_patch16_224.in1k_ft_in22k` ke `D:\cache_moved\huggingface\hub`, dan DMS46 Apple ke `D:\cache_moved\dms46`.

## Paket kunci
| paket | versi |
|---|---|
| numpy | 2.4.6 |
| pandas | 2.2.3 |
| polars | 1.39.3 |
| sklearn | 1.6.1 |
| lightgbm | 4.6.0 |
| xgboost | 3.2.0 |
| catboost | 1.2.10 |
| torch | 2.14.0+cu130 |
| torchvision | 0.29.0+cu130 |
| timm | 1.0.27 |
| transformers | 5.15.1 |
| peft | 0.19.1 |
| bitsandbytes | 0.50.0 |
| tabpfn | 9.0.0 |
| tabicl | 2.2.0 |
| shap | 0.51.0 |
| optuna | 4.9.0 |
| markitdown | TIDAK ADA |
| kaggle | 2.2.3 |
| nbformat | 5.10.4 |
| papermill | TIDAK ADA |
