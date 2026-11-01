"""Deteksi lingkungan kerja: conda env, kaggle credential, GPU, cache model, disk.

Jalankan dari root proyek:
    python3 scripts/utils/detect_env.py --write docs/environment.md
Isi credential tidak pernah dicetak, hanya ada/tidak dan lokasinya.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

HINT_ENVS = [r"E:\datsci\conda_envs\kaggle-py311-d"]
HINT_CACHES = [r"D:\cache_moved"]
KEY_PACKAGES = ["numpy", "pandas", "polars", "sklearn", "lightgbm", "xgboost", "catboost",
                "torch", "torchvision", "timm", "transformers", "peft", "bitsandbytes",
                "tabpfn", "tabicl", "shap", "optuna", "markitdown", "kaggle", "nbformat", "papermill"]


def run(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def find_envs():
    found = []
    for p in HINT_ENVS:
        exe = Path(p) / ("python.exe" if platform.system() == "Windows" else "bin/python")
        found.append({"path": p, "exists": Path(p).exists(), "python": str(exe) if exe.exists() else None})
    conda = shutil.which("conda")
    listed = []
    if conda:
        try:
            listed = json.loads(run([conda, "env", "list", "--json"]) or "{}").get("envs", [])
        except json.JSONDecodeError:
            pass
    return found, listed


def find_kaggle():
    locs = []
    if os.environ.get("KAGGLE_CONFIG_DIR"):
        locs.append(Path(os.environ["KAGGLE_CONFIG_DIR"]) / "kaggle.json")
    home = Path.home()
    locs += [home / ".kaggle" / "kaggle.json", home / ".kaggle" / "access_token",
             home / ".config" / "kaggle" / "kaggle.json"]
    files = [str(p) for p in locs if p.exists()]
    env = {k: bool(os.environ.get(k)) for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN")}
    return {"files": files, "env_vars_set": env, "cli": shutil.which("kaggle")}


def gpu_info():
    info = {"nvidia_smi": run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"])}
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device"] = torch.cuda.get_device_name(0)
            info["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            info["bf16"] = torch.cuda.is_bf16_supported()
    except Exception as e:
        info["torch_error"] = repr(e)
    return info


def cache_info():
    cands = list(HINT_CACHES)
    for k in ("NGEDATA_CACHE", "HF_HOME", "HF_HUB_CACHE", "TORCH_HOME"):
        if os.environ.get(k):
            cands.append(os.environ[k])
    out = []
    for c in dict.fromkeys(cands):
        p = Path(c)
        entry = {"path": c, "exists": p.exists()}
        if p.exists():
            try:
                entry["top_level"] = sorted(x.name for x in p.iterdir())[:50]
                entry["free_gb"] = round(shutil.disk_usage(p).free / 1024**3, 1)
            except OSError:
                pass
        out.append(entry)
    return out


def packages():
    import importlib.util
    res = {}
    for name in KEY_PACKAGES:
        spec = importlib.util.find_spec(name)
        if spec is None:
            res[name] = None
            continue
        try:
            from importlib.metadata import version
            dist = {"sklearn": "scikit-learn"}.get(name, name)
            res[name] = version(dist)
        except Exception:
            res[name] = "terpasang"
    return res


def to_md(d):
    L = [f"# Environment (terdeteksi {date.today()})", "",
         "## Python aktif", f"- executable: `{d['python']}`", f"- versi: {d['python_version']}",
         f"- OS: {d['os']}", "", "## Conda env (petunjuk)"]
    for e in d["hint_envs"]:
        L.append(f"- `{e['path']}` ada={e['exists']} python=`{e['python']}`")
    if d["conda_envs"]:
        L.append("- conda env list: " + ", ".join(f"`{x}`" for x in d["conda_envs"]))
    k = d["kaggle"]
    L += ["", "## Kaggle credential", f"- file ditemukan: {k['files'] or 'tidak ada'}",
          f"- env var: {k['env_vars_set']}", f"- kaggle CLI: `{k['cli']}`", "", "## GPU"]
    L += [f"- {kk}: {vv}" for kk, vv in d["gpu"].items()]
    L += ["", "## Cache model"]
    for c in d["cache"]:
        L.append(f"- `{c['path']}` ada={c['exists']} free={c.get('free_gb')} GB")
        if c.get("top_level"):
            L.append(f"  - isi: {', '.join(c['top_level'])}")
    L += ["", "## Paket kunci", "| paket | versi |", "|---|---|"]
    L += [f"| {p} | {v or 'TIDAK ADA'} |" for p, v in d["packages"].items()]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default=None, help="tulis markdown ke path ini")
    args = ap.parse_args()
    hint_envs, conda_envs = find_envs()
    d = {"python": sys.executable, "python_version": sys.version.split()[0],
         "os": f"{platform.system()} {platform.release()}", "hint_envs": hint_envs,
         "conda_envs": conda_envs, "kaggle": find_kaggle(), "gpu": gpu_info(),
         "cache": cache_info(), "packages": packages()}
    print(json.dumps(d, indent=2, default=str))
    if args.write:
        Path(args.write).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write).write_text(to_md(d), encoding="utf-8")
        print(f"\nditulis ke {args.write}")


if __name__ == "__main__":
    main()
