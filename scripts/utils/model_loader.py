"""Resolve model dari cache lokal dulu, fallback ke Hugging Face Hub.

Contoh:
    from scripts.utils.model_loader import resolve_model
    src, kw = resolve_model("microsoft/deberta-v3-base")
    tok = AutoTokenizer.from_pretrained(src, **kw)
    model = AutoModel.from_pretrained(src, **kw)

Urutan pencarian di CACHE_ROOT:
    1. <root>/<org>/<name>          (folder hasil save_pretrained / snapshot_download local_dir)
    2. <root>/<org>--<name>
    3. <root>/<name>
    4. <root>/hub/models--<org>--<name> atau <root>/models--<org>--<name> (layout cache HF)
Jika tidak ada, kembalikan model_id sehingga library mengunduh dari HF Hub.
"""
import os
from pathlib import Path

DEFAULT_CACHE_ROOT = os.environ.get("NGEDATA_CACHE", str(Path(r"D:\cache_moved") if Path(r"D:\cache_moved").exists() else Path.home() / ".cache" / "dsc-gelar-rasa"))


def _has_weights(p: Path) -> bool:
    if not p.is_dir():
        return False
    markers = ("config.json", "model.safetensors", "pytorch_model.bin",
               "model.safetensors.index.json", "open_clip_config.json")
    return any((p / m).exists() for m in markers) or any(p.glob("*.safetensors"))


def resolve_model(model_id: str, cache_root: str = DEFAULT_CACHE_ROOT, verbose: bool = True):
    """Kembalikan (source, kwargs) untuk from_pretrained."""
    root = Path(cache_root) if cache_root else None
    if root is not None and root.exists():
        org, _, name = model_id.rpartition("/")
        candidates = [root / model_id, root / model_id.replace("/", "--"), root / (name or model_id)]
        for c in candidates:
            if _has_weights(c):
                if verbose:
                    print(f"[model_loader] cache lokal: {c}")
                return str(c), {"local_files_only": True}
        hf_name = "models--" + model_id.replace("/", "--")
        for hub_dir in (root / "hub", root):
            if (hub_dir / hf_name).exists():
                if verbose:
                    print(f"[model_loader] cache HF layout: {hub_dir}")
                return model_id, {"cache_dir": str(hub_dir)}
    kwargs = {}
    if root is not None and root.exists() and os.access(root, os.W_OK):
        kwargs["cache_dir"] = str(root / "hub")
    if verbose:
        print(f"[model_loader] tidak ada di cache, unduh dari HF Hub: {model_id}")
    return model_id, kwargs


def setup_hf_env(cache_root: str = DEFAULT_CACHE_ROOT) -> None:
    """Arahkan cache HF/timm/torch ke cache lokal bila ada (panggil sebelum import library)."""
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    if root.exists():
        os.environ.setdefault("HF_HOME", str(root))
        os.environ.setdefault("HF_HUB_CACHE", str(root / "hub"))
        os.environ.setdefault("TORCH_HOME", str(root / "torch"))


if __name__ == "__main__":
    import sys
    mid = sys.argv[1] if len(sys.argv) > 1 else "microsoft/deberta-v3-base"
    print(resolve_model(mid))
