"""Seed strict dan deterministik untuk semua eksperimen ngedata.

Pakai di baris paling awal script, sebelum import torch/tensorflow bila memungkinkan:

    from scripts.utils.seed_utils import seed_everything
    SEED = seed_everything(42)

Untuk notebook, salin isi fungsi ke cell (notebook tidak boleh import script lokal).
"""
import os
import random

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = True, warn_only: bool = False) -> int:
    """Kunci semua sumber random. Kembalikan seed supaya bisa dicatat."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Harus diset sebelum CUDA context dibuat agar cuBLAS deterministik.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        if deterministic:
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            tf.config.experimental.enable_op_determinism()
    except (ImportError, AttributeError):
        pass

    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass

    return seed


def seed_worker(worker_id: int) -> None:
    """worker_init_fn untuk DataLoader agar tiap worker deterministik."""
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = DEFAULT_SEED):
    """Generator untuk DataLoader(generator=...)."""
    import torch
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def gbdt_params(lib: str, seed: int = DEFAULT_SEED, n_threads: int = 8) -> dict:
    """Parameter deterministik untuk library GBDT. Gabungkan dengan parameter model."""
    lib = lib.lower()
    if lib in ("lgbm", "lightgbm"):
        return {
            "seed": seed,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
            "data_random_seed": seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": n_threads,
            "verbosity": -1,
        }
    if lib in ("xgb", "xgboost"):
        return {"random_state": seed, "n_jobs": n_threads}
    if lib in ("cat", "catboost"):
        return {"random_seed": seed, "thread_count": n_threads, "allow_writing_files": False}
    if lib in ("sklearn", "rf", "et"):
        return {"random_state": seed, "n_jobs": n_threads}
    raise ValueError(f"library tidak dikenal: {lib}")


def prediction_hash(pred) -> str:
    """Hash prediksi untuk membandingkan script vs notebook."""
    import hashlib
    import numpy as np
    arr = np.ascontiguousarray(np.round(np.asarray(pred, dtype=np.float64), 6))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


if __name__ == "__main__":
    s = seed_everything()
    print(f"seed_everything OK, seed={s}")
