"""DINOv3 ViT-L pada label cluster (train saja), validasi hold-out.

    python temp/exp_006_cluster_dinov3/train_dino.py --tag holdout

Di label_folds.csv, fold 0 adalah hold-out dan fold 1 adalah data latih.
Skor hold-out mengukur kesepakatan dengan label cluster, bukan akurasi asli.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.utils.model_loader import resolve_model, setup_hf_env  # noqa: E402
from scripts.utils.seed_utils import make_generator, prediction_hash, seed_everything, seed_worker  # noqa: E402

setup_hf_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from timm.optim import create_optimizer_v2  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import transforms as T  # noqa: E402

DATA = ROOT / "data"
HERE = Path(__file__).resolve().parent
LABEL = HERE / "output" / "label_folds.csv"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
MODEL_ID = "timm/vit_large_patch16_dinov3.lvd1689m"
SEED = 42
UKURAN = 384
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

CFG = {"lr": 1e-4, "head_mult": 10.0, "layer_decay": 0.75, "wd": 0.05, "warmup_epoch": 1,
       "label_smoothing": 0.1, "drop_path": 0.1, "bs": 8, "accum": 2}
VRAM_MAKS_GIB = 13.5  # batas keras torch, proses lain memakai sekitar 1,2 GiB. Lewat batas = OOM, bukan bocor ke RAM


def tf_train():
    return T.Compose([T.RandomResizedCrop(UKURAN, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                      T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2, 0.02),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


def tf_eval():
    return T.Compose([T.Resize((UKURAN, UKURAN), interpolation=T.InterpolationMode.BICUBIC),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


class Citra(Dataset):
    def __init__(self, files, labels=None, tf=None):
        self.files, self.labels, self.tf = files, labels, tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.files[i]) as im:
            x = self.tf(im.convert("RGB"))
        return (x, -1) if self.labels is None else (x, int(self.labels[i]))


def buat_model(num_classes, drop_path=0.0):
    src, _ = resolve_model(MODEL_ID, verbose=False)
    overlay = {"file": str(Path(src) / "model.safetensors")} if Path(src).is_dir() else None
    return timm.create_model("vit_large_patch16_dinov3.lvd1689m", pretrained=True, num_classes=num_classes,
                             drop_path_rate=drop_path, pretrained_cfg_overlay=overlay, img_size=UKURAN)


def loader(ds, bs, shuffle=False, seed=SEED):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=4, persistent_workers=True,
                      pin_memory=True, worker_init_fn=seed_worker, generator=make_generator(seed),
                      drop_last=shuffle)


@torch.no_grad()
def prediksi(model, files, bs=16):
    """Probabilitas dengan TTA flip horizontal."""
    model.eval()
    hasil = []
    for x, _ in loader(Citra(files, tf=tf_eval()), bs):
        x = x.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.softmax(model(x).float(), 1) + torch.softmax(model(x.flip(3)).float(), 1)
        hasil.append((p / 2).cpu())
    return torch.cat(hasil).numpy()


@torch.no_grad()
def ekstrak(model, files, bs=16):
    model.eval()
    hasil = []
    for x, _ in loader(Citra(files, tf=tf_eval()), bs):
        x = x.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hasil.append(model(x).float().cpu())
    return torch.cat(hasil).numpy()


def probe(df, files, test_files, out):
    """Linear probe sebagai sanity check sebelum fine-tune."""
    model = buat_model(0).cuda()
    X, Xt = ekstrak(model, files), ekstrak(model, test_files)
    del model
    torch.cuda.empty_cache()
    y = df["label"].map(NAMES.index).values
    oof, skor, pt = np.zeros((len(df), 8)), [], np.zeros((len(test_files), 8))
    for f in sorted(df["fold"].unique()):
        tr, va = df["fold"].values != f, df["fold"].values == f
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])
        pt += clf.predict_proba(Xt) / df["fold"].nunique()
        skor.append(float((oof[va].argmax(1) == y[va]).mean()))
    hasil = {"cv_rata": round(float(np.mean(skor)), 4), "cv_std": round(float(np.std(skor)), 4),
             "per_fold": [round(s, 4) for s in skor]}
    print("linear probe:", hasil)
    np.save(out / "probe_test.npy", pt)
    (out / "probe.json").write_text(json.dumps(hasil, indent=2), encoding="utf-8")


def bobot_kelas(y):
    """Invers akar frekuensi. Ambang keyakinan membuang kelas secara tidak merata (paper 60%, textile 94%)."""
    n = np.bincount(y, minlength=8).astype(float)
    w = 1 / np.sqrt(n)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def latih_fold(f, df, files, test_files, sisa_files, args, out):
    seed_everything(SEED)
    tr, va = df["fold"].values != f, df["fold"].values == f
    y = df["label"].map(NAMES.index).values
    f_tr = [files[i] for i in np.where(tr)[0]]
    f_va = [files[i] for i in np.where(va)[0]]

    model = buat_model(8, CFG["drop_path"]).cuda()
    opt = create_optimizer_v2(model, opt="adamw", lr=CFG["lr"], weight_decay=CFG["wd"], layer_decay=CFG["layer_decay"])
    puncak = max(g.get("lr_scale", 1.0) for g in opt.param_groups)
    for g in opt.param_groups:
        g["lr_scale"] = g.get("lr_scale", 1.0) * (CFG["head_mult"] if g.get("lr_scale", 1.0) >= puncak else 1.0)

    dl = loader(Citra(f_tr, y[tr], tf_train()), CFG["bs"], shuffle=True, seed=SEED + f)
    per_epoch = len(dl) // CFG["accum"]
    total, warm = per_epoch * args.epochs, per_epoch * CFG["warmup_epoch"]
    w = bobot_kelas(y[tr]).cuda()
    riwayat, t0, step = [], time.time(), 0
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        rugi = []
        for i, (x, t) in enumerate(dl):
            x, t = x.cuda(non_blocking=True), t.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(x).float(), t, weight=w,
                                                         label_smoothing=CFG["label_smoothing"])
            (loss / CFG["accum"]).backward()
            rugi.append(loss.item())
            if (i + 1) % CFG["accum"] == 0:
                faktor = min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * step / total)))
                for g in opt.param_groups:
                    g["lr"] = CFG["lr"] * g["lr_scale"] * faktor
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
        pv = prediksi(model, f_va)
        acc = float((pv.argmax(1) == y[va]).mean())
        riwayat.append({"epoch": ep + 1, "loss": round(float(np.mean(rugi)), 4), "val_acc": round(acc, 4)})
        print(f"fold {f} epoch {ep + 1}/{args.epochs} loss {riwayat[-1]['loss']} val {acc:.4f} "
              f"t {time.time() - t0:.0f}s vram {torch.cuda.max_memory_reserved() / 2**30:.1f}GiB", flush=True)

    pt = prediksi(model, test_files)
    ps = prediksi(model, sisa_files) if sisa_files else np.zeros((0, 8))
    per_kelas = {NAMES[c]: round(float((pv[y[va] == c].argmax(1) == c).mean()), 4)
                 for c in range(8) if (y[va] == c).any()}
    np.save(out / f"oof_fold{f}.npy", pv)
    np.save(out / f"test_fold{f}.npy", pt)
    np.save(out / f"sisa_fold{f}.npy", ps)
    hasil = {"fold": f, "val_acc": riwayat[-1]["val_acc"], "per_kelas": per_kelas, "riwayat": riwayat,
             "detik": round(time.time() - t0, 1), "vram_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
             "hash_test": prediction_hash(pt)}
    (out / f"fold{f}.json").write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"fold {f} selesai val {hasil['val_acc']} hash {hasil['hash_test']}", flush=True)
    del model, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="*", default=[0])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="fine")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    seed_everything(SEED)
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(VRAM_MAKS_GIB / total_gib, 0)

    out = HERE / "output" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(LABEL, dtype={"image_id": str})
    peta = {p.stem: p for p in (DATA / "train").iterdir()}
    test_files = sorted((DATA / "test").iterdir())
    sisa = sorted(k for k in peta if k not in set(df["image_id"]))
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
        test_files, sisa = test_files[:32], sisa[:32]
    files = [peta[i] for i in df["image_id"]]
    sisa_files = [peta[i] for i in sisa]
    print(f"label {len(df)} | test {len(test_files)} | sisa {len(sisa_files)} | cfg {CFG}", flush=True)

    if args.probe:
        probe(df, files, test_files, out)
        return

    log_vram = str(out / "vram.csv")
    mon = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "utils" / "vram_monitor.py"), "--pid", str(os.getpid()),
                            "--interval", "15", "--log", log_vram], stdout=subprocess.DEVNULL)
    try:
        for f in args.folds:
            latih_fold(f, df, files, test_files, sisa_files, args, out)
    finally:
        mon.terminate()


if __name__ == "__main__":
    main()
