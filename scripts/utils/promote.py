"""Promosikan hasil dari temp/ ke folder final, lalu update log_exp dan katalog.

    python3 scripts/utils/promote.py temp/eda_003_sensor_drift eda/sensor_drift --exp EXP-003
    python3 scripts/utils/promote.py temp/exp_007_ratio/fe_ratio.py scripts/fe_ratio.py --exp EXP-007 --best

Default: salin (temp tetap utuh sebagai jejak). Pakai --move untuk memindahkan.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ALLOWED = ("eda", "scripts", "notebooks", "results", "submission", "docs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--exp", default=None, help="ID log, contoh EXP-007")
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--move", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        raise SystemExit(f"sumber tidak ada: {src}")
    if dst.parts[0] not in ALLOWED:
        raise SystemExit(f"tujuan harus di salah satu folder {ALLOWED}")
    if dst.exists():
        raise SystemExit(f"tujuan sudah ada, pilih nama lain: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if args.move:
        shutil.move(str(src), str(dst))
    elif src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    print(f"{'dipindah' if args.move else 'disalin'}: {src} -> {dst}")

    here = Path(__file__).resolve().parent
    if args.exp:
        status = "best" if args.best else "promoted"
        subprocess.run([sys.executable, str(here / "log_exp.py"), "update", args.exp,
                        "--status", status, "--path", dst.as_posix()], check=False)
    subprocess.run([sys.executable, str(here / "update_katalog.py")], check=False)


if __name__ == "__main__":
    main()
