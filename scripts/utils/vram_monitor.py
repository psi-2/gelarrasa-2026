"""Pantau VRAM dedicated dan shared GPU memory (deteksi bocor ke RAM di Windows).

Aturan: dedicated > 15 GB DAN shared > 0.5 GB  => BOCOR (anggap OOM, hentikan run).

Mode terminal terpisah:
    python3 scripts/utils/vram_monitor.py --pid 12345 --interval 5 --log temp/exp_001/vram.csv
Mode di dalam training loop:
    from scripts.utils.vram_monitor import check_vram
    if step % 50 == 0:
        check_vram(raise_on_leak=True)
"""
import argparse
import csv
import os
import platform
import subprocess
import time
from datetime import datetime

DED_LIMIT_GB = 15.0
SHARED_LIMIT_GB = 0.5


def dedicated_used_gb(gpu_index: int = 0):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        used, total = [float(x) for x in out.split(",")]
        return used / 1024, total / 1024
    except Exception:
        return None, None


def _ps_counter(path: str):
    cmd = ["powershell", "-NoProfile", "-Command",
           f"(Get-Counter '{path}' -ErrorAction SilentlyContinue).CounterSamples | "
           "ForEach-Object { $_.InstanceName + '|' + $_.CookedValue }"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return {}
    res = {}
    for line in out.splitlines():
        if "|" in line:
            k, v = line.rsplit("|", 1)
            try:
                res[k.strip()] = float(v)
            except ValueError:
                pass
    return res


def shared_used_gb(pid=None):
    """Shared GPU memory (GB). Windows saja; None bila tidak tersedia.

    Dengan pid: ambil adapter (luid) tempat proses memakai dedicated terbanyak (GPU diskrit),
    lalu laporkan shared usage proses itu di adapter tersebut.
    Tanpa pid: ambil shared usage adapter dengan dedicated usage terbesar.
    """
    if platform.system() != "Windows":
        return None
    if pid is not None:
        ded = _ps_counter(f"\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage")
        sha = _ps_counter(f"\\GPU Process Memory(pid_{pid}_*)\\Shared Usage")
    else:
        ded = _ps_counter("\\GPU Adapter Memory(*)\\Dedicated Usage")
        sha = _ps_counter("\\GPU Adapter Memory(*)\\Shared Usage")
    if not ded:
        return None
    best = max(ded, key=ded.get)
    luid = best.split("luid_")[-1] if "luid_" in best else best
    shared = sum(v for k, v in sha.items() if luid in k)
    return shared / 1024**3


def torch_reserved_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 1024**3
    except ImportError:
        pass
    return None


def check_vram(pid=None, ded_limit=DED_LIMIT_GB, shared_limit=SHARED_LIMIT_GB,
               raise_on_leak=False, verbose=True):
    """Kembalikan dict status. raise_on_leak=True melempar MemoryError bila bocor."""
    pid = pid if pid is not None else os.getpid()
    ded, total = dedicated_used_gb()
    shared = shared_used_gb(pid)
    reserved = torch_reserved_gb()
    leak = ded is not None and ded > ded_limit and shared is not None and shared > shared_limit
    status = {"dedicated_gb": ded, "total_gb": total, "shared_gb": shared,
              "torch_reserved_gb": reserved, "leak": leak}
    if verbose:
        fmt = lambda x: "n/a" if x is None else f"{x:.2f}"
        print(f"[vram] dedicated={fmt(ded)}/{fmt(total)} GB shared={fmt(shared)} GB "
              f"torch_reserved={fmt(reserved)} GB leak={leak}")
    if leak and raise_on_leak:
        raise MemoryError(f"VRAM bocor ke RAM: dedicated {ded:.2f} GB, shared {shared:.2f} GB. "
                          "Turunkan batch/resolusi/seq len, aktifkan AMP/checkpointing/LoRA.")
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, default=None, help="PID proses training")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--log", type=str, default=None, help="path CSV log")
    ap.add_argument("--ded-limit", type=float, default=DED_LIMIT_GB)
    ap.add_argument("--shared-limit", type=float, default=SHARED_LIMIT_GB)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    writer = None
    if args.log:
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        f = open(args.log, "a", newline="")
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["time", "dedicated_gb", "shared_gb", "leak"])
    while True:
        ded, _ = dedicated_used_gb()
        shared = shared_used_gb(args.pid)
        leak = ded is not None and ded > args.ded_limit and shared is not None and shared > args.shared_limit
        now = datetime.now().strftime("%H:%M:%S")
        print(f"{now} dedicated={ded} GB shared={shared} GB {'BOCOR: HENTIKAN RUN' if leak else 'ok'}", flush=True)
        if writer:
            writer.writerow([now, ded, shared, leak])
            f.flush()
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
