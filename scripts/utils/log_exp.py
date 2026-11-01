"""Tambah / update baris log_exp.md dengan format baku.

Tambah:
    python3 scripts/utils/log_exp.py add --fase prepro --desc "rasio domain" \
        --model "LGBM lr=0.03" --fitur "base + 12 rasio" --validasi "SGKF5" \
        --cv "0.81234 ± 0.00410" --path temp/exp_007_ratio/
    (mencetak ID baru, misal EXP-007)
Update:
    python3 scripts/utils/log_exp.py update EXP-007 --status promoted --path scripts/fe_ratio.py
    python3 scripts/utils/log_exp.py update EXP-007 --status best
Lihat nomor berikutnya:
    python3 scripts/utils/log_exp.py next
"""
import argparse
import re
from datetime import date
from pathlib import Path

COLS = ["ID", "Tanggal", "Fase", "Deskripsi", "Metode/Model", "Prepro/Fitur", "Validasi",
        "CV", "Public Score", "Temuan", "Status", "Path"]
FASE = {"setup", "research", "prepro", "modelling", "notebook"}
STATUS = {"temp", "promoted", "ditolak", "best"}
HEADER = "| " + " | ".join(COLS) + " |"
SEP = "|" + "---|" * len(COLS)


def clean(x):
    x = (x or "-").replace("|", "/").replace("\n", " ").replace("\u2014", "-").strip()
    return x or "-"


def load(path):
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines()
    try:
        h = lines.index(HEADER)
    except ValueError:
        raise SystemExit(f"Header tabel tidak sesuai format baku di {path}. Perbaiki dulu, jangan ubah kolom.")
    pre = lines[:h]
    rows = [l for l in lines[h + 2:] if l.startswith("| EXP-")]
    return pre, rows


def parse(row):
    return [c.strip() for c in row.strip().strip("|").split("|")]


def fmt(cells):
    return "| " + " | ".join(cells) + " |"


def save(path, pre, rows):
    path.write_text("\n".join(pre + [HEADER, SEP] + rows) + "\n", encoding="utf-8")


def next_id(rows):
    nums = [int(m.group(1)) for r in rows if (m := re.match(r"\| EXP-(\d+)", r))]
    return f"EXP-{(max(nums) + 1 if nums else 1):03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="log_exp.md")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    for k in ("fase", "desc", "model", "fitur", "validasi", "cv", "public", "temuan", "status", "path"):
        a.add_argument(f"--{k}", default=None)
    u = sub.add_parser("update")
    u.add_argument("id")
    for k in ("cv", "public", "temuan", "status", "path", "desc"):
        u.add_argument(f"--{k}", default=None)
    sub.add_parser("next")
    args = ap.parse_args()

    path = Path(args.file)
    pre, rows = load(path)

    if args.cmd == "next":
        print(next_id(rows))
        return

    if args.cmd == "add":
        if args.fase not in FASE:
            raise SystemExit(f"--fase harus salah satu {sorted(FASE)}")
        status = args.status or "temp"
        if status not in STATUS:
            raise SystemExit(f"--status harus salah satu {sorted(STATUS)}")
        eid = next_id(rows)
        cells = [eid, date.today().isoformat(), args.fase, clean(args.desc), clean(args.model),
                 clean(args.fitur), clean(args.validasi), clean(args.cv), clean(args.public),
                 clean(args.temuan), status, clean(args.path)]
        rows.append(fmt(cells))
        if status == "best":
            rows = demote_best(rows, keep=eid)
        save(path, pre, rows)
        print(eid)
        return

    idx = {parse(r)[0]: i for i, r in enumerate(rows)}
    if args.id not in idx:
        raise SystemExit(f"{args.id} tidak ada di log")
    cells = parse(rows[idx[args.id]])
    mapping = {"cv": 7, "public": 8, "temuan": 9, "status": 10, "path": 11, "desc": 3}
    for k, col in mapping.items():
        v = getattr(args, k)
        if v is not None:
            if k == "status" and v not in STATUS:
                raise SystemExit(f"--status harus salah satu {sorted(STATUS)}")
            cells[col] = clean(v)
    rows[idx[args.id]] = fmt(cells)
    if args.status == "best":
        rows = demote_best(rows, keep=args.id)
    save(path, pre, rows)
    print(f"{args.id} diperbarui")


def demote_best(rows, keep):
    out = []
    for r in rows:
        c = parse(r)
        if c[0] != keep and c[10] == "best":
            c[10] = "promoted"
        out.append(fmt(c))
    return out


if __name__ == "__main__":
    main()
