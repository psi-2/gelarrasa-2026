"""Regenerasi bagian AUTO di katalog.md. Bagian MANUAL tidak disentuh.

    python3 scripts/utils/update_katalog.py [--root .]

Deskripsi file diambil dari: judul markdown (# ...), docstring/komentar pertama file .py,
atau markdown cell pertama notebook. Folder data/ hanya diringkas (jumlah file, ukuran).
"""
import argparse
import ast
import json
from pathlib import Path

SECTIONS = ["docs", "eda", "scripts", "notebook", "notebooks", "results", "submission", "temp"]
SKIP = {"__pycache__", ".ipynb_checkpoints", ".git"}
AUTO_START, AUTO_END = "<!-- AUTO:START -->", "<!-- AUTO:END -->"


def describe(p: Path) -> str:
    try:
        if p.suffix == ".md":
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.strip():
                    return line.lstrip("# ").strip()[:120]
        if p.suffix == ".py":
            src = p.read_text(encoding="utf-8", errors="ignore")
            doc = ast.get_docstring(ast.parse(src))
            if doc:
                return doc.strip().splitlines()[0][:120]
            for line in src.splitlines():
                if line.startswith("#"):
                    return line.lstrip("# ").strip()[:120]
        if p.suffix == ".ipynb":
            nb = json.loads(p.read_text(encoding="utf-8"))
            for c in nb.get("cells", []):
                if c.get("cell_type") == "markdown":
                    baris = [l for l in "".join(c.get("source", [])).splitlines() if l.strip() and l.strip() != "---"]
                    if baris:
                        return baris[0].lstrip("# ").strip()[:120]
    except Exception:
        pass
    return ""


def size(p: Path) -> str:
    b = p.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.0f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def list_section(root: Path, name: str, max_depth: int):
    base = root / name
    if not base.exists():
        return []
    out = []
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(root)
        if any(part in SKIP for part in rel.parts) or len(rel.parts) - 1 > max_depth:
            continue
        if p.is_file():
            out.append(f"| `{rel.as_posix()}` | {size(p)} | {describe(p)} |")
    return out


def data_summary(root: Path):
    base = root / "data"
    if not base.exists():
        return ["(folder data belum ada)"]
    lines = []
    for d in sorted(x for x in base.iterdir()):
        if d.is_dir():
            files = [f for f in d.rglob("*") if f.is_file()]
            total = sum(f.stat().st_size for f in files) / 1024**2
            exts = sorted({f.suffix for f in files})
            lines.append(f"- `data/{d.name}/`: {len(files)} file, {total:.1f} MB, ekstensi {exts}")
        else:
            lines.append(f"- `data/{d.name}` ({size(d)})")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    kat = root / "katalog.md"
    text = kat.read_text(encoding="utf-8") if kat.exists() else f"# Katalog proyek\n\n{AUTO_START}\n{AUTO_END}\n"

    body = ["## Indeks file", "", "### data", *data_summary(root), ""]
    for s in SECTIONS:
        depth = 1 if s == "temp" else 3
        rows = list_section(root, s, depth)
        title = f"### {s}" + (" (BELUM TERBUKTI, jangan dianggap final)" if s == "temp" else "")
        body += [title, ""]
        body += (["| file | ukuran | deskripsi |", "|---|---|---|", *rows] if rows else ["(kosong)"])
        body.append("")
    for top in ("log_exp.md",):
        if (root / top).exists():
            body.append(f"- `{top}`: log semua eksperimen")
    auto = AUTO_START + "\n" + "\n".join(body) + "\n" + AUTO_END

    if AUTO_START in text and AUTO_END in text:
        pre = text.split(AUTO_START)[0]
        post = text.split(AUTO_END, 1)[1]
        text = pre + auto + post
    else:
        text = text.rstrip() + "\n\n" + auto + "\n"
    kat.write_text(text, encoding="utf-8")
    print(f"katalog diperbarui: {kat}")


if __name__ == "__main__":
    main()
