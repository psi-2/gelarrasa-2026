"""Validasi notebook ngedata sebelum dipromosikan ke notebooks/.

    python3 scripts/utils/check_notebook.py notebooks/final_xxx.ipynb [--final]

Cek:
1. Tidak ada em dash (U+2014) di cell manapun.
2. Tidak import script lokal proyek (from scripts..., import scripts...) atau sys.path ke folder proyek.
3. Tidak load artifact/checkpoint lokal (.pkl .joblib .pt .pth .ckpt .npy .parquet hasil) dari results/temp/scripts.
4. Setiap cell kode didahului cell markdown yang tidak kosong.
5. Ada seed_everything, ada fallback HF Hub (resolve_model atau pola serupa), ada SEED eksplisit.
6. Mode --final: ada bagian EDA dan XAI (kata kunci: EDA, SHAP/XAI/Grad-CAM/attribution).
7. Style (ngedata-notebook/references/notebook_style.md): cell judul HTML, Daftar Isi, header section
   dengan anchor, class Settings + CFG di cell bertag settings, Daftar Pustaka, SMOKE_TEST. Lewati dengan --no-style.
Keluar dengan kode 1 bila ada pelanggaran.
"""
import argparse
import json
import re
import sys
from pathlib import Path

EM = "\u2014"
BAD_IMPORT = re.compile(r"^\s*(from\s+scripts[\.\s]|import\s+scripts\b|sys\.path\.(append|insert))", re.M)
LOAD_CALL = re.compile(r"(joblib\.load|pickle\.load|torch\.load|np\.load|load_state_dict|read_pickle)\s*\(")
READ_CALL = re.compile(r"(\.load|read_csv|read_parquet|read_pickle|load_state_dict|open)\s*\(")
LOCAL_ARTIFACT = re.compile(r"(results|temp|scripts)[/\\][^'\"\s]*\.(pkl|joblib|pt|pth|ckpt|npy|npz|bin|safetensors|csv|parquet)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nb")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--no-style", action="store_true")
    args = ap.parse_args()
    nb = json.loads(Path(args.nb).read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    errs, warns = [], []
    all_code = ""
    prev_md = False
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if EM in src:
            errs.append(f"cell {i}: mengandung em dash")
        if c["cell_type"] == "code":
            all_code += src + "\n"
            if src.strip() and not prev_md:
                errs.append(f"cell {i}: cell kode tanpa markdown penjelas sebelumnya")
            if BAD_IMPORT.search(src):
                errs.append(f"cell {i}: import script lokal / ubah sys.path")
            if any(LOCAL_ARTIFACT.search(l) and READ_CALL.search(l) for l in src.splitlines()):
                errs.append(f"cell {i}: memakai file dari results/temp/scripts (artifact/hasil eksperimen)")
            elif LOAD_CALL.search(src):
                warns.append(f"cell {i}: kemungkinan load artifact lokal, pastikan bukan checkpoint/hasil eksperimen")
            prev_md = False
        elif c["cell_type"] == "markdown":
            prev_md = bool(src.strip())
    if "seed_everything" not in all_code:
        errs.append("tidak ada seed_everything")
    if not re.search(r"SEED\s*=\s*\d+", all_code):
        errs.append("tidak ada SEED eksplisit")
    uses_hf = re.search(r"from_pretrained|timm\.create_model|hf_hub_download|AutoModel", all_code)
    if uses_hf and "resolve_model" not in all_code and "local_files_only" not in all_code:
        errs.append("memakai model pretrained tapi tidak ada fallback cache -> HF Hub")
    if args.final:
        md_all = " ".join("".join(c.get("source", [])) for c in cells).lower()
        if "eda" not in md_all:
            errs.append("versi final wajib punya bagian EDA")
        if not re.search(r"shap|xai|grad-?cam|attribution|explainab|permutation importance", md_all + all_code.lower()):
            errs.append("versi final wajib punya bagian XAI")
    if not args.no_style:
        md_cells = ["".join(c.get("source", [])) for c in cells if c["cell_type"] == "markdown"]
        if not md_cells or "<h1" not in md_cells[0] or "#8E7B6B" not in md_cells[0]:
            errs.append("style: cell pertama harus judul HTML bergaris #8E7B6B")
        if not any("# Daftar Isi" in m for m in md_cells):
            errs.append("style: tidak ada Daftar Isi")
        anchors = re.findall(r'<a name="(\d+)"></a>', " ".join(md_cells))
        links = re.findall(r"\]\(#(\d+)\)", " ".join(md_cells))
        if not anchors:
            errs.append("style: header section tanpa anchor <a name=...>")
        elif sorted(set(links)) != sorted(set(anchors)):
            errs.append(f"style: anchor Daftar Isi {sorted(set(links))} tidak cocok dengan section {sorted(set(anchors))}")
        if not any("Daftar Pustaka" in m for m in md_cells):
            errs.append("style: tidak ada section Daftar Pustaka")
        settings = [c for c in cells if "settings" in c.get("metadata", {}).get("tags", [])]
        if not settings:
            errs.append("style: tidak ada cell bertag settings")
        elif not re.search(r"class Settings", "".join(settings[0]["source"])) or "CFG" not in "".join(settings[0]["source"]):
            errs.append("style: cell settings harus berisi class Settings dan CFG")
        if "SMOKE_TEST" not in all_code:
            errs.append("style: tidak ada SMOKE_TEST di Settings")
    for w in warns:
        print("PERINGATAN:", w)
    for e in errs:
        print("GAGAL:", e)
    print(f"{len(cells)} cell diperiksa, {len(errs)} pelanggaran, {len(warns)} peringatan")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
