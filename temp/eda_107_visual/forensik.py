"""Sub-sumber di dalam F_416 dan F_640: periode grid, anisotropi, padding, dan GMM forensik.

    python temp/eda_107_visual/forensik.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
pd.set_option("display.width", 240, "display.max_columns", 60, "display.max_rows", 300)
v = pd.read_csv(OUT / "visual.csv", dtype={"image_id": str})
m = pd.read_csv(ROOT / "temp" / "eda_102_sumber" / "output" / "meta_assign.csv", dtype={"image_id": str})
d = v.merge(m, on=["split", "image_id"])
d.to_csv(OUT / "gabung.csv", index=False)

print("per_x kuat median per fam:")
print(d.groupby("fam")[["per_x", "per_x_kuat", "per_y", "per_y_kuat", "aniso", "hf_aniso", "hf_025", "noise", "blk8", "bpp"]].median().round(3).to_string())

for fam in ("F_416", "F_640"):
    s = d[d.fam == fam].copy()
    k = s[(s.per_x_kuat > 8) & (s.per_y_kuat > 8)]
    print(f"\n{fam} n={len(s)} kuat kedua sumbu {len(k)}")
    print("periode x terbulatkan 0.25 (top 12)")
    print(((k.per_x * 4).round() / 4).value_counts().head(12).to_string())
    print("periode y terbulatkan 0.25 (top 12)")
    print(((k.per_y * 4).round() / 4).value_counts().head(12).to_string())
    # lebar asli tersirat bila grid 8 lama: W_asli = 8 * W / periode
    W = int(s.w.iloc[0])
    ox = (8 * W / k.per_x).round(-1)
    oy = (8 * W / k.per_y).round(-1)
    print("ukuran asli tersirat (top 15)")
    print((ox.astype(int).astype(str) + "x" + oy.astype(int).astype(str)).value_counts().head(15).to_string())
    print("padding hitam/putih:", {c: round(float((s[c] > .5).mean()), 3) for c in ("hitam_baris", "hitam_kolom", "putih_baris", "putih_kolom")})
    print("anisotropi kuantil", s.aniso.quantile([.02, .1, .25, .5, .75, .9, .98]).round(3).tolist())

    F = ["aniso", "hf_aniso", "hf_025", "hf_035", "noise", "blk8", "lap_var", "bpp", "tepi_std", "hitam_baris", "hitam_kolom", "putih_baris", "putih_kolom", "lok_halus"]
    X = RobustScaler().fit_transform(s[F].clip(lower=s[F].quantile(.005), upper=s[F].quantile(.995), axis=1))
    best = None
    for k_ in (2, 3, 4, 5, 6, 8, 10):
        g = GaussianMixture(k_, covariance_type="diag", random_state=42, n_init=3).fit(X)
        b = g.bic(X)
        lab = g.predict(X)
        print(k_, "BIC", round(b), "AMI ke cluster DINO", round(adjusted_mutual_info_score(s.cluster, lab), 3),
              "AMI ke nama", round(adjusted_mutual_info_score(s.nama, lab), 3))
        if best is None or b < best[0]:
            best = (b, k_, lab)
    s["gm"] = best[2]
    print("GMM terbaik k =", best[1])
    print(pd.crosstab(s.gm, s.nama).reindex(columns=NAMES).fillna(0).astype(int).assign(test=s[s.split == "test"].groupby("gm").size(), train=s[s.split == "train"].groupby("gm").size()).to_string())
    print(s.groupby("gm")[F].median().round(3).to_string())

print("\n== tabel cluster: nama, lolos, keluarga, test ==")
ct = pd.read_csv(ROOT / "temp" / "eda_102_sumber" / "output" / "cluster_nama.csv").set_index("cluster")
p = pd.concat([pd.read_csv(ROOT / "temp" / "exp_005_imagenet_map" / "output" / f"prob_{s}.csv", dtype={"image_id": str}).assign(split=s) for s in ("train", "test")])
x = d.merge(p[["split", "image_id", "top1_imagenet"]], on=["split", "image_id"])
rows = []
for c, g in x.groupby("cluster"):
    f = g.fam.value_counts()
    rows.append({"c": c, "nama": ct.loc[c, "nama"], "porsi": ct.loc[c, "porsi"], "kedua": ct.loc[c, "kedua"], "p2": ct.loc[c, "porsi2"], "lolos": bool(g.lolos.iloc[0]),
                 "tr": (g.split == "train").sum(), "te": (g.split == "test").sum(),
                 "f416": f.get("F_416", 0), "f640": f.get("F_640", 0), "web": f.get("F_lain_web", 0), "q75": f.get("F_q75_4x3", 0) + f.get("F_q75_lain", 0), "q2527": f.get("F_q2527_400x", 0), "q48": f.get("F_q4885_besar", 0), "q74": f.get("F_q74fb", 0),
                 "top": ", ".join(g.top1_imagenet.value_counts().head(4).index)})
print(pd.DataFrame(rows).sort_values(["nama", "te"], ascending=[True, False]).to_string(index=False))
