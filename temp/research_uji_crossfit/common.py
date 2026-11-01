"""Modul bersama uji metode: CPU saja, 4 thread, tanpa temuan.

Batasan yang dijaga di sini:
- data/temuan/ tidak pernah dibaca. Tidak ada path ke sana di folder ini.
- Citra tidak dibuka. Semua masukan adalah array dan tabel hasil eksperimen sebelumnya.
- Test hanya melewati fungsi predict. Semua fit, PCA, graf, dan offset kelas memakai train.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import csr_matrix, diags, identity  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.sparse.linalg import cg  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import torch  # noqa: E402

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
E6 = ROOT / "temp" / "exp_006_cluster_dinov3" / "output"
E9 = ROOT / "temp" / "exp_009_t1_dinov3" / "output"
E5 = ROOT / "temp" / "exp_005_imagenet_map" / "output"
F110 = ROOT / "temp" / "eda_110_fusi" / "output"
EOMT = ROOT / "temp" / "eda_106_eomt" / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
IDX = {n: i for i, n in enumerate(NAMES)}
N_KELAS = 8


def muat():
    """Muat semua masukan dan pastikan urutan baris sama. y = -1 untuk 294 citra tanpa label."""
    ids = np.load(E6 / "ids_train.npy", allow_pickle=True).astype(str)
    ids_te = np.load(E6 / "ids_test.npy", allow_pickle=True).astype(str)
    E_tr, E_te = np.load(E6 / "emb_train.npy").astype(np.float64), np.load(E6 / "emb_test.npy").astype(np.float64)
    assert E_tr.shape == (1920, 1024) and E_te.shape == (480, 1024)
    lab = pd.read_csv(E9 / "label_folds.csv", dtype={"image_id": str}).set_index("image_id")
    assert len(lab) == 1626
    y = np.full(1920, -1)
    pos = {k: i for i, k in enumerate(ids)}
    for k, c in lab.label.items():
        y[pos[k]] = IDX[c]
    assert (y >= 0).sum() == 1626

    V = np.load(F110 / "var_t1_in1k.npy").astype(np.float64)
    assert V.shape == (2400, 8)
    V_tr, V_te_diag = V[:1920], V[1920:]  # V_te_diag hanya diagnosis, tidak dipakai fit
    terpakai = V_tr.max(1) >= .75
    assert (terpakai == (y >= 0)).all(), "label tidak sama dengan suara >= 0,75"
    assert (V_tr.argmax(1)[terpakai] == y[terpakai]).all()

    meta = pd.read_csv(ROOT / "temp" / "eda_102_sumber" / "output" / "meta_assign.csv", dtype={"image_id": str})
    tr = meta[meta.split == "train"].sort_values("image_id").reset_index(drop=True)
    assert (tr.image_id.values == ids).all()
    fam = tr.fam.values  # hanya audit, tidak masuk fitur dan tidak masuk aturan

    # Group gabungan, rumus sama dengan temp/eda_110_fusi/label_final.py, pasangan train saja.
    S = E_tr @ E_tr.T
    np.fill_diagonal(S, 0)
    a, b = np.where(S >= .9)
    ph = pd.read_csv(ROOT / "temp" / "eda_104_duplikat" / "output" / "pasangan_hash.csv")
    ph = ph[(ph.sa == "train") & (ph.sb == "train") & (ph.ham <= 4)]
    ii, jj = np.r_[a, ph.a.values], np.r_[b, ph.b.values]
    _, grp = connected_components(csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(1920, 1920)), directed=False)
    # Cek: partisi group sama dengan kolom group di label EXP-009.
    g_lab = lab.group.reindex(ids[y >= 0]).values
    assert pd.crosstab(grp[y >= 0], g_lab).gt(0).sum(1).max() == 1, "group tidak cocok"

    P = np.load(E5 / "imagenet_train.npy").astype(np.float64)
    assert P.shape == (1920, 1000)
    # Probabilitas EXP-009 pada hold-out, sisa, dan test: pembanding tanpa temuan.
    sisa = np.load(E9 / "holdout" / "sisa_fold0.npy").astype(np.float64)
    te_ft = np.load(E9 / "holdout" / "test_fold0.npy").astype(np.float64)
    assert sisa.shape == (294, 8) and te_ft.shape == (480, 8)
    return dict(ids=ids, ids_te=ids_te, E=E_tr, E_te=E_te, y=y, V=V_tr, V_te_diag=V_te_diag, grp=grp, fam=fam,
                P=P, sisa=sisa, te_ft=te_ft, suara=V_tr.max(1), S=S)


def muat_cnn(ids, ids_te):
    """Embedding ConvNeXt-L DINOv3 hasil ekstrak_convnext.py, urutan dicek terhadap id DINOv3."""
    assert (np.load(OUT / "ids_cnn_train.npy").astype(str) == ids).all()
    assert (np.load(OUT / "ids_cnn_test.npy").astype(str) == ids_te).all()
    return np.load(OUT / "emb_cnn_train.npy").astype(np.float64), np.load(OUT / "emb_cnn_test.npy").astype(np.float64)


def pandangan_eomt():
    """Pandangan yang tidak ikut membentuk label t1_in1k: keluaran EoMT COCO dan ADE, train dan test.

    Hanya varian t1_in1k yang dipakai sebagai label, dan varian itu tidak membaca COCO atau ADE.
    Backbone EoMT tetap DINOv3, jadi pandangan ini independen parsial, bukan independen penuh.
    """
    def f(n):
        tr, te = np.load(EOMT / f"{n}_train.npy"), np.load(EOMT / f"{n}_test.npy")
        return tr.reshape(len(tr), -1), te.reshape(len(te), -1)
    a, b = f("coco"), f("ade")
    return np.log1p(np.c_[a[0], b[0]].astype(np.float64).clip(0)), np.log1p(np.c_[a[1], b[1]].astype(np.float64).clip(0))


def lipat(y_lab, grp_lab, n=5, seed=0):
    """StratifiedGroupKFold pada citra berlabel."""
    return list(StratifiedGroupKFold(n, shuffle=True, random_state=seed).split(np.zeros(len(y_lab)), y_lab, grp_lab))


def lengkap(p, kelas):
    """Rentangkan predict_proba sklearn ke 8 kolom bila ada kelas hilang."""
    q = np.zeros((len(p), N_KELAS))
    q[:, kelas] = p
    return q


def lr_fit_predict(Xtr, ytr, Xs, C=3.0):
    """Baseline: logistic pada embedding, sama dengan probe di label_final.py. Xs daftar array untuk diprediksi."""
    m = LogisticRegression(C=C, max_iter=3000).fit(Xtr, ytr)
    return [lengkap(m.predict_proba(x), m.classes_) for x in Xs]


def lr_lunak(Xtr, T, Xs, C=3.0, bobot=None, iter_maks=200):
    """Multinomial logistic dengan target lunak. Objektif sama dengan sklearn: C * jumlah CE + 0,5 * ||W||^2."""
    x = torch.as_tensor(Xtr, dtype=torch.float64)
    t = torch.as_tensor(T, dtype=torch.float64)
    bb = torch.ones(len(x), dtype=torch.float64) if bobot is None else torch.as_tensor(bobot, dtype=torch.float64)
    W = torch.zeros(x.shape[1], N_KELAS, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(N_KELAS, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=iter_maks, history_size=20, tolerance_grad=1e-5, tolerance_change=1e-9,
                            line_search_fn="strong_wolfe")

    def tutup():
        opt.zero_grad()
        lp = torch.log_softmax(x @ W + b, 1)
        loss = C * (-(t * lp).sum(1) * bb).sum() + .5 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(tutup)
    with torch.no_grad():
        return [torch.softmax(torch.as_tensor(q, dtype=torch.float64) @ W + b, 1).numpy() for q in Xs]


def satu_panas(y):
    return np.eye(N_KELAS)[y]


def kandidat(V, ambang=.1):
    """Himpunan kandidat dari suara bagging. Selalu memuat kelas teratas."""
    c = V >= ambang
    c[np.arange(len(V)), V.argmax(1)] = True
    return c


def proden(Xtr, V, Xs, ronde=4, C=3.0):
    """Partial label gaya PRODEN: bobot kandidat diperbarui dari posterior model, hanya di dalam kandidat."""
    c = kandidat(V)
    T = V * c
    T = T / T.sum(1, keepdims=True)
    for _ in range(ronde):
        p = lr_lunak(Xtr, T, [Xtr], C=C, iter_maks=80)[0]
        T = p * c + 1e-12
        T = T / T.sum(1, keepdims=True)
    return lr_lunak(Xtr, T, Xs, C=C)


def graf_knn(E, k=10, gamma=3):
    """Graf kNN cos simetris pada train. Bobot s^gamma seperti Iscen 2019."""
    S = E @ E.T
    np.fill_diagonal(S, -1)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    r = np.repeat(np.arange(len(E)), k)
    w = np.clip(S[r, idx.ravel()], 0, None) ** gamma
    A = csr_matrix((w, (r, idx.ravel())), shape=(len(E), len(E)))
    A = A.maximum(A.T)
    d = np.asarray(A.sum(1)).ravel()
    Dm = diags(1 / np.sqrt(np.clip(d, 1e-12, None)))
    return Dm @ A @ Dm


def label_spreading(Sn, Y, alpha=.9):
    """Selesaikan (I - alpha S) F = Y per kelas. Y baris nol untuk citra tanpa label."""
    M = identity(Sn.shape[0], format="csr") - alpha * Sn
    F = np.zeros(Y.shape)
    for c in range(Y.shape[1]):
        F[:, c], info = cg(M, Y[:, c], rtol=1e-6, maxiter=500)
        assert info == 0
    F = np.clip(F, 0, None)
    jumlah = F.sum(1, keepdims=True)
    return np.where(jumlah > 1e-12, F / np.clip(jumlah, 1e-12, None), 1 / N_KELAS)


def knn_prediksi(Etr, Ptr, Eq, k=10, suhu=.07):
    """Prediksi induktif: rata-rata tertimbang distribusi tetangga train. Eq tidak mengubah Etr atau Ptr."""
    S = Eq @ Etr.T
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    s = np.take_along_axis(S, idx, 1)
    w = np.exp(s / suhu)
    w = w / w.sum(1, keepdims=True)
    return (Ptr[idx] * w[..., None]).sum(1)


def fitur_pandangan(Xtr, Xs, dim=128):
    """Standarisasi plus PCA, keduanya fit pada Xtr saja."""
    sc = StandardScaler().fit(Xtr)
    pc = PCA(min(dim, min(Xtr.shape) - 1), random_state=0).fit(sc.transform(Xtr))
    return pc.transform(sc.transform(Xtr)), [pc.transform(sc.transform(x)) for x in Xs]


def simpan_json(nama, isi):
    (OUT / nama).write_text(json.dumps(isi, indent=2, ensure_ascii=False), encoding="utf-8")


def bulat(x, n=4):
    return round(float(x), n)


class Waktu:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self, teks):
        print(f"[{time.time() - self.t0:7.1f}s] {teks}", flush=True)
