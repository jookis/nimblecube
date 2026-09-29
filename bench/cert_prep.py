"""Write ADBench rows for examples/cert_real_eval.rs: the search index on real encoded data.

Same split and scaling as bench_pyod.py, seed 0: train = a random 60% of the normals (the stored
items), test = the other 40% plus every anomaly (the queries), min-max scaled so the training
normals span 0..1000. Per dataset, writes bench/certwork/<name>/{train,test}.txt in the
tabular_eval row format; test rows carry the label (0 normal, 1 anomaly) as a last column.
    python3 bench/cert_prep.py 2_annthyroid 23_mammography 28_pendigits 30_satellite 32_shuttle
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

def write_rows(path, M):
    with open(path, "w") as f:
        f.write(f"{M.shape[0]} {M.shape[1]}\n")
        np.savetxt(f, M, fmt="%d")

for stem in sys.argv[1:]:
    d = np.load(os.path.join(HERE, "adbench", stem + ".npz"), allow_pickle=True)
    X, y = d["X"].astype(float), d["y"].astype(int)
    rng = np.random.default_rng(0)
    normal = rng.permutation(np.flatnonzero(y == 0))
    ntr = int(0.6 * len(normal))
    tr, te = normal[:ntr], np.concatenate([normal[ntr:], np.flatnonzero(y == 1)])
    lo, hi = X[tr].min(axis=0), X[tr].max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    scale = lambda A: np.clip(np.round((A - lo) / span * 1000), -2**31, 2**31 - 1).astype(np.int64)
    out = os.path.join(HERE, "certwork", stem.split("_", 1)[1])
    os.makedirs(out, exist_ok=True)
    write_rows(os.path.join(out, "train.txt"), scale(X[tr]))
    write_rows(os.path.join(out, "test.txt"), np.column_stack([scale(X[te]), y[te]]))
    print(f"{stem}: {len(tr)} stored, {len(te)} queries ({int(y[te].sum())} anomalies)")
