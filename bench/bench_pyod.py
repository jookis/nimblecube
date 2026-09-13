"""nimblecube vs PyOD baselines on public ADBench/ODDS datasets (host).

Protocol (novelty detection, how nimblecube is used on a device): each seed trains on a random
60% of the normal rows only; the test set is the other 40% of normals plus every anomaly. All
methods see the same split. Scores: ROC-AUC and average precision (AP), mean over 5 seeds.

nimblecube: rows scaled so the training normals span 0..1000, encoded by the real
FeatureEncoder (16 levels) in examples/tabular_eval.rs; score = Hamming distance to the nearest
training normal ("nc nearest") or to the bundle of all of them ("nc centroid").
Baselines, PyOD defaults: HBOS, LODA, IForest; plus 1-NN Euclidean on standardized floats, the
same scoring rule as nc nearest without the encoding.
"""
import glob, os, subprocess, sys, time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from pyod.models.hbos import HBOS
from pyod.models.loda import LODA
from pyod.models.iforest import IForest
from pyod.models.knn import KNN

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = sys.argv[1]                      # target/release/examples/tabular_eval
SCALING = sys.argv[2] if len(sys.argv) > 2 else "minmax"   # minmax | quantile
LEVELS = int(sys.argv[3]) if len(sys.argv) > 3 else 16
NC_ONLY = len(sys.argv) > 4 and sys.argv[4] == "nc-only"
SEEDS = range(5)
WORK = os.path.join(HERE, f"tabwork_{SCALING}_{LEVELS}")
os.makedirs(WORK, exist_ok=True)

def write_rows(path, M):
    with open(path, "w") as f:
        f.write(f"{M.shape[0]} {M.shape[1]}\n")
        np.savetxt(f, M, fmt="%d")

def nimblecube(Xtr, Xte):
    lo, hi = Xtr.min(axis=0), Xtr.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    if SCALING == "minmax":
        scale = lambda X: np.clip(np.round((X - lo) / span * 1000), -2**31, 2**31 - 1).astype(np.int64)
    else:
        # equal-frequency levels inside the training range; the two outer levels are kept for
        # values beyond it, so out-of-range anomalies stay distinguishable from the extremes
        srt = np.sort(Xtr, axis=0)
        def scale(X):
            q = np.stack([np.searchsorted(srt[:, j], X[:, j], side="right") for j in range(X.shape[1])], 1)
            v = (1 + (LEVELS - 2) * np.minimum(q / len(srt), 0.9999)) / LEVELS * 1000
            v = np.where(X < lo, 0, np.where(X > hi, 1000, v))
            return np.round(v).astype(np.int64)
    tr, te, out = (os.path.join(WORK, n) for n in ("train.txt", "test.txt", "scores.txt"))
    write_rows(tr, scale(Xtr)); write_rows(te, scale(Xte))
    subprocess.run([EXE, tr, te, out, str(LEVELS)], check=True)
    s = np.loadtxt(out, ndmin=2)
    return {"nc nearest": s[:, 0], "nc centroid": s[:, 1]}

def baselines(Xtr, Xte, seed):
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    out = {}
    for name, m in (("HBOS", HBOS()), ("LODA", LODA()), ("IForest", IForest(random_state=seed)),
                    ("1-NN float", KNN(n_neighbors=1))):
        m.fit(Ztr)
        out[name] = m.decision_function(Zte)
    return out

METHODS = ["nc nearest", "nc centroid"] + ([] if NC_ONLY else ["HBOS", "LODA", "IForest", "1-NN float"])
print(f"nimblecube scaling={SCALING} levels={LEVELS}")
files = sorted(glob.glob(os.path.join(HERE, "adbench", "*.npz")),
               key=lambda p: int(os.path.basename(p).split("_")[0]))
auc_all, ap_all = {m: [] for m in METHODS}, {m: [] for m in METHODS}
print(f"{'dataset':15} {'rows':>6} {'cols':>4} {'anom%':>5} | ROC-AUC: " + " ".join(f"{m:>11}" for m in METHODS))
for f in files:
    name = os.path.basename(f)[:-4].split("_", 1)[1]
    d = np.load(f, allow_pickle=True)
    X, y = d["X"].astype(float), d["y"].astype(int)
    auc, ap = {m: [] for m in METHODS}, {m: [] for m in METHODS}
    t = time.time()
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        normal, anom = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
        normal = rng.permutation(normal)
        ntr = int(0.6 * len(normal))
        tr, te = normal[:ntr], np.concatenate([normal[ntr:], anom])
        scores = nimblecube(X[tr], X[te])
        if not NC_ONLY:
            scores.update(baselines(X[tr], X[te], seed))
        for m in METHODS:
            auc[m].append(roc_auc_score(y[te], scores[m]))
            ap[m].append(average_precision_score(y[te], scores[m]))
    for m in METHODS:
        auc_all[m].append(np.mean(auc[m])); ap_all[m].append(np.mean(ap[m]))
    print(f"{name:15} {len(y):6} {X.shape[1]:4} {100*y.mean():5.1f} |          "
          + " ".join(f"{np.mean(auc[m]):11.3f}" for m in METHODS) + f"   ({time.time() - t:.0f} s)", flush=True)

print("\nmean ROC-AUC over datasets:  " + " ".join(f"{m}={np.mean(auc_all[m]):.3f}" for m in METHODS))
print("mean AP over datasets:       " + " ".join(f"{m}={np.mean(ap_all[m]):.3f}" for m in METHODS))
ranks = np.array([[sorted(METHODS, key=lambda m: -auc_all[m][i]).index(m) + 1 for m in METHODS]
                  for i in range(len(files))])
print("mean ROC-AUC rank (1 best):  " + " ".join(f"{m}={r:.2f}" for m, r in zip(METHODS, ranks.mean(axis=0))))
print("per-dataset nc nearest ROC-AUC: " + " ".join(f"{a:.3f}" for a in auc_all["nc nearest"]))
wins = {m: sum(auc_all["nc nearest"][i] > auc_all[m][i] for i in range(len(files))) for m in METHODS[1:]}
print("nc nearest beats, datasets:  " + " ".join(f"{m}={w}/{len(files)}" for m, w in wins.items()))
