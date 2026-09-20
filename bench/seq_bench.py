"""Sequence benchmark: nimblecube against the usual sequence baselines (BENCHMARKS.md §13).

Data: UCI UNIX user command sessions prepared by seq_prep.py (9 users, chronological 70/30 split).
Question: §11 says the encoding's edge should be where a raw distance is unavailable or expensive, such as
sequences. This is that test.

Methods (baselines hand-written, numpy only; no scikit-learn needed)
  nimblecube 1-NN       nearest training session by Hamming distance between 4096-bit n-gram bundles
  nimblecube centroid   nearest class bundle: one 512-byte vector per user, 9 comparisons per query
  n-gram cosine 1-NN    exact sparse cosine over n-gram counts, via an inverted index
  hashed TF-IDF+linear  n-grams hashed to 4096 dimensions (the same budget as a hypervector), TF-IDF,
                        multinomial logistic regression trained by gradient descent
  edit distance 1-NN    Levenshtein against every training session: the "proper" sequence method
  compression 1-NN      normalised compression distance with zlib: parameter-free sequence distance
  majority class        the floor
Reported: accuracy, macro-F1 (classes are unbalanced), time per query, bytes per stored item. Timings mix
Rust, numpy and plain Python, so only accuracy and bytes are comparable across rows.

Run, after seq_prep.py and after running the Rust half for each n:
    cargo run --release --example seq_eval -- <prep>/train.txt <prep>/test.txt <prep>/hdc_n3.txt 3
  python3 seq_bench.py <prep dir> <dir with hdc_n*.txt> [--sub=150]
The Rust example prints its own timing; put it in <dir>/hdc_n<N>.time.json as {"us_per_query": ...} if you
want the timing column filled for those rows.
"""
import sys, os, time, zlib, json
from collections import Counter, defaultdict
import numpy as np

prep, hdcdir = sys.argv[1], sys.argv[2]
SUB = int(next((a.split('=')[1] for a in sys.argv if a.startswith('--sub=')), 150))
STEM = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seq_bench_result')

def load(p):
    lines = open(p).read().splitlines()
    rows = [l.split() for l in lines[1:]]
    return [[int(v) for v in r[1:]] for r in rows], np.array([int(r[0]) for r in rows]), int(lines[0].split()[1])

Xtr, ytr, vocab = load(os.path.join(prep, 'train.txt'))
Xte, yte, _ = load(os.path.join(prep, 'test.txt'))
NC = int(ytr.max()) + 1
print(f"{len(Xtr)} train, {len(Xte)} test, {NC} users, vocab {vocab}, "
      f"median length {sorted(len(s) for s in Xte)[len(Xte)//2]}, "
      f"majority class share {np.bincount(yte).max()/len(yte):.3f}")

res = []
def report(pred, truth, name, t_per, bytes_item, note=''):
    pred = np.asarray(pred); truth = np.asarray(truth)
    acc = (pred == truth).mean()
    f1 = []
    for c in range(NC):
        tp = ((pred == c) & (truth == c)).sum(); fp = ((pred == c) & (truth != c)).sum(); fn = ((pred != c) & (truth == c)).sum()
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        f1.append(2 * p * r / max(p + r, 1e-9))
    print(f"  {name:26s}{acc:9.3f}{np.mean(f1):9.3f}{t_per*1e6:11.0f}{bytes_item:11.0f}  {note}")
    res.append({'name': name, 'accuracy': float(acc), 'macro_f1': float(np.mean(f1)),
                'us_per_query': t_per * 1e6, 'bytes_per_item': float(bytes_item), 'note': note})

print(f"\n  {'method':26s}{'accuracy':>9s}{'macro F1':>9s}{'us/query':>11s}{'bytes/item':>11s}")
report(np.full(len(yte), np.bincount(ytr).argmax()), yte, 'majority class', 0, 0)

for n in (1, 2, 3, 4):
    p = os.path.join(hdcdir, f'hdc_n{n}.txt')
    if not os.path.exists(p): continue
    rows = [l.split() for l in open(p).read().splitlines()]
    nn = np.array([int(r[0]) for r in rows])
    cent = np.array([[int(v) for v in r[2:]] for r in rows]).argmin(1)
    tp = os.path.join(hdcdir, f'hdc_n{n}.time.json')
    t = json.load(open(tp))['us_per_query'] / 1e6 if os.path.exists(tp) else 0.0
    report(nn, yte, f'nimblecube 1-NN, n={n}', t, 512)
    report(cent, yte, f'nimblecube centroid, n={n}', t * NC / len(Xtr), 512, '9 vectors total')

def grams(s, n):
    return [tuple(s[i:i + n]) for i in range(len(s) - n + 1)] if len(s) >= n else [tuple(s)]

for n in (1, 2, 3):
    post = defaultdict(list)                      # gram -> (train index, count)
    norms = np.zeros(len(Xtr))
    for i, s in enumerate(Xtr):
        c = Counter(grams(s, n))
        for g, v in c.items(): post[g].append((i, v))
        norms[i] = np.sqrt(sum(v * v for v in c.values()))
    t0 = time.time(); pred = []
    for s in Xte:
        acc = np.zeros(len(Xtr))
        c = Counter(grams(s, n))
        for g, v in c.items():
            for i, w in post.get(g, ()): acc[i] += v * w
        q = np.sqrt(sum(v * v for v in c.values())) or 1.0
        pred.append(ytr[int(np.argmax(acc / (norms * q + 1e-9)))])
    t = (time.time() - t0) / len(Xte)
    avg_nnz = np.mean([len(set(grams(s, n))) for s in Xtr])
    report(pred, yte, f'n-gram cosine 1-NN, n={n}', t, avg_nnz * 8)

    D = 4096                                       # same budget as one hypervector
    def hashed(X):
        M = np.zeros((len(X), D), np.float32)
        for i, s in enumerate(X):
            for g, v in Counter(grams(s, n)).items(): M[i, hash(g) % D] += v
        return M
    A, B = hashed(Xtr), hashed(Xte)
    idf = np.log(len(A) / ((A > 0).sum(0) + 1))
    nz = lambda M: M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
    At, Bt = nz(nz(A) * idf), nz(nz(B) * idf)
    W = np.zeros((D, NC), np.float32); Y = np.eye(NC, dtype=np.float32)[ytr]
    for _ in range(400):
        Z = At @ W; Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
        W -= 2.0 * (At.T @ (P - Y) / len(At) + 1e-4 * W)
    t0 = time.time(); pred = (Bt @ W).argmax(1); t = (time.time() - t0) / len(Xte)
    report(pred, yte, f'hashed TF-IDF + linear, n={n}', t, D * NC * 4 / len(Xtr), 'model shared')

sub = np.random.default_rng(0).choice(len(Xte), min(SUB, len(Xte)), replace=False)
def lev(a, b):
    if len(a) < len(b): a, b = b, a
    prev = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        cur = np.empty_like(prev); cur[0] = i
        sub_cost = prev[:-1] + (np.array(b) != ca)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, sub_cost[j - 1])
        prev = cur
    return prev[-1]

t0 = time.time()
pred = [ytr[int(np.argmin([lev(Xte[i], s) / max(len(Xte[i]), len(s)) for s in Xtr]))] for i in sub]
t = (time.time() - t0) / len(sub)
report(pred, yte[sub], 'edit distance 1-NN', t, np.mean([len(s) for s in Xtr]) * 4, f'on {len(sub)} test sequences')

blob = lambda s: bytes((v % 251) + 1 for v in s)
tr_b = [blob(s) for s in Xtr]; tr_c = np.array([len(zlib.compress(b, 6)) for b in tr_b])
t0 = time.time(); pred = []
for i in sub:
    b = blob(Xte[i]); cb = len(zlib.compress(b, 6))
    d = [(len(zlib.compress(tb + b, 6)) - min(tc, cb)) / max(tc, cb) for tb, tc in zip(tr_b, tr_c)]
    pred.append(ytr[int(np.argmin(d))])
t = (time.time() - t0) / len(sub)
report(pred, yte[sub], 'compression 1-NN', t, tr_c.mean(), f'on {len(sub)} test sequences')

json.dump(res, open(STEM + '.json', 'w'), indent=1)
print(f"\nsaved {STEM}.json")
