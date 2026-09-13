# Accuracy benchmark: nimblecube vs PyOD baselines

Answers the README's open question, "is nimblecube competitive with mature anomaly-detection tooling?",
on public tabular datasets. The Rust example does the nimblecube half; this Python harness handles the
public baselines (which only exist in Python) and scoring.

Results are in [`../BENCHMARKS.md`](../BENCHMARKS.md) §11.

## What it does

For each dataset and 5 random seeds it trains on 60% of the *normal* rows only (novelty detection, how
nimblecube runs on a device), tests on the held-out normals plus every anomaly, and scores every method
on the same split by ROC-AUC and average precision.

- **nimblecube:** rows scaled so the training normals span `0..1000`, encoded by the real
  `FeatureEncoder` in `examples/tabular_eval.rs`; score = Hamming distance to the nearest enrolled
  normal (`nc nearest`) or to their bundle (`nc centroid`).
- **Baselines (PyOD defaults):** HBOS, LODA, Isolation Forest, and 1-NN Euclidean on standardized
  floats (the same scoring rule as `nc nearest`, without the encoding).

## Run it

```bash
# 1. build the nimblecube scorer
cargo build --release --example tabular_eval

# 2. Python deps (uv shown; pip works too)
uv venv bench/env && uv pip install --python bench/env/bin/python pyod scikit-learn numpy

# 3. datasets: the ADBench "Classical" .npz files into bench/adbench/
mkdir -p bench/adbench && cd bench/adbench
for n in 2_annthyroid 4_breastw 6_cardio 18_Ionosphere 23_mammography 27_PageBlocks \
         28_pendigits 29_Pima 30_satellite 31_satimage-2 32_shuttle 38_thyroid \
         39_vertebral 40_vowels 42_WBC 43_WDBC 44_Wilt 45_wine 47_yeast; do
  curl -sfLO https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/$n.npz
done && cd ../..

# 4. run (default: min-max scaling, 16 levels)
bench/env/bin/python bench/bench_pyod.py target/release/examples/tabular_eval
# variants: <scaling: minmax|quantile> <levels: 16|64> [nc-only]
bench/env/bin/python bench/bench_pyod.py target/release/examples/tabular_eval minmax 64
```

Datasets and the `env/` and `tabwork_*/` working directories are git-ignored; only the code is tracked.
ADBench is MIT-licensed; the datasets are redistributed there from the ODDS collection.
