"""Pack the cert_prep.py rows for the on-chip pole bench (nimblecube-esp32/src/bin/pole_bench.rs).

Reads bench/certwork/<name>/{train,test}.txt and writes nimblecube-esp32/certdata.bin (git-ignored),
which the bench bakes in with include_bytes!. Per dataset, at most MAX_STORED stored rows (8 MB PSRAM)
and QUERIES test rows, both sampled with a fixed seed. Layout, little-endian:
    u32 datasets; then per dataset: u8 name length, name, u16 cols, u32 stored, u32 queries,
    stored*cols i32, queries*cols i32, queries u8 labels (1 = anomaly)
    python3 bench/cert_chip_data.py annthyroid mammography satellite pendigits shuttle
"""
import os, struct, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "nimblecube-esp32", "certdata.bin")
MAX_STORED, QUERIES = 12000, 500

def rows(path):
    return np.loadtxt(path, skiprows=1, dtype=np.int64, ndmin=2)

with open(OUT, "wb") as f:
    f.write(struct.pack("<I", len(sys.argv) - 1))
    for name in sys.argv[1:]:
        tr, te = rows(os.path.join(HERE, "certwork", name, "train.txt")), rows(os.path.join(HERE, "certwork", name, "test.txt"))
        rng = np.random.default_rng(0)
        if len(tr) > MAX_STORED:
            tr = tr[np.sort(rng.permutation(len(tr))[:MAX_STORED])]
        te = te[np.sort(rng.permutation(len(te))[:QUERIES])]
        cols = tr.shape[1]
        label = name + ("_12k" if name == "shuttle" else "")
        f.write(struct.pack("<B", len(label)) + label.encode())
        f.write(struct.pack("<HII", cols, len(tr), len(te)))
        f.write(np.clip(tr, -2**31, 2**31 - 1).astype("<i4").tobytes())
        f.write(np.clip(te[:, :cols], -2**31, 2**31 - 1).astype("<i4").tobytes())
        f.write(te[:, cols].astype("u1").tobytes())
        print(f"{label}: {len(tr)} stored, {len(te)} queries ({int(te[:, cols].sum())} anomalies), {cols} cols")
print(f"wrote {os.path.getsize(OUT)} bytes to {os.path.normpath(OUT)}")
