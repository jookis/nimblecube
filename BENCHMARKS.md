# Benchmarks

All numbers are **measured**, not estimated: on the host for the core ops and the index, and on a real
**ESP32-S3 @ 240 MHz** for the on-device figures. Deterministic, seeded, with `black_box` and a printed
checksum so nothing is optimized away.

## 1. Footprint

| | size |
|---|---|
| one hypervector (`Hv`) | **512 B** (4096 bits) |
| one stored item | 516 B (vector + `u32` id) |
| `FixedStore<64>` | **33 KB** |
| bare-metal Cortex-M image (insert + nearest + hamming + vector table + panic handler) | **2.9 KB flash (2908 B), 0 static RAM** |

The code fits *any* MCU; only the store RAM (`N × 516 B`) scales.

## 2. Core-op throughput (host, x86-64)

`FixedStore::nearest` over N=1024, scalar vs auto-vectorized. The default x86-64 baseline lacks
`POPCNT`, so it lowers to a software popcount; enabling hardware `POPCNT` + AVX2 is a free win (same
integer result, identical checksum):

| build | ns / comparison | speedup |
|---|---|---|
| default (software popcount) | 63.9 | 1.00× |
| `target-feature=+popcnt` | 37.1 | 1.72× |
| `target-cpu=native` (AVX2 + POPCNT) | **25.8** | **2.47×** |

`cargo run --release --example hamming_bench` (set `RUSTFLAGS` for the variants).

## 3. On-device throughput (ESP32-S3 @ 240 MHz)

The same `nimblecube-core` ops on the real chip. The Xtensa LX7 has **no scalar `POPCNT`** (and the Rust/LLVM
backend doesn't autovectorize to its 128-bit SIMD extensions), so `count_ones` is a software popcount -
hence the gap to a desktop with hardware POPCNT.

| op | time |
|---|---|
| `Hv::hamming` | **7.8 µs** |
| `FeatureEncoder::encode` | **103 µs** |
| `FixedStore::nearest` (matched query) | **22 µs** |
| `FixedStore::nearest` (random scan of 64) | 749 µs |
| **full sensor → encode → nearest cycle (matched)** | **≈ 125 µs** |

Ample for sensor anomaly detection (a few Hz). `cd nimblecube-esp32 && cargo run --release --bin bench`.

## 4. Optimizations (all bit-identical)

Each core-op rewrite is verified **byte-for-byte identical** to the original: a property test against the
retained scalar version on the host, *and* an unchanged on-device output checksum:

| op | change | speedup |
|---|---|---|
| `bundle` | bit-serial majority → **word-parallel bit-sliced majority** | **12.7×** (encode 1.31 ms → 103 µs) |
| `nearest` | full scan → **branch-and-bound early termination** | **33×** matched (910 µs → 22 µs)¹ |
| `hamming` | per-word `count_ones` → **Harley-Seal carry-save popcount** | 1.72× (13.4 → 7.8 µs) |

¹ Data-dependent: a clean reading that matches a stored pattern bails after one word; a random/anomaly
query (no close match) prunes less.

The matched detection path went from ~2.2 ms to ~125 µs (~18×) across these.

## 5. Capacity (ESP32-S3-N16R8, 8 MB octal PSRAM)

| store | items | nearest |
|---|---|---|
| internal SRAM (`FixedStore`) | ~256-512 (fast) | ~22 µs matched |
| PSRAM (`Vec<Hv>`) | **~12,000** (~6.1 MB) | 278 ms random / 11 ms matched |

PSRAM buys **capacity, not speed**: it has access latency internal SRAM lacks, so a linear scan over
thousands of vectors is hundreds of ms. That motivates the index below.
`cd nimblecube-esp32 && cargo run --release --bin psram_bench`.

The 278 ms is a *random* query against *random* items, where early termination never fires. On
clustered data with a query near a cluster it prunes hard: the same 12,000-item PSRAM scan measures
**59.7 ms** (§8). That, not 278 ms, is the baseline an index has to beat on realistic data.

## 6. IVF index: sub-linear nearest (host)

An HDC-native inverted-file index: centroid = `Hv::bundle` (majority), assign/probe = `Hv::hamming`,
k-means in Hamming. Measured vs the linear scan, N = 12,000:

| data | recall@1 | speedup |
|---|---|---|
| **clustered** (real sensor/embedding-like) | **100%** | **37×** (321 compares vs 12,000) |
| uniform-random (no structure) | ~2% | n/a (no index helps) |

On structured data (which real sensor and embedding data is), IVF finds the true nearest essentially
always, at ~√N cost. On uniform-random data no index helps: the honest curse-of-dimensionality
boundary. `cargo run --release --example ivf_eval`.

The 37× is a **compare-count ratio, not wall-clock**. §8 measures both for a comparable index and
finds compare count over-predicts on-chip speedup by ~4×, because an index trades many sequential
PSRAM reads for few random ones. Treat every compare-count figure here as an upper bound.

## 7. RejectNet: one-compare rejection (ESP32-S3)

`nearest` terminates early only when a close match tightens the bound, so an *anomalous* query prunes
nothing and pays the full scan. The linear scan is therefore slowest exactly on the path a detector
exists to catch. `RejectNet` bundles every enrolled vector into one `Hv`, so a single `hamming`
answers "is anything here close at all?" for the whole store.

Measured on-chip, N = 64 coherent baselines (spread 100 bits, margin 150):

| query | `nearest` | reject + scan | gain |
|---|---|---|---|
| **anomaly** (nothing close) | 765 µs | **7 µs** | **109×** |
| normal (a fresh clean reading) | 742 µs | 749 µs | ~1% cost |

Correct on 8/8 anomalies and 0/8 normals on-device. The margin is a guarantee, not a tuning knob:
Hamming is a metric, so any query within `margin` bits of an enrolled vector is *structurally* unable
to be rejected. Usefulness depends on how coherent the store is, not how large it is: 256 enrolled
baselines separate as well as 4 (separation ~1946 bits), while 256 *unrelated* vectors collapse to 7
bits and the net simply stops rejecting. That degradation is fail-safe, never wrong.
`cd nimblecube-esp32 && cargo run --release --bin reject_bench`.

Those separations were measured on uniform-random vectors, where the noise floor has
sd = √4096/2 = **32 bits**. SimHash-projected embeddings carry a second term (input wobble,
`DIM/π/√D` ≈ 47, measured in `simhash_eval`), giving **sd ≈ 57**. The same bit separations are
therefore worth ~1.8× fewer standard deviations on embedding data, so treat this envelope as
optimistic for that case until it is measured there.

## 8. Net tree: an incremental index, host and on-chip

Cells are `Hv::bundle` superpositions, assigned greedily as items arrive. No k-means, no training
pass, so enrollment stays one-shot. Host eval, N = 12,000, 200 clusters, 500 queries:

| index | recall@1 | compares | vs linear |
|---|---|---|---|
| **net tree** (cap 64) | **100%** | **260** | 46× |
| IVF (§6) | 100% | 321 | 37× |
| insertion-order cells (control) | 1.8% | 249 | n/a (wrong answers) |
| net tree, uniform-random data | 100% | 12,001 | 1.0× |

The control row matters: cells filled in arrival order score 1.8%, so the greedy assignment is doing
all the work. Enrollment costs ~2.75M compare-equivalents against IVF's ~12M, and can be done one
item at a time. On unstructured data the tree **fails safe**, degenerating to one cell per item so it
stays correct and merely stops helping, where IVF returns wrong answers at ~2% recall. The cost of
that is O(n²) enrollment when nothing clusters.

Then the same index on real hardware, N = 12,000 in PSRAM:

| | per query | recall |
|---|---|---|
| linear scan | 59,652 µs | 100% by definition |
| **net tree** (nprobe 1) | **5,401 µs** | **20/20** |
| net tree (nprobe 2) | 6,685 µs | 20/20 |

**Recall transferred; the speedup did not.** 46× of compare-count became **11×** of wall-clock,
because per-compare cost is not constant: the linear scan streams PSRAM sequentially (4.97 µs per
compare) while the tree fetches scattered cells and members by index (20.8 µs per compare, a 4.2×
penalty from access pattern alone). Building the index is 73 s for a bulk load of 12,000, about 6 ms
per insert. `cd nimblecube-esp32 && cargo run --release --bin net_tree_bench`.

Recall is the number to watch rather than speed. The detector's decision is `d > threshold`, so a
missed nearest returns an inflated `d` and shows up as a **false alarm**. At 100% recall the index is
behaviorally identical to a full scan, which is what makes it safe to substitute.

## 9. SimHash encode cost: is on-device embedding encoding viable?

`simhash_*` projects a dense float embedding into an `Hv`, costing 4096 bits × D input elements.
That is orders of magnitude more work than `FeatureEncoder`, so it decides whether embeddings can
be encoded on the device at all. Measured on-chip:

| encoder | per vector |
|---|---|
| `FeatureEncoder<3,16>` (sensor features) | 115 µs |
| `simhash_i32`, D=128 | 103 ms |
| `simhash_i32`, D=384 | 304 ms |
| **`simhash_i32`, D=768** (typical embedding) | **607 ms** |
| **`simhash_f32`, D=768** | **342 ms** |

Cost is exactly linear in D (0.79 ms per dimension at every size). **On-device embedding encoding
is not viable**: one 768-d vector costs 0.6 s, and a 12,000-item corpus would take over two hours.

The architecture that does work is **encode off-device, search on-device**. Search is unaffected
(`hamming` 8.8 µs, and §8 gets 12,000 items to 5.4 ms), so a host produces the hypervector and the
device stores and searches it.

**`simhash_f32` is 1.77× faster than `simhash_i32` and produces bit-identical codes** (measured
Hamming between the two outputs: 0). The integer path accumulates into `i64` on a 32-bit core, so
each add is multi-instruction, while the LX7 has a single-precision FPU. Prefer the float path on
any target with an FPU. `cd nimblecube-esp32 && cargo run --release --bin simhash_bench`.

### Where encode time actually goes

Measured in one binary, so the parts are comparable to each other:

| part | time | share of `encode` |
|---|---|---|
| `Hv::hamming` (reference point) | 8.8 µs | - |
| 3 × `bind` | 20.8 µs | 18% |
| `bundle` of 3 | 80.3 µs | 70% |
| `encode` (whole) | 115.1 µs | 100% |

`bundle` dominates. For small odd `CH` its bit-sliced counters do far more work than a closed-form
majority (`(a&b)|(a&c)|(b&c)`) would, which is the obvious lever and is **not yet measured on-chip**.

> **Caution on absolute figures.** `FeatureEncoder::encode` measured 92.7 µs, 103.4 µs and 115.1 µs
> in three different binaries in one session, from code layout and instruction-cache effects alone.
> That ±20% is larger than most optimizations worth chasing, so only compare variants built into the
> **same binary**, and treat any single absolute encode number as approximate.

## 10. Encoder codebook: encode as a flash lookup

`FeatureEncoder<CH,L>` is frozen once seed and ranges are chosen, so it can only ever emit `L^CH`
distinct vectors. At CH=3, L=16 that is 4,096 entries × 512 B = **2 MB**, against 16 MB of flash.
Encoding then becomes an index into memory-mapped flash instead of ~100 µs of bind and bundle.
All 4,096 entries verify **bit-identical** to the computed encoder on-device.

The win is entirely a **caching** story: 2 MB cannot fit the flash cache, so a cold 512-byte read
costs ~63 µs by itself. Measured with `hamming` included, since encoding is always followed by a
compare:

| locality | encode + `hamming` | vs computed |
|---|---|---|
| computed encoder (baseline) | 105.9 µs | 1.0× |
| codebook, hot entry (repeat readings) | **7.8 µs** | **13.6×** |
| codebook, 8 entries cycling | 12.0 µs | 8.8× |
| codebook, uniform over all 4,096 | 70.6 µs | 1.5× |

Lookup itself (quantize + index) is 2.1 µs and `hamming` alone is 8.0 µs, so in the hot case the
flash read is effectively free. **Quote the number that matches your access pattern.** A stable
sensor repeats its level tuple (which is why the MQ-2 demo sees `d=0`) and lands in the hot case;
one sweeping across levels lands in the cold case and gains almost nothing.

Generate with `cargo run --release --example gen_codebook -- nimblecube-esp32/codebook.bin`, then
`cd nimblecube-esp32 && cargo run --release --bin codebook_bench`.

The table scales as `L^CH` × 512 B, so CH=3, L=16 fits in 2 MB while CH=4, L=16 would need 33 MB.
Unlike precomputing the *answers*, freezing only the encoder **preserves runtime enrollment and
search**: the store stays mutable.

**Untested idea:** hold the codebook in PSRAM rather than flash. §8 measured PSRAM random reads at
~12.8 µs per 512 B against flash's ~63 µs here, and the whole table can be computed at boot in
~0.43 s (4,096 × 105 µs), which would remove the 2 MB of flash entirely. No number claimed.

## 11. Accuracy vs mature tooling (host, public datasets)

The one benchmark that could sink the project: does nimblecube actually detect anomalies as well as
established methods, or is it only fast? Run on **19 public ADBench/ODDS datasets** (6 to 36 features,
129 to 49,097 rows) under a novelty protocol: train on 60% of the normal rows only, test on the rest
plus every anomaly, 5 seeds. nimblecube uses the real `FeatureEncoder`; baselines are PyOD defaults.
Harness and instructions in [`bench/`](bench/).

Mean over the 19 datasets:

| method | ROC-AUC | avg precision | notes |
|---|---|---|---|
| 1-NN Euclidean on floats | **0.887** | **0.784** | same scoring rule, no encoding |
| nimblecube, 64 levels | 0.872 | 0.762 | |
| nimblecube, 16 levels (device default) | 0.853 | 0.749 | beats HBOS 12/19, LODA 17/19 |
| Isolation Forest | 0.844 | 0.692 | |
| HBOS | 0.799 | 0.613 | |
| LODA | 0.787 | 0.595 | |

**Honest reading.** nimblecube clearly beats the fast histogram baselines (HBOS, LODA) it is usually
compared with, and ties Isolation Forest at 64 levels. It **loses to plain nearest-neighbour on the raw
floats**: the binary encoding costs ~0.03 ROC-AUC here, so on small numeric vectors the encoding buys
no accuracy, only the integer-only, fixed-512-B, `POPCNT`-friendly representation that the rest of these
benchmarks measure. Its edge is where a raw distance is unavailable or expensive (sequences, symbols,
many mixed sensors bound into one vector), not on small dense tables. Resolution matters: 16 to 64
levels recovers about half the gap, but 64 levels would need `64^3 × 512 B` = 128 MB of codebook (§10),
so the device default stays at 16.

## 12. Liquid query: what leaving the query un-snapped is worth (host)

Everything above snaps both sides of a comparison to a cube corner. The alternative keeps the **query**
as raw projection accumulators (`simhash::project_f32`) and leaves the store binary, so each stored bit
only selects add or subtract. No multiplies, and no extra bytes per stored item.

Bussgang's theorem fixes the payoff in closed form: snapping both sides carries a gain of `2/pi`, one
side `sqrt(2/pi)`, so the estimator-variance ratio is `pi/2` at cosine 0.

Measured, 8000 trials per row, input width 256, both estimators unbiased (|bias| <= 0.001):

| cosine | 0.00 | 0.40 | 0.70 | 0.90 |
|---|---|---|---|---|
| variance ratio | **1.552** | 1.608 | 1.608 | 1.142 |
| predicted | 1.571 (`pi/2`) | 1.613 | 1.624 | 1.142 |

**Verdict: not worth doing.** Theory and measurement agree within 1.5%, and the payoff is the constant
`pi/2` and nothing more: a liquid query against the 4096-bit store is worth a binary store of 6355 bits.
Host cost is 59 ns binary vs 5060 ns liquid; that ratio will not transfer (the host has hardware
`POPCNT`, the LX7 does not, §3), but it cannot be favourable enough to matter. The whole question is on
the compare side, and §9 shows compare is not the bottleneck: searching 12,000 items costs 1.6% of one
`simhash` encode.

Kept as a closed door. `cargo run --release --example liquid_query_eval -- [trials]`
