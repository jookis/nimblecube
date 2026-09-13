# Nimblecube

> An integer-only, dependency-free **hyperdimensional-computing (HDC)** similarity memory for the
> edge, built to run where float vector databases and neural runtimes can't: on a microcontroller,
> offline, in kilobytes.

[![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm%20Noncommercial-blue.svg)](LICENSE.md)

## Where this came from

Nimblecube started as an attempt to build a **semantic cache for LLM embeddings**. Along the way I
noticed the representation *underneath* (binary, integer-only **hypervectors** compared by **Hamming
distance**) is a far more natural fit for **microcontrollers** than for the cloud: no floats, no FPU,
kilobytes of RAM, one-shot learning, graceful under bit-flips. So the edge engine became the real
project: an integer HDC associative memory that runs where the float/neural stack can't.

## What it is

A 4096-bit binary vector ("hypervector") fingerprints any pattern: a sensor window, a feature vector,
a token. Similarity is **Hamming distance** = `XOR + popcount`, a single integer path with no
floating-point. You enroll patterns into a fixed store and recall the nearest. On clear signals this is
a fast, robust, fully on-device **anomaly / pattern detector**.

- **`Hv`**: a 4096-bit hypervector with `bind` (XOR), `bundle` (bitwise majority = superposition),
  `hamming` (popcount distance), `permute` (rotate).
- **`FixedStore<N>`**: an inline, heap-free associative memory with `insert` + nearest-by-Hamming.
- **`FeatureEncoder`**: turns quantized integer sensor features into a hypervector (level-encode →
  bind-per-channel → bundle), so similar inputs get similar codes.

All `#![no_std]`, alloc-free, zero-dependency.

## Proven on real hardware

The end-to-end edge story runs on an **ESP32-S3** (Xtensa, 240 MHz):

- An **MQ-2 gas/smoke anomaly detector** that auto-enrolls "clean air" and prints `ANOMALY` on smoke
  (clean `d=0` vs smoke `d=233`), stability-verified over minutes with **zero false alarms**.
- A **full detection cycle in ~125 µs**, integer-only, offline.
- Footprint down to **~2.9 KB flash, 0 static RAM** (bare-metal Cortex-M image).

Every performance number is **measured on the chip** (see [BENCHMARKS.md](BENCHMARKS.md)).

## Layout

| crate | what it is |
|---|---|
| **`nimblecube-core/`** | the engine (`Hv`, `FixedStore`, `FeatureEncoder`). `no_std`, alloc-free, zero-dep; host examples live here. |
| **`nimblecube-esp32/`** | ESP32-S3 firmware (esp-hal): the MQ-2 detector, on-chip timing bench, PSRAM capacity bench. |
| **`nimblecube-firmware/`** | bare-metal Cortex-M image: the flash/RAM-footprint proof. |

## Quickstart

```bash
# core engine: tests + the three optimizations (word-parallel bundle, early-term nearest, Harley-Seal popcount)
cargo test -p nimblecube-core

# host examples
cargo run --release --example smart_sensor   # synthetic sensor -> encode -> detect anomaly
cargo run --release --example ivf_eval        # IVF index: recall vs speed frontier (sub-linear nearest)
cargo run --release --example hamming_bench    # core-op throughput on the host
cargo run --release --example dna_eval         # k-mer encoding with bind + permute: strands, shifts, errors
cargo run --release --example relative_features # per-unit baseline features: same result on every board
cargo run --release --example tabular_eval      # scorer for the accuracy benchmark (see bench/)
```

The microcontroller crates need Espressif's Rust toolchain (`espup`) and target hardware (see
[`nimblecube-esp32/README.md`](nimblecube-esp32/README.md)).

## Honest scope

This is a **proven substrate, not a packaged product.** What it is and isn't:

- ✅ The integer-HDC core is real, tested, and hardware-proven; the optimizations are bit-identical
  (property-tested + on-device checksum-verified).
- ✅ An **IVF index** makes nearest-search sub-linear on *structured* data (100% recall at ~37× fewer
  compares), measured on the host.
- ⚠️ It is **unproven against mature edge-ML tooling** (Edge Impulse, ST NanoEdge AI) on a real task;
  the sensor demo has been exercised with smoke vs clean air, not yet a calibrated test rig.
- ⚠️ On **uniform-random** data no index helps (the curse of dimensionality); IVF wins because real
  data clusters. Stated plainly because it matters.

The differentiation is the **substrate**: integer HDC, `no_std`, zero-dep, one-shot enrollment, running
where the float/NN stack won't fit. The common edge *use cases* (gesture/keyword/anomaly) are well
served by mature tools; this packages the *different computational core* underneath them.

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md): **free for any noncommercial use** (personal, research,
education, evaluation). **Commercial use requires a separate license.** See the license for details.
