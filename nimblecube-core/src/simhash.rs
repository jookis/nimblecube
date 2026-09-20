//! SimHash: random-hyperplane projection of a dense vector into an `Hv`.
//!
//! Complements `FeatureEncoder`, which encodes quantized sensor channels. This
//! encodes a dense vector (an embedding) so that Hamming distance in the `Hv`
//! tracks the angle between the inputs:
//!
//! ```text
//! P[bit differs] = theta / pi
//! ```
//!
//! so at 4096 bits identical inputs give 0, orthogonal inputs give ~2048, and a
//! cosine of 0.9 gives ~590. That is the same normalization `Hv::similarity`
//! already uses (0.5 = orthogonal), so projected vectors drop straight into
//! `FixedStore` and the IVF index without any further scaling.
//!
//! Hyperplanes are never stored. Each output bit derives its own sign pattern
//! from `seed` and the bit index, so encoding costs no RAM beyond the output:
//! 4096 explicit hyperplanes over a 768-dim input would be about 12 MB, which
//! no target here has. Signs are +/-1 (Rademacher) rather than Gaussian, which
//! keeps the projection to adds and subtracts and preserves the angle law.
//!
//! Same `seed` always yields the same code, so a store enrolled with one seed
//! must be queried with it.

use crate::hv::{Hv, DIM_BITS, WORDS};

/// splitmix64 finalizer: gives each output bit an independent, reproducible
/// stream derived from `seed` and the bit index, with nothing stored.
#[inline]
fn mix(seed: u64, bit: usize) -> u64 {
    let mut z = seed.wrapping_add((bit as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xorshift64, the repo's deterministic RNG (see `encode.rs`).
#[inline]
fn xs(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Project a quantized integer vector into an `Hv`. Integer-only path: no FPU,
/// no multiplies. The accumulator is `i64`, so a full-scale `i32` input cannot
/// overflow it at any realistic dimension.
///
/// An empty input yields the all-zero `Hv`.
pub fn simhash_i32(v: &[i32], seed: u64) -> Hv {
    let mut out = [0u64; WORDS];
    for b in 0..DIM_BITS {
        let mut state = mix(seed, b) | 1;
        let mut acc: i64 = 0;
        let mut i = 0;
        while i < v.len() {
            let signs = xs(&mut state);
            let n = if v.len() - i < 64 { v.len() - i } else { 64 };
            for k in 0..n {
                // Branchless. The sign bits are random, so a branch here
                // mispredicts about half the time. mask is 0 to keep x and
                // all-ones to negate it, since (x ^ -1) + 1 == -x.
                let x = v[i + k] as i64;
                let mask = ((signs >> k) & 1) as i64 - 1;
                acc += (x ^ mask) - mask;
            }
            i += n;
        }
        // Ties (acc == 0) resolve to 0 so the mapping stays deterministic.
        if acc > 0 {
            out[b / 64] |= 1u64 << (b % 64);
        }
    }
    Hv(out)
}

/// Project a float vector into an `Hv`. Same construction as `simhash_i32`,
/// for hosts and targets with an FPU. Only adds and comparisons are used, so
/// no libm dependency is introduced.
///
/// An empty input yields the all-zero `Hv`.
pub fn simhash_f32(v: &[f32], seed: u64) -> Hv {
    let mut out = [0u64; WORDS];
    for b in 0..DIM_BITS {
        let mut state = mix(seed, b) | 1;
        let mut acc: f32 = 0.0;
        let mut i = 0;
        while i < v.len() {
            let signs = xs(&mut state);
            let n = if v.len() - i < 64 { v.len() - i } else { 64 };
            for k in 0..n {
                // Same idea as the integer path: flip the IEEE sign bit rather
                // than branch. Still no multiply, so still no libm.
                let flip = ((((signs >> k) & 1) ^ 1) << 31) as u32;
                acc += f32::from_bits(v[i + k].to_bits() ^ flip);
            }
            i += n;
        }
        if acc > 0.0 {
            out[b / 64] |= 1u64 << (b % 64);
        }
    }
    Hv(out)
}

/// The raw projection accumulators behind `simhash_f32`, written to `out`
/// instead of being thresholded: the "liquid" point, before it is snapped to a
/// cube corner. `simhash_f32(v, seed)` is exactly `out[b] > 0.0` per bit.
///
/// Costs 16 KB of caller-provided space at `DIM_BITS = 4096`, which is why the
/// encoder itself does not go through here: the chip path must stay alloc-free
/// and 512 bytes wide. This exists for asymmetric scoring on the query side,
/// where one liquid vector is held per query rather than per stored item.
///
/// An empty input yields all-zero accumulators.
pub fn project_f32(v: &[f32], seed: u64, out: &mut [f32; DIM_BITS]) {
    for b in 0..DIM_BITS {
        let mut state = mix(seed, b) | 1;
        let mut acc: f32 = 0.0;
        let mut i = 0;
        while i < v.len() {
            let signs = xs(&mut state);
            let n = if v.len() - i < 64 { v.len() - i } else { 64 };
            for k in 0..n {
                let flip = ((((signs >> k) & 1) ^ 1) << 31) as u32;
                acc += f32::from_bits(v[i + k].to_bits() ^ flip);
            }
            i += n;
        }
        out[b] = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(state: &mut u64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let r = xs(state);
                ((r >> 40) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    }

    /// `project_f32` is the un-snapped `simhash_f32`: thresholding its output at
    /// zero must reproduce the packed code bit for bit, or the asymmetric score
    /// is measuring a different projection than the store was built with.
    #[test]
    fn project_sign_matches_simhash() {
        let mut state = 0x1234_5678_9abc_def1u64;
        for d in [1usize, 7, 64, 65, 128, 768] {
            let v = rand_vec(&mut state, d);
            let packed = simhash_f32(&v, 99);
            let mut acc = [0f32; DIM_BITS];
            project_f32(&v, 99, &mut acc);
            for b in 0..DIM_BITS {
                let from_acc = acc[b] > 0.0;
                let from_packed = (packed.0[b / 64] >> (b % 64)) & 1 == 1;
                assert_eq!(from_acc, from_packed, "bit {} diverged at d={}", b, d);
            }
        }
    }

    /// Empty input must not leave the accumulators uninitialized or non-zero.
    #[test]
    fn project_of_empty_is_zero() {
        let mut acc = [1f32; DIM_BITS];
        project_f32(&[], 7, &mut acc);
        assert!(acc.iter().all(|&x| x == 0.0));
    }
}
