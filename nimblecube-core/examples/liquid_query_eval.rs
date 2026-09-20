//! Asymmetric ("liquid query") scoring vs the fully binary path. Host/std, deterministic.
//!   cargo run --release --example liquid_query_eval
//!
//! The store stays binary. Only the query is held un-snapped, as the raw
//! projection accumulators behind `simhash_f32` (`project_f32`). Scoring a
//! liquid query against a binary item needs no multiplies: each stored bit
//! selects add or subtract on the query's accumulator.
//!
//! Bussgang's theorem predicts exactly what that buys. With both sides snapped
//! the estimator carries a gain of 2/pi; with one side left liquid it carries
//! sqrt(2/pi). Converting to estimator variance at fixed width:
//!
//!   Var(rho_binary) = pi^2 * (1-rho^2) * p * (1-p) / DIM_BITS,  p = theta/pi
//!   Var(rho_liquid) = (pi/2) * (1 - (2/pi) * rho^2 * (2.5 - rho^2)) / DIM_BITS
//!
//! whose ratio is pi/2 = 1.5708 at rho = 0. So a liquid query against a
//! 4096-bit store should be worth a fully binary store of 4096 * pi/2 = 6434
//! bits, and no more. This measures whether that constant actually shows up.
//!
//! Column `ratio` is the payoff. `predicted` is the closed form above. The
//! `bias` columns are the self-check: if either estimator is not centred on the
//! true cosine, its variance is not comparable and the ratio means nothing.

use std::time::Instant;

use nimblecube_core::hv::{Hv, DIM_BITS, WORDS};
use nimblecube_core::simhash::{project_f32, simhash_f32};

const D: usize = 256; // input width; large enough for the +/-1 signs to look Gaussian
const TRIALS: usize = 1500; // override with argv[1]
const SEED: u64 = 0x5111_4A57_0000_0002;
const PI: f64 = std::f64::consts::PI;

fn xs(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// Uniform in [-1, 1).
fn rand_f32(s: &mut u64) -> f32 {
    ((xs(s) >> 40) as f32 / 8_388_608.0) - 1.0
}

fn rand_vec(s: &mut u64) -> Vec<f32> {
    (0..D).map(|_| rand_f32(s)).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let n = dot(&v, &v).sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x = (*x as f64 / n) as f32;
        }
    }
    v
}

/// A pair at an exact angle, as in `simhash_eval`: v = cos(t)*u + sin(t)*w, w orthonormal to u.
fn pair_at_angle(t: f64, s: &mut u64) -> (Vec<f32>, Vec<f32>) {
    let u = unit(rand_vec(s));
    let mut w = rand_vec(s);
    let proj = dot(&w, &u);
    for i in 0..D {
        w[i] = (w[i] as f64 - proj * u[i] as f64) as f32;
    }
    let w = unit(w);
    let v = (0..D).map(|i| (t.cos() * u[i] as f64 + t.sin() * w[i] as f64) as f32).collect();
    (u, v)
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64]) -> f64 {
    let m = mean(xs);
    xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len() - 1) as f64
}

/// Pack the sign of each accumulator: exactly what `simhash_f32` stores.
fn snap(acc: &[f32; DIM_BITS]) -> Hv {
    let mut out = [0u64; WORDS];
    for b in 0..DIM_BITS {
        if acc[b] > 0.0 {
            out[b / 64] |= 1u64 << (b % 64);
        }
    }
    Hv(out)
}

/// Scale accumulators to unit RMS so the estimator is self-calibrating and
/// never needs the query's original norm.
fn normalize(acc: &mut [f32; DIM_BITS]) {
    let ss: f64 = acc.iter().map(|x| *x as f64 * *x as f64).sum();
    let rms = (ss / DIM_BITS as f64).sqrt();
    if rms > 0.0 {
        for x in acc.iter_mut() {
            *x = (*x as f64 / rms) as f32;
        }
    }
}

/// Asymmetric score: liquid query against a binary item. No multiplies. Each
/// stored bit picks add or subtract, via the IEEE sign bit as `simhash_f32` does.
fn asym_score(q: &[f32; DIM_BITS], item: &Hv) -> f64 {
    let mut acc = 0f32;
    for w in 0..WORDS {
        let mut bits = item.0[w];
        let base = w * 64;
        for k in 0..64 {
            let flip = (((bits & 1) ^ 1) << 31) as u32;
            acc += f32::from_bits(q[base + k].to_bits() ^ flip);
            bits >>= 1;
        }
    }
    acc as f64 / DIM_BITS as f64
}

/// Closed-form variance ratio from Bussgang, for the `predicted` column.
fn predicted_ratio(rho: f64) -> f64 {
    let p = rho.clamp(-1.0, 1.0).acos() / PI;
    let r2 = rho * rho;
    let v_bin = PI * PI * (1.0 - r2) * p * (1.0 - p);
    // The query is rescaled by its own empirical RMS, so the estimator is a
    // ratio R/Q, not a plain sum. The delta method on that ratio contributes
    // -2*mu*Cov(R,Q) + mu^2*Var(Q), which is what the (2.5 - rho^2) factor
    // collects. Dropping it (using the plain 1 - (2/pi)*rho^2) overstates the
    // liquid variance badly above cosine 0.4 and agrees only at cosine 0.
    let v_liq = (PI / 2.0) * (1.0 - (2.0 / PI) * r2 * (2.5 - r2));
    v_bin / v_liq
}

fn main() {
    let trials: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(TRIALS);
    let mut s = SEED;
    let gain = (2.0 / PI).sqrt(); // Bussgang gain for one snapped side

    println!("liquid query vs binary: estimator variance at DIM_BITS={}", DIM_BITS);
    println!("input D={} trials={} per row\n", D, trials);
    println!(
        "   {:>7}  {:>9}  {:>9}  {:>9}  {:>9}  {:>7}  {:>9}",
        "cosine", "bias_bin", "bias_liq", "sd_bin", "sd_liq", "ratio", "predicted"
    );

    let mut ratios = Vec::new();
    for target in [0.0_f64, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9] {
        let t = target.acos();
        let mut est_bin = Vec::with_capacity(trials);
        let mut est_liq = Vec::with_capacity(trials);

        for _ in 0..trials {
            let (a, b) = pair_at_angle(t, &mut s);
            // A fresh projection seed per trial isolates the hyperplane noise,
            // which is the randomness Bussgang's theorem is a statement about.
            let pseed = xs(&mut s) | 1;

            let mut acc_a = [0f32; DIM_BITS];
            let mut acc_b = [0f32; DIM_BITS];
            project_f32(&a, pseed, &mut acc_a);
            project_f32(&b, pseed, &mut acc_b);

            let code_a = snap(&acc_a);
            let code_b = snap(&acc_b);

            // Fully binary: invert the angle law to get back a cosine.
            let h = code_a.hamming(&code_b) as f64 / DIM_BITS as f64;
            est_bin.push((PI * h).cos());

            // Liquid query (a) against the binary stored item (b).
            normalize(&mut acc_a);
            est_liq.push(asym_score(&acc_a, &code_b) / gain);
        }

        let (v_bin, v_liq) = (variance(&est_bin), variance(&est_liq));
        let ratio = v_bin / v_liq;
        ratios.push((target, ratio, predicted_ratio(target)));
        println!(
            "   {:>7.2}  {:>+9.4}  {:>+9.4}  {:>9.4}  {:>9.4}  {:>7.3}  {:>9.3}",
            target,
            mean(&est_bin) - target,
            mean(&est_liq) - target,
            v_bin.sqrt(),
            v_liq.sqrt(),
            ratio,
            predicted_ratio(target)
        );
    }

    let m = mean(&ratios.iter().map(|r| r.1).collect::<Vec<_>>());
    println!("\n   mean measured ratio {:.3}  (pi/2 = {:.4} at cosine 0)", m, PI / 2.0);
    println!(
        "   effective width of a liquid query against a {}-bit store: {:.0} bits",
        DIM_BITS,
        DIM_BITS as f64 * ratios[0].1
    );

    // Host-only scoring cost. Says nothing about the ESP32-S3: no popcount
    // instruction there, and an FPU that changes the balance. Measure on chip.
    let mut acc_q = [0f32; DIM_BITS];
    project_f32(&rand_vec(&mut s), 7, &mut acc_q);
    normalize(&mut acc_q);
    let items: Vec<Hv> = (0..512).map(|_| simhash_f32(&rand_vec(&mut s), 7)).collect();
    let probe = simhash_f32(&rand_vec(&mut s), 7);

    let reps = 200;
    let t0 = Instant::now();
    let mut sink = 0u64;
    for _ in 0..reps {
        for it in &items {
            sink = sink.wrapping_add(probe.hamming(it) as u64);
        }
    }
    let bin_ns = t0.elapsed().as_nanos() as f64 / (reps * items.len()) as f64;

    let t1 = Instant::now();
    let mut sink2 = 0f64;
    for _ in 0..reps {
        for it in &items {
            sink2 += asym_score(&acc_q, it);
        }
    }
    let liq_ns = t1.elapsed().as_nanos() as f64 / (reps * items.len()) as f64;

    println!("\n   host scoring cost per compare (NOT predictive of ESP32-S3):");
    println!("   binary hamming {:>8.1} ns", bin_ns);
    println!("   liquid asym    {:>8.1} ns   ({:.1}x)", liq_ns, liq_ns / bin_ns);
    println!("   (sinks {} {:.3})", sink % 7, sink2.abs() % 7.0);
}
