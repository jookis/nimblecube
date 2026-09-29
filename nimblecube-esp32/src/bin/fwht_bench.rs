//! Structured projection for the SimHash encoder: does a Walsh-Hadamard
//! butterfly run fast enough on the chip to make embedding encoding viable?
//!
//! `simhash_f32` projects onto 4096 independent random +-1 directions (cube
//! diagonals), 4096 x 768 multiply-adds. The structured version projects onto
//! Hadamard rows instead: per 1024-bit block, `stages` rounds of random sign
//! flip then a 1024-point fast Walsh-Hadamard transform, keeping the signs.
//! Four blocks give 4096 bits. The open worry was the butterfly's strided
//! memory access, so the bare transform is timed alongside the full encode.
//!
//! Sanity check, not validation (the angle law is settled in the literature):
//! mean Hamming against the theta/pi * 4096 that SimHash predicts, over near and
//! unrelated pairs, from both encoders.
//!   cd nimblecube-esp32 && cargo run --release --bin fwht_bench
//!
//! Measured on chip 2026-09-29, D=768, uniform random inputs:
//!
//! ```text
//! simhash_f32 (4096 x 768 dense) : 341.8 ms
//! bare FWHT, 1024 points         : 0.49 ms (f32 and i32 alike)
//! structured encode, 3 stages    : 7.8 ms  (43x)
//! structured encode, 2 stages    : 5.4 ms  (63x)
//!
//! mean hamming, 16 pairs each      near  unrelated
//! expected theta/pi                 598       2061
//! simhash_f32                       609       2057
//! structured 2 stages               598       2054
//! structured 3 stages               601       2061
//! ```
//!
//! Strided butterfly access costs nothing visible at 4 KB. Inputs here are not
//! spiky, so 2 vs 3 stages is still open for real data.

#![no_std]
#![no_main]

use core::hint::black_box;
use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::delay::Delay;
use esp_hal::main;
use esp_hal::time::Instant;
use esp_println::println;

use nimblecube_core::hv::{Hv, DIM_BITS, WORDS};
use nimblecube_core::simhash::simhash_f32;

esp_bootloader_esp_idf::esp_app_desc!();

const D: usize = 768; // a typical embedding width
const N: usize = 1024; // transform size, one output block
const BLOCKS: usize = DIM_BITS / N; // 4
const MAX_STAGES: usize = 3;
const MS: u64 = 3; // simhash reps (slow)
const MB: u64 = 1000; // bare butterfly reps
const ME: u64 = 50; // structured encode reps

fn xs(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn fwht_f32(x: &mut [f32; N]) {
    let mut h = 1;
    while h < N {
        let mut i = 0;
        while i < N {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
}

fn fwht_i32(x: &mut [i32; N]) {
    let mut h = 1;
    while h < N {
        let mut i = 0;
        while i < N {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a.wrapping_add(b);
                x[j + h] = a.wrapping_sub(b);
            }
            i += 2 * h;
        }
        h *= 2;
    }
}

/// Random sign masks, one 1024-bit mask per (block, stage).
struct Signs([[[u32; N / 32]; MAX_STAGES]; BLOCKS]);

impl Signs {
    fn new(seed: u64) -> Self {
        let mut s = seed;
        let mut m = [[[0u32; N / 32]; MAX_STAGES]; BLOCKS];
        for b in m.iter_mut() {
            for st in b.iter_mut() {
                for w in st.iter_mut() {
                    *w = xs(&mut s) as u32;
                }
            }
        }
        Signs(m)
    }
}

/// Structured SimHash: per block, `stages` x (sign flip, FWHT), then keep signs.
fn structured_encode(v: &[f32; D], signs: &Signs, stages: usize) -> Hv {
    let mut w = [0u64; WORDS];
    for (b, bs) in signs.0.iter().enumerate() {
        let mut x = [0f32; N];
        x[..D].copy_from_slice(v);
        for mask in bs.iter().take(stages) {
            for (i, xi) in x.iter_mut().enumerate() {
                let flip = (mask[i / 32] >> (i % 32)) & 1;
                *xi = f32::from_bits(xi.to_bits() ^ (flip << 31));
            }
            fwht_f32(&mut x);
        }
        for (i, &xi) in x.iter().enumerate() {
            if xi > 0.0 {
                let k = b * N + i;
                w[k / 64] |= 1u64 << (k % 64);
            }
        }
    }
    Hv(w)
}

/// `acos` is not in core; a polynomial approximation is enough for the sanity check.
fn acos(x: f32) -> f32 {
    // Abramowitz-Stegun 4.4.45, |error| < 7e-5 rad
    let a = if x < 0.0 { -x } else { x };
    let r = (1.5707288 - 0.2121144 * a + 0.0742610 * a * a - 0.0187293 * a * a * a) * sqrt(1.0 - a);
    if x < 0.0 {
        core::f32::consts::PI - r
    } else {
        r
    }
}

fn sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    // bit-level initial guess, then Newton; a guess of `x` itself needs ~30 steps at 1e16
    let mut g = f32::from_bits((x.to_bits() >> 1) + 0x1fbd_1df5);
    for _ in 0..6 {
        g = 0.5 * (g + x / g);
    }
    g
}

fn cosine(a: &[f32; D], b: &[f32; D]) -> f32 {
    let (mut ab, mut aa, mut bb) = (0f32, 0f32, 0f32);
    for i in 0..D {
        ab += a[i] * b[i];
        aa += a[i] * a[i];
        bb += b[i] * b[i];
    }
    ab / sqrt(aa * bb)
}

#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let _p = esp_hal::init(config);
    let delay = Delay::new();
    // Give the serial reader time to attach.
    delay.delay_millis(8000);
    println!("fwht_bench: start");

    let mut st: u64 = 0x1234_5678_9abc_def1;
    let rnd = |s: &mut u64| (xs(s) % 2001) as f32 - 1000.0;
    let mut v = [0f32; D];
    let mut near = [0f32; D];
    let mut other = [0f32; D];
    for i in 0..D {
        v[i] = rnd(&mut st);
        near[i] = v[i] + 0.5 * rnd(&mut st);
        other[i] = rnd(&mut st);
    }
    let signs = Signs::new(0x5157_1A6E);

    let t = Instant::now();
    let mut acc = 0u32;
    for _ in 0..MS {
        acc = acc.wrapping_add(simhash_f32(black_box(&v), 7).0[0] as u32);
    }
    let simhash_us = t.elapsed().as_micros() / MS;

    let mut xf = [0f32; N];
    let mut xi = [0i32; N];
    for i in 0..D {
        xf[i] = v[i];
        xi[i] = v[i] as i32;
    }
    let t = Instant::now();
    for _ in 0..MB {
        fwht_f32(black_box(&mut xf));
        // keep values bounded: one transform scales by sqrt(N)=32 in norm
        for x in xf.iter_mut() {
            *x *= 1.0 / 32.0;
        }
    }
    let bf_f32_ns = t.elapsed().as_micros() * 1000 / MB;
    let t = Instant::now();
    for _ in 0..MB {
        for x in xf.iter_mut() {
            *x *= 1.0 / 32.0;
        }
    }
    let scale_ns = t.elapsed().as_micros() * 1000 / MB;

    let t = Instant::now();
    for _ in 0..MB {
        fwht_i32(black_box(&mut xi));
    }
    let bf_i32_ns = t.elapsed().as_micros() * 1000 / MB;

    let mut enc_us = [0u64; MAX_STAGES + 1];
    for stages in [2usize, 3] {
        let t = Instant::now();
        for _ in 0..ME {
            acc = acc.wrapping_add(structured_encode(black_box(&v), &signs, stages).0[0] as u32);
        }
        enc_us[stages] = t.elapsed().as_micros() / ME;
    }
    black_box((acc, &xf, &xi));

    // Sanity: mean Hamming vs theta/pi * 4096 over PAIRS near and PAIRS unrelated pairs.
    // Columns: expected, simhash_f32, structured 2 stages, structured 3 stages.
    const PAIRS: u32 = 16;
    let expect = |a: &[f32; D], b: &[f32; D]| acos(cosine(a, b)) / core::f32::consts::PI * DIM_BITS as f32;
    let mut sums = [[0f32; 4]; 2]; // [near, unrelated]
    let mut ones = [0u32; MAX_STAGES + 1];
    for _ in 0..PAIRS {
        let mut a = [0f32; D];
        let mut b = [0f32; D];
        let mut c = [0f32; D];
        for i in 0..D {
            a[i] = rnd(&mut st);
            b[i] = a[i] + 0.5 * rnd(&mut st);
            c[i] = rnd(&mut st);
        }
        for (k, other) in [&b, &c].into_iter().enumerate() {
            sums[k][0] += expect(&a, other);
            sums[k][1] += simhash_f32(&a, 7).hamming(&simhash_f32(other, 7)) as f32;
            for stages in [2usize, 3] {
                let ea = structured_encode(&a, &signs, stages);
                sums[k][stages] += ea.hamming(&structured_encode(other, &signs, stages)) as f32;
                if k == 0 {
                    ones[stages] += ea.0.iter().map(|w| w.count_ones()).sum::<u32>();
                }
            }
        }
    }
    let mean = |x: f32| (x / PAIRS as f32) as u32;

    loop {
        println!("fwht_bench (ESP32-S3 @ 240 MHz)  D={} N={} blocks={}", D, N, BLOCKS);
        println!("simhash_f32 (4096 x {} dense)   : {} us", D, simhash_us);
        println!("bare FWHT f32, 1024 points     : {} ns  (+ {} ns rescale, subtracted: {} ns)", bf_f32_ns, scale_ns, bf_f32_ns.saturating_sub(scale_ns));
        println!("bare FWHT i32, 1024 points     : {} ns", bf_i32_ns);
        println!("structured encode, 2 stages    : {} us  ({}x vs simhash_f32)", enc_us[2], simhash_us / enc_us[2].max(1));
        println!("structured encode, 3 stages    : {} us  ({}x vs simhash_f32)", enc_us[3], simhash_us / enc_us[3].max(1));
        println!("sanity, mean hamming over {} pairs each:", PAIRS);
        println!("  expected theta/pi : near {}  unrelated {}", mean(sums[0][0]), mean(sums[1][0]));
        println!("  simhash_f32       : near {}  unrelated {}", mean(sums[0][1]), mean(sums[1][1]));
        for stages in [2usize, 3] {
            println!(
                "  structured {} stg  : near {}  unrelated {}  ones {}",
                stages,
                mean(sums[0][stages]),
                mean(sums[1][stages]),
                ones[stages] / PAIRS
            );
        }
        println!("--");
        delay.delay_millis(3000);
    }
}
