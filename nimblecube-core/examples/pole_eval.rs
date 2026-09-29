//! Pole (pivot) search: exact nearest neighbour from distances to a few fixed
//! poles, the LAESA scheme. Each item's distance to a pole is its "latitude"
//! along the diagonal through that pole; by the triangle inequality
//! `|d(q,p) - d(x,p)| <= d(q,x)`, so the largest latitude gap over all poles is a
//! lower bound on `d(q,x)`. Items are compared in lower-bound order and the
//! search stops when the bound reaches the best distance found: exact.
//!
//! Pole kinds: stored items (random, and farthest-first), bundles of k groups
//! snapped to bits (on the sphere), and the same bundles unsnapped (inside the
//! sphere, as L1 against fractional coordinates, which keeps the triangle
//! inequality). The exact centre is the control: every corner is equidistant
//! from it, so it can prune nothing. Host/std.
//!   cargo run --release --example pole_eval -- bench/certwork/*
//!
//! Real data comes from `bench/cert_prep.py`; the synthetic set is the
//! `net_tree_eval` generator (12000 items, 200 clusters, flip 200).
//!
//! Measured 2026-09-29, all variants 100% recall. Speedup in item compares
//! (pole compares included) over a linear scan, k=16 unless noted, 2000 queries:
//!
//! ```text
//!               centre  item rnd  item far  bundle snap  bundle in   cert (cert_real_eval)
//! synthetic       1.0x      1.2x      1.2x   3.5x (27.9x at 63) 1.0x  93x
//! annthyroid      2.2x    185x      185x       186x       145x       2.8x
//! mammography     4.6x    217x      346x    431x (k=7)    342x       2.9x
//! shuttle         5.6x    250x      157x       267x       216x       3.6x
//! satellite       1.0x      5.3x      4.3x       6.3x       4.5x      1.7x
//! pendigits       1.0x      7.1x      7.1x       8.5x       4.7x      4.1x
//! ```
//!
//! - The centre prunes nothing (every corner is 2048 from it). Its 2-6x is the
//!   stop at an exact duplicate (distance 0), which three datasets are full of;
//!   poles put duplicates first because their lower bound is exactly 0.
//! - Poles beat the certificate by 2-150x on real data and lose badly on the
//!   synthetic clusters: 200 randomly placed clusters leave every pole about
//!   2048 from most items (concentration), while the real rows have few
//!   features, so their distances spread. Snapped bundles work on synthetic
//!   only when there are enough of them to sit near many cluster centres.
//! - Inside (unsnapped) poles lost to snapped ones almost everywhere.
//! - Anomalies prune well except on pendigits and satellite.
//! - Not counted: the per-query lower-bound pass, n x k integer subtract/max.
//!   Cheap next to compares on the host, but at n=27k, k=16 it is ~440k ops
//!   and likely dominates on the chip; sorting items by one pole's distance and
//!   scanning a band (iDistance) is the known fix.
//! - Build is n x k compares against the certificate's n^2 (shuttle: 0.4M vs 374M).

use std::io::{BufRead, BufReader};

use nimblecube_core::encode::FeatureEncoder;
use nimblecube_core::hv::{Hv, DIM_BITS, WORDS};

const MAX_QUERIES: usize = 2000;

fn xs(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn read(path: &str) -> (usize, Vec<Vec<i32>>) {
    let mut lines = BufReader::new(std::fs::File::open(path).expect("open input")).lines();
    let head = lines.next().expect("header").expect("read");
    let cols: usize = head.split_whitespace().nth(1).expect("cols").parse().expect("cols");
    let rows = lines
        .map(|l| l.expect("read").split_whitespace().map(|v| v.parse().expect("int")).collect())
        .collect();
    (cols, rows)
}

fn encode<const CH: usize>(rows: &[Vec<i32>]) -> Vec<Hv> {
    let enc = FeatureEncoder::<CH, 16>::new(7, [(0, 1000); CH]);
    rows.iter().map(|r| enc.encode(&core::array::from_fn(|i| r[i]))).collect()
}

macro_rules! dispatch {
    ($cols:expr, $rows:expr, [$($n:literal),*]) => {
        match $cols {
            $($n => encode::<$n>($rows),)*
            c => panic!("add {c} columns to the dispatch list"),
        }
    };
}

/// A pole: a corner (bits) or an interior point (per-bit fraction in [0, 1]).
enum Pole {
    Corner(Hv),
    Inside(Vec<f32>),
}

impl Pole {
    /// Hamming for a corner, L1 for an interior point (equal on corners).
    fn dist(&self, x: &Hv) -> f32 {
        match self {
            Pole::Corner(p) => p.hamming(x) as f32,
            Pole::Inside(f) => (0..DIM_BITS)
                .map(|i| {
                    let b = ((x.0[i / 64] >> (i % 64)) & 1) as f32;
                    (b - f[i]).abs()
                })
                .sum(),
        }
    }
}

fn bit(h: &Hv, i: usize) -> u32 {
    ((h.0[i / 64] >> (i % 64)) & 1) as u32
}

/// k groups by one assignment pass to k random seed items; returns per-group bit counts and sizes.
fn groups(items: &[Hv], k: usize, s: &mut u64) -> (Vec<Vec<u32>>, Vec<u32>) {
    let seeds: Vec<&Hv> = (0..k).map(|_| &items[(xs(s) as usize) % items.len()]).collect();
    let mut counts = vec![vec![0u32; DIM_BITS]; k];
    let mut sizes = vec![0u32; k];
    for x in items {
        let g = (0..k).min_by_key(|&j| seeds[j].hamming(x)).unwrap();
        sizes[g] += 1;
        for (i, c) in counts[g].iter_mut().enumerate() {
            *c += bit(x, i);
        }
    }
    (counts, sizes)
}

fn poles(kind: &str, items: &[Hv], k: usize, seed: u64) -> Vec<Pole> {
    let mut s = seed;
    match kind {
        "centre" => vec![Pole::Inside(vec![0.5; DIM_BITS])],
        "item random" => (0..k).map(|_| Pole::Corner(items[(xs(&mut s) as usize) % items.len()].clone())).collect(),
        "item farthest" => {
            let mut chosen = vec![(xs(&mut s) as usize) % items.len()];
            let mut near: Vec<u32> = items.iter().map(|x| x.hamming(&items[chosen[0]])).collect();
            while chosen.len() < k {
                let next = (0..items.len()).max_by_key(|&i| near[i]).unwrap();
                chosen.push(next);
                for (i, x) in items.iter().enumerate() {
                    near[i] = near[i].min(x.hamming(&items[next]));
                }
            }
            chosen.into_iter().map(|i| Pole::Corner(items[i].clone())).collect()
        }
        "bundle snapped" | "bundle inside" => {
            let (counts, sizes) = groups(items, k, &mut s);
            counts
                .iter()
                .zip(&sizes)
                .filter(|(_, &m)| m > 0)
                .map(|(c, &m)| {
                    if kind == "bundle inside" {
                        Pole::Inside(c.iter().map(|&v| v as f32 / m as f32).collect())
                    } else {
                        let mut w = [0u64; WORDS];
                        for (i, &v) in c.iter().enumerate() {
                            // majority, ties to the seeded coin so even groups do not bias to 0
                            if 2 * v > m || (2 * v == m && xs(&mut s) & 1 == 1) {
                                w[i / 64] |= 1 << (i % 64);
                            }
                        }
                        Pole::Corner(Hv(w))
                    }
                })
                .collect()
        }
        _ => unreachable!(),
    }
}

fn evaluate(name: &str, items: &[Hv], queries: &[Hv], anom: &[bool]) {
    let n = items.len();
    let nq = queries.len();
    let n_anom = anom.iter().filter(|&&a| a).count();
    let truth: Vec<u32> = queries.iter().map(|q| items.iter().map(|h| h.hamming(q)).min().unwrap()).collect();
    println!("{}: {} stored, {} queries ({} anomalies)", name, n, nq, n_anom);
    println!(
        "  {:<15} {:>3} {:>7} {:>13} {:>13} {:>9} {:>10}",
        "poles", "k", "recall", "cmp normal", "cmp anomaly", "speedup", "build"
    );
    for kind in ["centre", "item random", "item farthest", "bundle snapped", "bundle inside"] {
        let ks: &[usize] = if kind == "centre" { &[1] } else { &[4, 16, 64] };
        for &k in ks {
            let ps = poles(kind, items, k, 0xB01E ^ k as u64);
            let table: Vec<Vec<f32>> = items.iter().map(|x| ps.iter().map(|p| p.dist(x)).collect()).collect();
            let build = match kind {
                "item farthest" => n * ps.len(),
                "bundle snapped" | "bundle inside" => 2 * n * ps.len(),
                _ => n * ps.len(),
            };
            let (mut hit, mut cmp_n, mut cmp_a) = (0usize, 0u64, 0u64);
            let mut order: Vec<(f32, u32)> = Vec::with_capacity(n);
            for (qi, q) in queries.iter().enumerate() {
                let dq: Vec<f32> = ps.iter().map(|p| p.dist(q)).collect();
                order.clear();
                for (i, row) in table.iter().enumerate() {
                    let lb = row.iter().zip(&dq).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                    order.push((lb, i as u32));
                }
                order.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let (mut best, mut cmp) = (u32::MAX, ps.len() as u64);
                for &(lb, i) in &order {
                    if lb >= best as f32 {
                        break;
                    }
                    cmp += 1;
                    best = best.min(items[i as usize].hamming(q));
                }
                hit += (best == truth[qi]) as usize;
                if anom[qi] {
                    cmp_a += cmp;
                } else {
                    cmp_n += cmp;
                }
            }
            let avg = (cmp_n + cmp_a) as f64 / nq as f64;
            let per = |c: u64, m: usize| if m == 0 { 0.0 } else { c as f64 / m as f64 };
            println!(
                "  {:<15} {:>3} {:>6.1}% {:>13.0} {:>13.0} {:>8.1}x {:>10}",
                kind,
                ps.len(),
                100.0 * hit as f64 / nq as f64,
                per(cmp_n, nq - n_anom),
                per(cmp_a, n_anom),
                n as f64 / avg,
                build
            );
        }
    }
    println!();
}

/// Keep at most MAX_QUERIES, sampled without replacement by a fixed seed.
fn sample(queries: Vec<Hv>, anom: Vec<bool>) -> (Vec<Hv>, Vec<bool>) {
    if queries.len() <= MAX_QUERIES {
        return (queries, anom);
    }
    let mut idx: Vec<usize> = (0..queries.len()).collect();
    let mut s: u64 = 0x5A3F_1E;
    for i in (1..idx.len()).rev() {
        idx.swap(i, (xs(&mut s) as usize) % (i + 1));
    }
    idx.truncate(MAX_QUERIES);
    idx.sort_unstable();
    (idx.iter().map(|&i| queries[i].clone()).collect(), idx.iter().map(|&i| anom[i]).collect())
}

fn main() {
    // synthetic set, identical to net_tree_eval
    let mut s: u64 = 0x000A_11CE_5EED;
    let rand_hv = |s: &mut u64| {
        let mut w = [0u64; WORDS];
        for x in w.iter_mut() {
            *x = xs(s);
        }
        Hv(w)
    };
    let noisy = |base: &Hv, s: &mut u64| {
        let mut h = base.clone();
        for _ in 0..200 {
            let b = (xs(s) as usize) % DIM_BITS;
            h.0[b / 64] ^= 1u64 << (b % 64);
        }
        h
    };
    let bases: Vec<Hv> = (0..200).map(|_| rand_hv(&mut s)).collect();
    let items: Vec<Hv> = (0..12000).map(|i| noisy(&bases[i % 200], &mut s)).collect();
    let queries: Vec<Hv> = (0..500)
        .map(|_| {
            let b = (xs(&mut s) as usize) % 200;
            noisy(&bases[b], &mut s)
        })
        .collect();
    evaluate("synthetic", &items, &queries, &[false; 500]);

    for dir in std::env::args().skip(1) {
        let (cols, train) = read(&format!("{dir}/train.txt"));
        let (_, test) = read(&format!("{dir}/test.txt"));
        let anom: Vec<bool> = test.iter().map(|r| r[cols] == 1).collect();
        let items = dispatch!(cols, &train, [6, 9, 16, 36]);
        let queries = dispatch!(cols, &test, [6, 9, 16, 36]);
        let (queries, anom) = sample(queries, anom);
        let name = dir.trim_end_matches('/').rsplit('/').next().unwrap().to_string();
        evaluate(&name, &items, &queries, &anom);
    }
}
