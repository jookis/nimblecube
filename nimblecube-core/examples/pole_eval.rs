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
//!   cargo run --release --example pole_eval -- combined bench/certwork/*
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
//!
//! **Combined mode** (flat PM-tree: net tree cells with pole rings and covering
//! radii, 16 snapped-bundle poles), measured 2026-09-29, all 100% recall.
//! Speedup in compares; bound ops = integer pole/ring work per query:
//!
//! ```text
//!                  synthetic annthyroid mammography satellite pendigits  shuttle
//! poles only           3.5x      186x       432x      6.3x      8.5x    267x
//! combined            55.8x       52x       118x      5.2x      8.7x    106x
//! combined + hash     41.1x      3.2x       3.0x      3.3x      6.5x    8.6x
//! rings + poles        3.5x      113x       183x      5.8x      7.8x    248x
//! bound ops, shuttle: poles only 356k, combined 29k, rings + poles 73k
//! ```
//!
//! Centre compares carry the clustered case (synthetic 3.5x to 55.8x) and waste
//! compares on dense real data, where cells are wide. Rings + poles nearly match
//! poles only on real data with 3-12x fewer bound ops. The hash start hurts on
//! dense data: buckets fill with near-duplicates. Next: compare a cell's centre
//! only when the cell is tight.

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

const CAP: usize = 32; // net tree cell cap
const RADIUS: u32 = (DIM_BITS / 4) as u32; // net tree join radius
const K: usize = 16; // global poles in the combined index
const M: usize = 256; // cross-polytope width

/// Greedy net tree cells (as in `net_tree_eval`): members and snapped bundle centres.
fn cells(items: &[Hv]) -> (Vec<Hv>, Vec<Vec<u32>>) {
    let (mut nets, mut members): (Vec<Hv>, Vec<Vec<u32>>) = (Vec::new(), Vec::new());
    for (id, h) in items.iter().enumerate() {
        let mut best = usize::MAX;
        let mut bd = u32::MAX;
        for (c, net) in nets.iter().enumerate() {
            if members[c].len() >= CAP {
                continue;
            }
            let d = net.hamming(h);
            if d < bd {
                bd = d;
                best = c;
            }
        }
        if best == usize::MAX || bd > RADIUS {
            nets.push(h.clone());
            members.push(vec![id as u32]);
        } else {
            members[best].push(id as u32);
            let mems: Vec<Hv> = members[best].iter().map(|&i| items[i as usize].clone()).collect();
            nets[best] = Hv::bundle(&mems);
        }
    }
    (nets, members)
}

/// Cross-polytope hash, same construction and seed as `net_tree_eval` (m=256).
struct CrossPolytope {
    bits: Vec<usize>,
    signs: Vec<i32>,
}

impl CrossPolytope {
    fn new(s: &mut u64) -> Self {
        let mut pool: Vec<usize> = (0..DIM_BITS).collect();
        for i in (1..DIM_BITS).rev() {
            pool.swap(i, (xs(s) as usize) % (i + 1));
        }
        pool.truncate(M);
        let signs = (0..M).map(|_| if xs(s) & 1 == 0 { 1 } else { -1 }).collect();
        CrossPolytope { bits: pool, signs }
    }

    fn hash(&self, h: &Hv) -> usize {
        let mut y: Vec<i32> =
            self.bits.iter().zip(&self.signs).map(|(&b, &sg)| if bit(h, b) == 1 { sg } else { -sg }).collect();
        let mut len = 1;
        while len < M {
            for i in (0..M).step_by(2 * len) {
                for j in i..i + len {
                    let (a, b) = (y[j], y[j + len]);
                    y[j] = a + b;
                    y[j + len] = a - b;
                }
            }
            len *= 2;
        }
        let (mut bi, mut bv) = (0usize, -1i32);
        for (i, &v) in y.iter().enumerate() {
            if v.abs() > bv {
                bv = v.abs();
                bi = 2 * i + (v < 0) as usize;
            }
        }
        bi
    }

    /// Compare-equivalents: m*log2(m) add/subs plus m sign loads over 64 word ops.
    fn cost() -> f64 {
        let m = M as f64;
        (m * m.log2() + m) / WORDS as f64
    }
}

/// Poles only against the combined index (a flat PM-tree): per cell a ring of
/// member distances to each pole and a covering radius, per item its distance to
/// its cell centre. A cell is skipped when its ring bound or `d(q, centre) - radius`
/// reaches the best distance; an item is skipped when its parent-distance or pole
/// bound does. "bound ops" counts the integer work: k per ring or pole bound.
fn evaluate_combined(name: &str, items: &[Hv], queries: &[Hv], anom: &[bool]) {
    let n = items.len();
    let nq = queries.len();
    let n_anom = anom.iter().filter(|&&a| a).count();
    let truth: Vec<u32> = queries.iter().map(|q| items.iter().map(|h| h.hamming(q)).min().unwrap()).collect();

    let ps: Vec<Hv> = poles("bundle snapped", items, K, 0xB01E ^ K as u64)
        .into_iter()
        .map(|p| match p {
            Pole::Corner(h) => h,
            Pole::Inside(_) => unreachable!(),
        })
        .collect();
    let k = ps.len();
    let table: Vec<Vec<u32>> = items.iter().map(|x| ps.iter().map(|p| p.hamming(x)).collect()).collect();
    let (nets, members) = cells(items);
    let parent: Vec<Vec<u32>> =
        members.iter().zip(&nets).map(|(m, c)| m.iter().map(|&i| items[i as usize].hamming(c)).collect()).collect();
    let radius: Vec<u32> = parent.iter().map(|p| *p.iter().max().unwrap()).collect();
    let rings: Vec<Vec<(u32, u32)>> = members
        .iter()
        .map(|m| {
            (0..k)
                .map(|j| {
                    let it = m.iter().map(|&i| table[i as usize][j]);
                    (it.clone().min().unwrap(), it.max().unwrap())
                })
                .collect()
        })
        .collect();
    let mut hs: u64 = 0xC0FF_EE11;
    let cp = CrossPolytope::new(&mut hs);
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); 2 * M];
    for (i, h) in items.iter().enumerate() {
        buckets[cp.hash(h)].push(i as u32);
    }

    println!("{}: {} stored, {} queries ({} anomalies), {} cells, {} poles", name, n, nq, n_anom, nets.len(), k);
    println!(
        "  {:<18} {:>7} {:>11} {:>11} {:>9} {:>11}",
        "variant", "recall", "cmp normal", "cmp anomaly", "speedup", "bound ops"
    );
    for variant in ["poles only", "combined", "combined + hash", "rings + poles"] {
        let (mut hit, mut cmp_n, mut cmp_a, mut ops) = (0usize, 0f64, 0f64, 0u64);
        for (qi, q) in queries.iter().enumerate() {
            let dq: Vec<u32> = ps.iter().map(|p| p.hamming(q)).collect();
            let mut cmp = k as f64;
            let mut best = u32::MAX;
            let pole_lb = |x: usize| (0..k).map(|j| table[x][j].abs_diff(dq[j])).max().unwrap();
            if variant == "poles only" {
                let mut order: Vec<(u32, u32)> = (0..n).map(|i| (pole_lb(i), i as u32)).collect();
                ops += (n * k) as u64;
                order.sort_unstable();
                for &(lb, i) in &order {
                    if lb >= best {
                        break;
                    }
                    cmp += 1.0;
                    best = best.min(items[i as usize].hamming(q));
                }
            } else {
                if variant == "combined + hash" {
                    cmp += CrossPolytope::cost();
                    for &i in &buckets[cp.hash(q)] {
                        cmp += 1.0;
                        best = best.min(items[i as usize].hamming(q));
                    }
                }
                let mut order: Vec<(u32, usize)> = rings
                    .iter()
                    .enumerate()
                    .map(|(c, r)| {
                        let lb = (0..k)
                            .map(|j| dq[j].saturating_sub(r[j].1).max(r[j].0.saturating_sub(dq[j])))
                            .max()
                            .unwrap();
                        (lb, c)
                    })
                    .collect();
                ops += (rings.len() * k) as u64;
                order.sort_unstable();
                // "rings + poles" skips the cell-centre compare and its two bounds
                let use_centre = variant != "rings + poles";
                for &(lb, c) in &order {
                    if lb >= best {
                        break;
                    }
                    let dc = if use_centre {
                        cmp += 1.0;
                        nets[c].hamming(q)
                    } else {
                        0
                    };
                    if use_centre && dc.saturating_sub(radius[c]) >= best {
                        continue;
                    }
                    for (m, &x) in members[c].iter().enumerate() {
                        if use_centre {
                            ops += 1;
                            if dc.abs_diff(parent[c][m]) >= best {
                                continue;
                            }
                        }
                        ops += k as u64;
                        if pole_lb(x as usize) >= best {
                            continue;
                        }
                        cmp += 1.0;
                        best = best.min(items[x as usize].hamming(q));
                    }
                }
            }
            hit += (best == truth[qi]) as usize;
            if anom[qi] {
                cmp_a += cmp;
            } else {
                cmp_n += cmp;
            }
        }
        let avg = (cmp_n + cmp_a) / nq as f64;
        let per = |c: f64, m: usize| if m == 0 { 0.0 } else { c / m as f64 };
        println!(
            "  {:<18} {:>6.1}% {:>11.0} {:>11.0} {:>8.1}x {:>11.0}",
            variant,
            100.0 * hit as f64 / nq as f64,
            per(cmp_n, nq - n_anom),
            per(cmp_a, n_anom),
            n as f64 / avg,
            ops as f64 / nq as f64
        );
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
    let args: Vec<String> = std::env::args().skip(1).collect();
    let combined = args.first().is_some_and(|a| a == "combined");
    let run: fn(&str, &[Hv], &[Hv], &[bool]) = if combined { evaluate_combined } else { evaluate };
    let dirs = if combined { &args[1..] } else { &args[..] };

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
    run("synthetic", &items, &queries, &[false; 500]);

    for dir in dirs {
        let (cols, train) = read(&format!("{dir}/train.txt"));
        let (_, test) = read(&format!("{dir}/test.txt"));
        let anom: Vec<bool> = test.iter().map(|r| r[cols] == 1).collect();
        let items = dispatch!(cols, &train, [6, 9, 16, 36]);
        let queries = dispatch!(cols, &test, [6, 9, 16, 36]);
        let (queries, anom) = sample(queries, anom);
        let name = dir.trim_end_matches('/').rsplit('/').next().unwrap().to_string();
        run(&name, &items, &queries, &anom);
    }
}
