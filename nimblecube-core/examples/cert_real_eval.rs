//! The certified search index on real data: ADBench rows encoded by the real
//! `FeatureEncoder`, the training normals stored, held-out normals and anomalies
//! as queries. Checks whether the synthetic `net_tree_eval` result survives data
//! that is dense, duplicated and unevenly spread. Host/std.
//!   python3 bench/cert_prep.py 2_annthyroid 23_mammography 28_pendigits 30_satellite 32_shuttle
//!   cargo run --release --example cert_real_eval -- bench/certwork/*
//!
//! Neighbour lists keep each item's `k` nearest within `RADIUS`. A truncated list
//! is still complete up to one below the (k+1)-th distance, so each item carries
//! its own certified radius and the certificate stays exact.
//!
//! Measured 2026-09-29, seed 0 split, 16 levels. Best variant (two hash tables,
//! lazy, k=1024), speedup in compares over a linear scan, all at 100% recall:
//!
//! ```text
//! dataset      stored  within 1024 (p50)  certified normal / anomaly  speedup (k=64)
//! annthyroid     3999               3971        98.2% / 83.0%         2.8x  (2.2x)
//! mammography    6553               5901        99.8% / 87.3%         2.9x  (2.7x)
//! shuttle       27351              27344        99.9% /  2.5%         3.6x  (3.6x)
//! satellite      2639               1509        88.3% / 44.4%         1.7x  (0.8x)
//! pendigits      4028                409        89.2% /  0.0%         4.1x  (1.6x)
//! synthetic     12000                 59       100%   (no anomalies)  93x
//! ```
//!
//! Real encoded rows are dense: most items lie within the list radius of most
//! others, so lists truncate and the certified radius shrinks. Anomalies rarely
//! certify and fall back to a full scan. Two bugs found on the way, both from
//! exact duplicates: an unsigned wrap gave a truncated list a huge radius, and
//! the lazy table skipped a first bucket whose failure was truncation, not distance.

use std::collections::BinaryHeap;
use std::io::{BufRead, BufReader};

use nimblecube_core::encode::FeatureEncoder;
use nimblecube_core::hv::{Hv, DIM_BITS, WORDS};

const RADIUS: u32 = (DIM_BITS / 4) as u32; // join radius and list radius
const KMAX: usize = 1024; // neighbours kept per item at build
const CAP: usize = 32; // net tree cell cap
const M: usize = 256; // cross-polytope width

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

fn within(a: &Hv, b: &Hv, limit: u32) -> Option<u32> {
    let mut d = 0u32;
    for w in 0..WORDS {
        d += (a.0[w] ^ b.0[w]).count_ones();
        if d > limit {
            return None;
        }
    }
    Some(d)
}

struct NetTree {
    nets: Vec<Hv>,
    members: Vec<Vec<u32>>,
}

impl NetTree {
    fn build(items: &[Hv]) -> Self {
        let mut t = NetTree { nets: Vec::new(), members: Vec::new() };
        for (id, h) in items.iter().enumerate() {
            let mut best = usize::MAX;
            let mut bd = u32::MAX;
            for (c, net) in t.nets.iter().enumerate() {
                if t.members[c].len() >= CAP {
                    continue;
                }
                let d = net.hamming(h);
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            if best == usize::MAX || bd > RADIUS {
                t.nets.push(h.clone());
                t.members.push(vec![id as u32]);
            } else {
                t.members[best].push(id as u32);
                let mems: Vec<Hv> = t.members[best].iter().map(|&i| items[i as usize].clone()).collect();
                t.nets[best] = Hv::bundle(&mems);
            }
        }
        t
    }

    /// Probe the `nprobe` nearest cells. Returns (distance, id, first cell, compares).
    fn query(&self, items: &[Hv], q: &Hv, nprobe: usize) -> (u32, u32, usize, u64) {
        let mut nd: Vec<(u32, usize)> = self.nets.iter().enumerate().map(|(c, n)| (n.hamming(q), c)).collect();
        nd.sort_unstable();
        let mut cmp = self.nets.len() as u64;
        let (mut bd, mut bi) = (u32::MAX, 0u32);
        for &(_, c) in nd.iter().take(nprobe) {
            for &id in &self.members[c] {
                cmp += 1;
                let d = items[id as usize].hamming(q);
                if d < bd {
                    bd = d;
                    bi = id;
                }
            }
        }
        (bd, bi, nd[0].1, cmp)
    }
}

/// Each item's nearest neighbours within RADIUS, sorted, at most KMAX, plus how many
/// lay within RADIUS in total.
struct Lists {
    near: Vec<Vec<(u32, u32)>>,
    total: Vec<usize>,
}

impl Lists {
    fn build(items: &[Hv]) -> Self {
        let n = items.len();
        let mut heaps: Vec<BinaryHeap<(u32, u32)>> = (0..n).map(|_| BinaryHeap::new()).collect();
        let mut total = vec![0usize; n];
        let keep = |h: &mut BinaryHeap<(u32, u32)>, e: (u32, u32)| {
            if h.len() < KMAX {
                h.push(e);
            } else if e < *h.peek().unwrap() {
                h.pop();
                h.push(e);
            }
        };
        for i in 0..n {
            for j in (i + 1)..n {
                if let Some(d) = within(&items[i], &items[j], RADIUS) {
                    total[i] += 1;
                    total[j] += 1;
                    keep(&mut heaps[i], (d, j as u32));
                    keep(&mut heaps[j], (d, i as u32));
                }
            }
        }
        let near = heaps.into_iter().map(|h| h.into_sorted_vec()).collect();
        Lists { near, total }
    }

    /// Radius up to which item `c`'s first `k` entries are complete, or `None` when
    /// the list is truncated among exact duplicates and certifies nothing.
    fn radius(&self, c: usize, k: usize) -> Option<u32> {
        let l = &self.near[c];
        if self.total[c] <= k {
            Some(RADIUS)
        } else if l.len() > k {
            l[k].0.checked_sub(1)
        } else {
            // k >= KMAX and truncated at build: ties at the last kept distance may be missing
            l[l.len() - 1].0.checked_sub(1)
        }
    }

    /// Exact nearest from candidate `c` at `r` using `k` slots, or `None` if the
    /// certified radius is too small. Returns (distance, compares).
    fn certify(&self, items: &[Hv], q: &Hv, c: u32, r: u32, k: usize, skip: &[u32]) -> Option<(u32, u64)> {
        if r == 0 {
            return Some((0, 0));
        }
        match self.radius(c as usize, k) {
            Some(rad) if 2 * r - 1 <= rad => {}
            _ => return None,
        }
        let (mut bd, mut cmp) = (r, 0u64);
        for &(dc, id) in self.near[c as usize].iter().take(k) {
            if dc >= 2 * r {
                break;
            }
            if skip.contains(&id) {
                continue;
            }
            cmp += 1;
            let d = items[id as usize].hamming(q);
            if d < bd {
                bd = d;
            }
        }
        Some((bd, cmp))
    }
}

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
        let mut y: Vec<i32> = self
            .bits
            .iter()
            .zip(&self.signs)
            .map(|(&b, &sg)| if (h.0[b / 64] >> (b % 64)) & 1 == 1 { sg } else { -sg })
            .collect();
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

fn pct(v: &mut [u32], p: f64) -> u32 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[((v.len() - 1) as f64 * p) as usize]
}

#[derive(Default)]
struct Tally {
    hit: usize,
    scanned: usize,
    cert_normal: usize,
    cert_anom: usize,
    cmp: f64,
}

fn evaluate(name: &str, items: &[Hv], queries: &[Hv], anom: &[bool]) {
    let n = items.len();
    let nq = queries.len();
    let n_anom = anom.iter().filter(|&&a| a).count();
    println!("{}: {} stored, {} queries ({} anomalies)", name, n, nq, n_anom);

    let truth: Vec<u32> = queries.iter().map(|q| items.iter().map(|h| h.hamming(q)).min().unwrap()).collect();
    let mut s: u64 = 0x5EED_0F_DA7A;
    let mut pair: Vec<u32> = (0..20000)
        .map(|_| {
            let (i, j) = ((xs(&mut s) as usize) % n, (xs(&mut s) as usize) % n);
            items[i].hamming(&items[j])
        })
        .collect();
    let mut qn: Vec<u32> = (0..nq).filter(|&i| !anom[i]).map(|i| truth[i]).collect();
    let mut qa: Vec<u32> = (0..nq).filter(|&i| anom[i]).map(|i| truth[i]).collect();
    println!(
        "  random pair distance  : p1 {} p50 {} p99 {}",
        pct(&mut pair, 0.01),
        pct(&mut pair, 0.5),
        pct(&mut pair, 0.99)
    );
    println!(
        "  query nearest distance: normal p50 {} p99 {}, anomaly p50 {} p99 {}",
        pct(&mut qn, 0.5),
        pct(&mut qn, 0.99),
        pct(&mut qa, 0.5),
        pct(&mut qa, 0.99)
    );

    let t = std::time::Instant::now();
    let lists = Lists::build(items);
    let tree = NetTree::build(items);
    let mut hs: u64 = 0xC0FF_EE11;
    let cps = [CrossPolytope::new(&mut hs), CrossPolytope::new(&mut hs)];
    let buckets: Vec<Vec<Vec<u32>>> = cps
        .iter()
        .map(|cp| {
            let mut b: Vec<Vec<u32>> = vec![Vec::new(); 2 * M];
            for (i, h) in items.iter().enumerate() {
                b[cp.hash(h)].push(i as u32);
            }
            b
        })
        .collect();
    let mut within_r: Vec<u32> = lists.total.iter().map(|&t| t as u32).collect();
    let full64 = (0..n).filter(|&c| lists.radius(c, 64) == Some(RADIUS)).count();
    let full1k = (0..n).filter(|&c| lists.radius(c, KMAX) == Some(RADIUS)).count();
    println!(
        "  items within {} of an item: p50 {} p99 {} max {}; list complete to {}: {:.0}% at k=64, {:.0}% at k={}  (build {:.1} s)",
        RADIUS,
        pct(&mut within_r, 0.5),
        pct(&mut within_r, 0.99),
        pct(&mut within_r, 1.0),
        RADIUS,
        100.0 * full64 as f64 / n as f64,
        100.0 * full1k as f64 / n as f64,
        KMAX,
        t.elapsed().as_secs_f64()
    );
    println!(
        "  cells={}  {:<22} {:>5} {:>8} {:>8} {:>9} {:>9} {:>10} {:>8}",
        tree.nets.len(),
        "variant",
        "k",
        "recall",
        "scanned",
        "cert nrm",
        "cert anm",
        "compares",
        "speedup"
    );

    let report = |label: &str, k: &str, t: &Tally| {
        let avg = t.cmp / nq as f64;
        println!(
            "  {:>9}  {:<22} {:>5} {:>7.1}% {:>7.1}% {:>8.1}% {:>8.1}% {:>10.0} {:>7.1}x",
            "",
            label,
            k,
            100.0 * t.hit as f64 / nq as f64,
            100.0 * t.scanned as f64 / nq as f64,
            100.0 * t.cert_normal as f64 / (nq - n_anom).max(1) as f64,
            100.0 * t.cert_anom as f64 / n_anom.max(1) as f64,
            avg,
            n as f64 / avg
        );
    };

    for nprobe in [1usize, 2] {
        let mut t = Tally::default();
        for (qi, q) in queries.iter().enumerate() {
            let (d, _, _, cmp) = tree.query(items, q, nprobe);
            t.cmp += cmp as f64;
            t.hit += (d == truth[qi]) as usize;
        }
        report(if nprobe == 1 { "cells nprobe=1" } else { "cells nprobe=2" }, "-", &t);
    }

    // Certified cell path; falls back to a full scan. Returns (distance, compares, certified).
    let cells_cert = |q: &Hv, qi: usize, k: usize| -> (u32, f64, bool) {
        let (r, c, cell, cmp) = tree.query(items, q, 1);
        match lists.certify(items, q, c, r, k, &tree.members[cell]) {
            Some((d, k2)) => (d, (cmp + k2) as f64, true),
            None => (truth[qi], (cmp as usize + n) as f64, false),
        }
    };
    // Best of a bucket (skipping ids already compared), then certify. The result is
    // exact over everything except `prior`, so the caller takes the min with the
    // prior bucket's best. Returns (certified distance, compares, bucket best).
    let bucket_cert = |q: &Hv, b: &[u32], prior: &[u32], k: usize| -> (Option<u32>, f64, u32) {
        let (mut r, mut c, mut cmp) = (u32::MAX, 0u32, 0f64);
        for &id in b {
            if prior.contains(&id) {
                continue;
            }
            cmp += 1.0;
            let d = items[id as usize].hamming(q);
            if d < r {
                r = d;
                c = id;
            }
        }
        if r == u32::MAX {
            return (None, cmp, r);
        }
        let seen: Vec<u32> = b.iter().chain(prior).copied().collect();
        match lists.certify(items, q, c, r, k, &seen) {
            Some((d, k2)) => (Some(d), cmp + k2 as f64, r),
            None => (None, cmp, r),
        }
    };

    for k in [64usize, KMAX] {
        for variant in ["cells -> cert", "hash -> cert/cells", "hash x2 lazy -> cert"] {
            let mut t = Tally::default();
            for (qi, q) in queries.iter().enumerate() {
                let (d, cmp, certified) = if variant == "cells -> cert" {
                    cells_cert(q, qi, k)
                } else {
                    let b0 = &buckets[0][cps[0].hash(q)];
                    let (mut out, mut cmp, r0) = bucket_cert(q, b0, &[], k);
                    cmp += CrossPolytope::cost();
                    if out.is_none() && variant.starts_with("hash x2") {
                        let b1 = &buckets[1][cps[1].hash(q)];
                        let (o, c2, _) = bucket_cert(q, b1, b0, k);
                        out = o.map(|d| d.min(r0));
                        cmp += c2 + CrossPolytope::cost();
                    }
                    match out {
                        Some(d) => (d, cmp, true),
                        None => {
                            let (d, c2, ok) = cells_cert(q, qi, k);
                            (d, cmp + c2, ok)
                        }
                    }
                };
                t.cmp += cmp;
                t.hit += (d == truth[qi]) as usize;
                if certified {
                    if anom[qi] {
                        t.cert_anom += 1;
                    } else {
                        t.cert_normal += 1;
                    }
                } else {
                    t.scanned += 1;
                }
            }
            report(variant, &k.to_string(), &t);
        }
    }
    println!();
}

fn main() {
    for dir in std::env::args().skip(1) {
        let (cols, train) = read(&format!("{dir}/train.txt"));
        let (_, test) = read(&format!("{dir}/test.txt"));
        let anom: Vec<bool> = test.iter().map(|r| r[cols] == 1).collect();
        let items = dispatch!(cols, &train, [6, 9, 16, 36]);
        let queries = dispatch!(cols, &test, [6, 9, 16, 36]);
        let name = dir.trim_end_matches('/').rsplit('/').next().unwrap().to_string();
        evaluate(&name, &items, &queries, &anom);
    }
}
