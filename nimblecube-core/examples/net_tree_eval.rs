//! Net-tree eval: an HDC-native index whose cells are `Hv::bundle` superpositions,
//! built incrementally with no training pass. Measures recall@1, compare count,
//! enrollment cost, fallback economics, and whether self-repair makes the miss
//! rate decay with use. Host/std, deterministic.
//!   cargo run --release --example net_tree_eval
//!
//! Same generators and sizes as `ivf_eval.rs`, so rows compare directly.
//!
//! Measured 2026-09-09, n=12000, 200 clusters, flip=200, 500 queries:
//!
//! ```text
//! clustered, greedy cap=64 : 100% recall @ nprobe=1, 260 compares, 46.2x
//! clustered, IVF (existing): 100% recall,             321 compares, 37x
//! clustered, in-order ctrl :  1.8% recall  <- the assignment rule does all the work
//! uniform,   greedy        : 100% recall,           12001 compares, 1.0x
//! uniform,   IVF (existing):   ~2% recall
//! enrollment: greedy 2.75M compare-equivalents vs IVF ~12M (200 x 12000 x 5 iters)
//! ```
//!
//! Two ideas measured and rejected, both at cap=32 where 37.2% of queries miss:
//!
//! - **Distance-threshold fallback does not work.** Returned distance on hits
//!   spans 328-358 and on misses 342-364. The ranges overlap, so no threshold
//!   separates a right answer from a wrong one. A miss still lands inside the
//!   correct cluster, just not on its nearest member, so it looks identical.
//! - **Self-repair does not converge.** Duplicating the true nearest into the
//!   wrongly chosen cell left the miss rate flat across five batches
//!   (38/34/33/33/37%). The cause is not a few misfiled items, it is a cluster
//!   split across two cells, and one duplicate per miss cannot cover the 12000
//!   possible answers.
//!
//! `nprobe=2` fixes 100% of those misses for 460 compares. Widening the probe
//! beats both detecting and repairing, and its cost is deterministic, which
//! matters more than average cost on a device with a deadline.
//!
//! **Triangle-inequality certificate** (measured 2026-09-29). Probe one cell, take
//! candidate `c` at distance `r`, then scan `c`'s build-time neighbour list up to
//! `2r`. Anything closer than `c` must be in that ball, so the result is proven exact:
//!
//! ```text
//! cap=32: certificate 100% at 460 compares, nprobe=2 100% at 460  (tie)
//! cap=16: certificate 100% at 860 compares, nprobe=2 62.6% at 832
//! uniform: lists empty, certificate unavailable on 500 of 500 queries
//! cost: 59 list entries per item, O(n^2) build (72M compares)
//! ```
//!
//! **Cross-polytope hash in place of cell ranking** (measured 2026-09-29). Hash the
//! query to the nearest face of the cube after a Walsh-Hadamard rotation, certify
//! the bucket's best item, fall back when the hash fails. Hash cost is an op-count
//! estimate in compare-equivalents, not timing:
//!
//! ```text
//! m=256 : hash ok 92.0%, fallback cells 100% at 152 (78.9x), full scan 1075
//! m=1024: hash ok 87.4%, fallback cells 100% at 293 (40.9x)
//! m=4096: hash ok 85.4%, fallback cells 100% at 953 (12.6x)
//! ```
//!
//! The fallback decides it: 8% of queries hashing off-cluster cost a full scan each.

use nimblecube_core::hv::{Hv, DIM_BITS, WORDS};

fn xs(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}
fn rand_hv(s: &mut u64) -> Hv {
    let mut w = [0u64; WORDS];
    for x in w.iter_mut() {
        *x = xs(s);
    }
    Hv(w)
}
fn noisy(base: &Hv, nbits: usize, s: &mut u64) -> Hv {
    let mut h = base.clone();
    for _ in 0..nbits {
        let b = (xs(s) as usize) % DIM_BITS;
        h.0[b / 64] ^= 1u64 << (b % 64);
    }
    h
}

/// Ground truth: nearest distance and its id.
fn linear_nearest(items: &[Hv], q: &Hv) -> (u32, u32) {
    let mut bd = u32::MAX;
    let mut bi = 0u32;
    for (i, h) in items.iter().enumerate() {
        let d = h.hamming(q);
        if d < bd {
            bd = d;
            bi = i as u32;
        }
    }
    (bd, bi)
}

/// Cells are bundles. An item may appear in more than one cell, which is what
/// lets repair add reachability without ever removing it.
struct NetTree {
    nets: Vec<Hv>,
    members: Vec<Vec<u32>>,
    cap: usize,
    join_radius: u32,
}

impl NetTree {
    fn rebundle(&mut self, items: &[Hv], c: usize) {
        let mems: Vec<Hv> = self.members[c].iter().map(|&id| items[id as usize].clone()).collect();
        self.nets[c] = Hv::bundle(&mems);
    }

    /// Greedy incremental insert. Returns work done, in compare-equivalents:
    /// one per net probed, plus one per member touched by the re-bundle.
    fn insert(&mut self, items: &[Hv], id: u32) -> u64 {
        let mut work = self.nets.len() as u64;
        let mut best = usize::MAX;
        let mut bd = u32::MAX;
        for (c, net) in self.nets.iter().enumerate() {
            if self.members[c].len() >= self.cap {
                continue;
            }
            let d = net.hamming(&items[id as usize]);
            if d < bd {
                bd = d;
                best = c;
            }
        }
        if best == usize::MAX || bd > self.join_radius {
            self.nets.push(items[id as usize].clone());
            self.members.push(vec![id]);
            return work;
        }
        self.members[best].push(id);
        work += self.members[best].len() as u64;
        self.rebundle(items, best);
        work
    }

    /// Probe the `nprobe` nearest nets. Returns (best distance, compares, ranked nets).
    fn query(&self, items: &[Hv], q: &Hv, nprobe: usize) -> (u32, u64, Vec<usize>) {
        let (bd, _, compares, ranked) = self.query_id(items, q, nprobe);
        (bd, compares, ranked)
    }

    /// As `query`, also returning the id of the best item.
    fn query_id(&self, items: &[Hv], q: &Hv, nprobe: usize) -> (u32, u32, u64, Vec<usize>) {
        let mut nd: Vec<(u32, usize)> =
            self.nets.iter().enumerate().map(|(c, net)| (net.hamming(q), c)).collect();
        nd.sort_by_key(|x| x.0);
        let mut compares = self.nets.len() as u64;
        let mut bd = u32::MAX;
        let mut bi = u32::MAX;
        for &(_, c) in nd.iter().take(nprobe.min(nd.len())) {
            for &id in &self.members[c] {
                let d = items[id as usize].hamming(q);
                compares += 1;
                if d < bd {
                    bd = d;
                    bi = id;
                }
            }
        }
        (bd, bi, compares, nd.iter().map(|x| x.1).collect())
    }

    fn total_members(&self) -> usize {
        self.members.iter().map(|m| m.len()).sum()
    }
}

fn build_greedy(items: &[Hv], cap: usize, join_radius: u32) -> (NetTree, u64) {
    let mut t = NetTree { nets: Vec::new(), members: Vec::new(), cap, join_radius };
    let mut work = 0u64;
    for i in 0..items.len() {
        work += t.insert(items, i as u32);
    }
    (t, work)
}

/// Control: cells filled in arrival order, no assignment logic at all. If this
/// scores as well as greedy, the assignment rule is buying nothing.
fn build_inorder(items: &[Hv], cap: usize) -> (NetTree, u64) {
    let mut t = NetTree { nets: Vec::new(), members: Vec::new(), cap, join_radius: u32::MAX };
    for (i, _) in items.iter().enumerate() {
        if t.members.last().map(|m| m.len() >= cap).unwrap_or(true) {
            t.members.push(Vec::new());
            t.nets.push(Hv::zero());
        }
        t.members.last_mut().unwrap().push(i as u32);
    }
    let mut work = 0u64;
    for c in 0..t.nets.len() {
        work += t.members[c].len() as u64;
        t.rebundle(items, c);
    }
    (t, work)
}

struct Truth {
    d: Vec<u32>,
    id: Vec<u32>,
}

fn truth_of(items: &[Hv], queries: &[Hv]) -> Truth {
    let mut d = Vec::with_capacity(queries.len());
    let mut id = Vec::with_capacity(queries.len());
    for q in queries {
        let (bd, bi) = linear_nearest(items, q);
        d.push(bd);
        id.push(bi);
    }
    Truth { d, id }
}

/// Stage 1a: recall and cost. The kill gate.
fn run_recall(items: &[Hv], queries: &[Hv], truth: &Truth, label: &str) {
    println!("{}:", label);
    println!(
        "  {:<20} {:>5} {:>7} {:>9} {:>8} {:>13} {:>10}",
        "variant", "cells", "nprobe", "recall@1", "speedup", "avg_compares", "enroll"
    );
    let n = items.len() as f64;
    for &cap in &[32usize, 64, 128] {
        let (greedy, gwork) = build_greedy(items, cap, (DIM_BITS / 4) as u32);
        let (inorder, iwork) = build_inorder(items, cap);
        for (name, tree, work) in
            [("greedy", &greedy, gwork), ("in-order (control)", &inorder, iwork)]
        {
            for &nprobe in &[1usize, 2, 4] {
                let mut hit = 0usize;
                let mut total = 0u64;
                for (qi, q) in queries.iter().enumerate() {
                    let (bd, cmp, _) = tree.query(items, q, nprobe);
                    if bd == truth.d[qi] {
                        hit += 1;
                    }
                    total += cmp;
                }
                let avg = total as f64 / queries.len() as f64;
                println!(
                    "  {:<20} {:>5} {:>7} {:>8.1}% {:>7.1}x {:>13.0} {:>10}",
                    name,
                    tree.nets.len(),
                    nprobe,
                    100.0 * hit as f64 / queries.len() as f64,
                    n / avg,
                    avg,
                    work
                );
            }
        }
    }
    println!();
}

/// Stage 1a instrumentation: can a distance threshold tell a good answer from a
/// wrong one, and would probing one more cell have fixed the misses anyway?
fn run_fallback(items: &[Hv], queries: &[Hv], truth: &Truth, cap: usize) {
    let (tree, _) = build_greedy(items, cap, (DIM_BITS / 4) as u32);
    let mut hit_d: Vec<u32> = Vec::new();
    let mut miss_d: Vec<u32> = Vec::new();
    let mut miss_in_2nd = 0usize;
    for (qi, q) in queries.iter().enumerate() {
        let (bd, _, ranked) = tree.query(items, q, 1);
        if bd == truth.d[qi] {
            hit_d.push(bd);
        } else {
            miss_d.push(bd);
            if ranked.len() > 1 && tree.members[ranked[1]].contains(&truth.id[qi]) {
                miss_in_2nd += 1;
            }
        }
    }
    hit_d.sort_unstable();
    miss_d.sort_unstable();
    let pct = |v: &Vec<u32>, p: f64| -> i64 {
        if v.is_empty() {
            -1
        } else {
            v[((v.len() - 1) as f64 * p) as usize] as i64
        }
    };
    let p = 100.0 * miss_d.len() as f64 / queries.len() as f64;
    println!("fallback analysis (cap={}, nprobe=1):", cap);
    println!("  miss rate p           : {:.1}%  ({} of {})", p, miss_d.len(), queries.len());
    println!(
        "  returned d on hits    : min {} median {} max {}",
        pct(&hit_d, 0.0),
        pct(&hit_d, 0.5),
        pct(&hit_d, 1.0)
    );
    println!(
        "  returned d on misses  : min {} median {} max {}",
        pct(&miss_d, 0.0),
        pct(&miss_d, 0.5),
        pct(&miss_d, 1.0)
    );
    let verdict = if miss_d.is_empty() {
        "n/a (no misses to detect)"
    } else if hit_d.is_empty() || miss_d[0] as i64 > pct(&hit_d, 1.0) {
        "YES (ranges disjoint)"
    } else {
        "NO (ranges overlap)"
    };
    println!("  threshold separable   : {}", verdict);
    if !miss_d.is_empty() {
        println!(
            "  misses fixed by nprobe=2: {:.0}%  ({} of {})",
            100.0 * miss_in_2nd as f64 / miss_d.len() as f64,
            miss_in_2nd,
            miss_d.len()
        );
    }
    let n = items.len() as f64;
    let base = tree.nets.len() as f64 + cap as f64;
    println!(
        "  expected compares w/ full-scan fallback: {:.0}  ({:.1}x)",
        base + p / 100.0 * n,
        n / (base + p / 100.0 * n)
    );
    println!();
}

/// Build-time neighbour lists: for each item, every other item within `radius`,
/// sorted by distance. Returns the lists and the build work in compares.
fn neighbour_lists(items: &[Hv], radius: u32) -> (Vec<Vec<(u32, u32)>>, u64) {
    let n = items.len();
    let mut lists: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
    let mut work = 0u64;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = items[i].hamming(&items[j]);
            work += 1;
            if d <= radius {
                lists[i].push((d, j as u32));
                lists[j].push((d, i as u32));
            }
        }
    }
    for l in lists.iter_mut() {
        l.sort_unstable();
    }
    (lists, work)
}

/// Triangle-inequality certificate. Probe one cell, get candidate `c` at distance
/// `r`. Any item closer to the query than `r` lies within `2r - 1` of `c`, so
/// scanning `c`'s list up to that radius returns the exact nearest. If the list
/// radius is below `2r - 1` the certificate is unavailable and the query falls
/// back to a full scan.
fn run_certificate(items: &[Hv], queries: &[Hv], truth: &Truth, cap: usize, label: &str) {
    let list_radius = (DIM_BITS / 4) as u32;
    let (tree, _) = build_greedy(items, cap, list_radius);
    let (lists, build) = neighbour_lists(items, list_radius);
    let entries: usize = lists.iter().map(|l| l.len()).sum();
    let n = items.len();

    let mut base_hit = 0usize;
    let mut cert_hit = 0usize;
    let mut fixed = 0usize;
    let mut unavailable = 0usize;
    let mut cmp_raw = 0u64;
    let mut cmp_dedup = 0u64;
    let mut scanned_max = 0usize;
    for (qi, q) in queries.iter().enumerate() {
        let (r, c, base_cmp, ranked) = tree.query_id(items, q, 1);
        let base_ok = r == truth.d[qi];
        if base_ok {
            base_hit += 1;
        }
        cmp_raw += base_cmp;
        cmp_dedup += base_cmp;
        let bound = 2 * r;
        if bound > list_radius + 1 {
            unavailable += 1;
            cmp_raw += n as u64;
            cmp_dedup += n as u64;
            cert_hit += 1;
            continue;
        }
        let probed = &tree.members[ranked[0]];
        let mut bd = r;
        let mut scanned = 0usize;
        for &(dc, id) in &lists[c as usize] {
            if dc >= bound {
                break;
            }
            scanned += 1;
            cmp_raw += 1;
            if probed.contains(&id) {
                continue;
            }
            cmp_dedup += 1;
            let d = items[id as usize].hamming(q);
            if d < bd {
                bd = d;
            }
        }
        scanned_max = scanned_max.max(scanned);
        if bd == truth.d[qi] {
            cert_hit += 1;
            if !base_ok {
                fixed += 1;
            }
        }
    }
    let nq = queries.len() as f64;
    let misses = queries.len() - base_hit;
    let (mut p2_hit, mut p2_cmp) = (0usize, 0u64);
    for (qi, q) in queries.iter().enumerate() {
        let (bd, cmp, _) = tree.query(items, q, 2);
        if bd == truth.d[qi] {
            p2_hit += 1;
        }
        p2_cmp += cmp;
    }
    println!("certificate ({}, cap={}, list radius {}):", label, cap, list_radius);
    println!("  nprobe=1 recall        : {:.1}%  ({} misses)", 100.0 * base_hit as f64 / nq, misses);
    println!("  certified recall       : {:.1}%", 100.0 * cert_hit as f64 / nq);
    if misses > 0 {
        println!("  misses fixed           : {} of {}", fixed, misses);
    }
    println!("  unavailable (2r > list): {} of {}", unavailable, queries.len());
    println!(
        "  avg compares           : {:.0} raw, {:.0} skipping probed cell ({:.1}x)",
        cmp_raw as f64 / nq,
        cmp_dedup as f64 / nq,
        n as f64 / (cmp_dedup as f64 / nq)
    );
    println!(
        "  nprobe=2 for reference : {:.1}% recall, {:.0} compares",
        100.0 * p2_hit as f64 / nq,
        p2_cmp as f64 / nq
    );
    println!("  largest list scanned   : {}", scanned_max);
    println!(
        "  list storage           : {} entries, {:.1} per item; build {} compares",
        entries,
        entries as f64 / n as f64,
        build
    );
    println!();
}

/// Cross-polytope hash: take `m` fixed bit positions as +-1, flip fixed random
/// signs, Walsh-Hadamard transform, and return the face the largest coordinate
/// points at (one of `2m` buckets).
struct CrossPolytope {
    bits: Vec<usize>,
    signs: Vec<i32>,
}

impl CrossPolytope {
    fn new(m: usize, s: &mut u64) -> Self {
        let mut pool: Vec<usize> = (0..DIM_BITS).collect();
        for i in (1..DIM_BITS).rev() {
            pool.swap(i, (xs(s) as usize) % (i + 1));
        }
        pool.truncate(m);
        let signs = (0..m).map(|_| if xs(s) & 1 == 0 { 1 } else { -1 }).collect();
        CrossPolytope { bits: pool, signs }
    }

    fn hash(&self, h: &Hv) -> usize {
        let mut y: Vec<i32> = self
            .bits
            .iter()
            .zip(&self.signs)
            .map(|(&b, &sg)| if (h.0[b / 64] >> (b % 64)) & 1 == 1 { sg } else { -sg })
            .collect();
        let m = y.len();
        let mut len = 1;
        while len < m {
            for i in (0..m).step_by(2 * len) {
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

    /// Hash cost in compare-equivalents: m*log2(m) add/subs plus m sign loads,
    /// counted against 64 word ops per compare. An op-count estimate, not timing.
    fn cost(&self) -> f64 {
        let m = self.bits.len() as f64;
        (m * m.log2() + m) / WORDS as f64
    }
}

/// Scan `c`'s neighbour list below `2r`, skipping ids in `skip`. Returns the best
/// distance and the compares spent. Exact whenever `2r - 1` is within the list radius.
fn certify(items: &[Hv], lists: &[Vec<(u32, u32)>], q: &Hv, c: u32, r: u32, skip: &[u32]) -> (u32, u64) {
    let mut bd = r;
    let mut cmp = 0u64;
    for &(dc, id) in &lists[c as usize] {
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
    (bd, cmp)
}

/// Replace cell ranking with a cross-polytope hash, then certify the bucket's
/// best item with the neighbour lists. When the hash fails (empty bucket, or `2r`
/// past the list radius) fall back either to a full scan or to the cell path
/// (cap=32 tree, nprobe=1, certified), which itself falls back to a full scan.
fn run_hash_certificate(items: &[Hv], queries: &[Hv], truth: &Truth) {
    let list_radius = (DIM_BITS / 4) as u32;
    let (lists, _) = neighbour_lists(items, list_radius);
    let (tree, _) = build_greedy(items, 32, list_radius);
    let n = items.len();
    let nq = queries.len() as f64;
    println!("hash -> certificate (list radius {}, fallback tree cap=32):", list_radius);
    println!(
        "  {:>5} {:>8} {:>8} {:>10} {:>9} {:>9} {:>9} {:>9} {:>8}",
        "m", "buckets", "hash ok", "fallback", "recall", "compares", "hash eq", "total", "speedup"
    );
    let mut s: u64 = 0xC0FF_EE11;
    for &m in &[256usize, 1024, 4096] {
        let cp = CrossPolytope::new(m, &mut s);
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); 2 * m];
        for (i, h) in items.iter().enumerate() {
            buckets[cp.hash(h)].push(i as u32);
        }
        for fallback in ["full scan", "cells"] {
            let (mut hit, mut ok, mut cmp) = (0usize, 0usize, 0u64);
            for (qi, q) in queries.iter().enumerate() {
                let bucket = &buckets[cp.hash(q)];
                let (mut r, mut c) = (u32::MAX, 0u32);
                for &id in bucket {
                    let d = items[id as usize].hamming(q);
                    cmp += 1;
                    if d < r {
                        r = d;
                        c = id;
                    }
                }
                let bd = if !bucket.is_empty() && 2 * r <= list_radius + 1 {
                    ok += 1;
                    let (bd, k) = certify(items, &lists, q, c, r, bucket);
                    cmp += k;
                    bd
                } else if fallback == "cells" {
                    let (r2, c2, k, ranked) = tree.query_id(items, q, 1);
                    cmp += k;
                    if 2 * r2 <= list_radius + 1 {
                        let (bd, k2) = certify(items, &lists, q, c2, r2, &tree.members[ranked[0]]);
                        cmp += k2;
                        bd
                    } else {
                        cmp += n as u64;
                        truth.d[qi]
                    }
                } else {
                    cmp += n as u64;
                    truth.d[qi]
                };
                if bd == truth.d[qi] {
                    hit += 1;
                }
            }
            let avg = cmp as f64 / nq;
            let total = avg + cp.cost();
            println!(
                "  {:>5} {:>8} {:>7.1}% {:>10} {:>8.1}% {:>9.0} {:>9.0} {:>9.0} {:>7.1}x",
                m,
                2 * m,
                100.0 * ok as f64 / nq,
                fallback,
                100.0 * hit as f64 / nq,
                avg,
                cp.cost(),
                total,
                n as f64 / total
            );
        }
    }
    println!();
}

/// Stage 1b: does learning from a fallback make the miss rate decay?
///
/// On a miss the full scan has already found the true nearest, so its id is free.
/// Add it to the cell we wrongly descended into, leaving it in its original cell
/// too. That only ever adds reachability. Overlap is what hard assignment cannot do.
fn run_repair(items: &[Hv], queries: &[Hv], truth: &Truth, cap: usize, batches: usize) {
    // Repair needs headroom above the insert cap. A cell at capacity is precisely
    // the cell that splits a cluster and causes the miss, so gating repair on the
    // insert cap would block it exactly when it is needed.
    let repair_cap = cap * 2;
    let (mut tree, _) = build_greedy(items, cap, (DIM_BITS / 4) as u32);
    let per = queries.len() / batches;
    println!("self-repair (cap={}, nprobe=1, {} batches of {}):", cap, batches, per);
    println!("  {:>5} {:>10} {:>12} {:>12}", "batch", "miss rate", "duplicates", "cells");
    for b in 0..batches {
        let mut miss = 0usize;
        let mut added = 0usize;
        for (qi, query) in queries.iter().enumerate().skip(b * per).take(per) {
            let (bd, _, ranked) = tree.query(items, query, 1);
            if bd != truth.d[qi] {
                miss += 1;
                let c = ranked[0];
                if tree.members[c].len() < repair_cap && !tree.members[c].contains(&truth.id[qi]) {
                    tree.members[c].push(truth.id[qi]);
                    tree.rebundle(items, c);
                    added += 1;
                }
            }
        }
        println!(
            "  {:>5} {:>9.1}% {:>12} {:>12}",
            b,
            100.0 * miss as f64 / per as f64,
            added,
            tree.nets.len()
        );
    }
    let biggest = tree.members.iter().map(|m| m.len()).max().unwrap_or(0);
    println!(
        "  total stored refs: {} (items: {}), largest cell: {} (insert cap {}, repair cap {})",
        tree.total_members(),
        items.len(),
        biggest,
        cap,
        repair_cap
    );
    println!();
}

fn main() {
    let mut s: u64 = 0x000A_11CE_5EED;
    let n = 12000usize;
    let g = 200usize;
    let flip = 200usize;
    let q = 500usize;

    let bases: Vec<Hv> = (0..g).map(|_| rand_hv(&mut s)).collect();
    let items: Vec<Hv> = (0..n).map(|i| noisy(&bases[i % g], flip, &mut s)).collect();
    let queries: Vec<Hv> = (0..q)
        .map(|_| {
            let b = (xs(&mut s) as usize) % g;
            noisy(&bases[b], flip, &mut s)
        })
        .collect();
    let rand_items: Vec<Hv> = (0..n).map(|_| rand_hv(&mut s)).collect();
    let rand_queries: Vec<Hv> = (0..q).map(|_| rand_hv(&mut s)).collect();

    println!("net_tree_eval  n={} queries={} clusters={} flip={}", n, q, g, flip);
    println!("IVF bar to beat (from ivf_eval): 100% recall at 321 avg compares\n");

    let truth = truth_of(&items, &queries);
    run_recall(&items, &queries, &truth, "clustered");
    // cap=64 holds a whole cluster, so nothing misses. cap=32 splits every
    // cluster in two, which is where fallback and repair actually get exercised.
    run_fallback(&items, &queries, &truth, 64);
    run_fallback(&items, &queries, &truth, 32);
    run_repair(&items, &queries, &truth, 32, 5);
    run_certificate(&items, &queries, &truth, 32, "clustered");
    run_certificate(&items, &queries, &truth, 16, "clustered");
    run_hash_certificate(&items, &queries, &truth);

    let rtruth = truth_of(&rand_items, &rand_queries);
    run_recall(&rand_items, &rand_queries, &rtruth, "uniform-random (no structure expected)");
    run_certificate(&rand_items, &rand_queries, &rtruth, 32, "uniform-random");
}
