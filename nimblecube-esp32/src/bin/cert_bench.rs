//! On-chip check of the triangle-inequality certificate and the cross-polytope
//! hash from `examples/net_tree_eval.rs`. Host compare counts predicted, at cap=32:
//!
//! ```text
//! cells nprobe=1          : 62.8% recall, 432 compares
//! cells nprobe=2          : 100%,         460
//! cells -> certificate    : 100%,         460
//! hash m=256 -> cert, cells fallback: 100%, 116 compares + hash (~36 eq)
//! ```
//!
//! Measured on chip 2026-09-29, 100 queries, two passes agreeing within 1 us:
//!
//! ```text
//! linear scan           : 59169 us/query
//! cells nprobe=1        :  9211 us   recall  58/100
//! cells nprobe=2        :  9809 us   recall 100/100
//! cells -> certificate  :  9820 us   recall 100/100
//! hash256 -> cert/cells :  2812 us   recall 100/100, 10 fallbacks   21x
//! hash256 alone         :   153 us
//! build: tree 66.5 s, neighbour lists 966 s (all pairs)
//! ```
//!
//! The 3x ratio between hash and cells transferred (3.5x here); the 79x against
//! linear did not (21x), the same access-pattern penalty `net_tree_bench` showed.
//!
//! Compare count over-predicted chip speed by 4x for the net tree, so every
//! variant is timed here in one binary, on the same queries, in two passes.
//! Neighbour lists hold ids only (60 u16 slots per item, 1.4 MB) to fit next to
//! the 6.1 MB of items in 8 MB PSRAM. Without stored distances the whole list is
//! scanned, which is still exact whenever `2r - 1` is within the list radius.
//!   cd nimblecube-esp32 && cargo run --release --bin cert_bench

#![no_std]
#![no_main]

extern crate alloc;
use alloc::vec::Vec;
use core::hint::black_box;
use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::delay::Delay;
use esp_hal::main;
use esp_hal::time::Instant;
use esp_println::println;
use nimblecube_core::hv::{Hv, DIM_BITS, WORDS};

esp_bootloader_esp_idf::esp_app_desc!();

const N: usize = 12000; // matches the host eval
const G: usize = 200; // clusters
const FLIP: usize = 200; // intra-cluster spread, in bits
const CAP: usize = 32; // splits every cluster in two, where nprobe=1 misses
const RADIUS: u32 = (DIM_BITS / 4) as u32; // join radius and list radius
const K: usize = 60; // list slots per item (a cluster has 59 neighbours)
const OVER: u8 = u8::MAX; // list overflowed, certificate unavailable for this item
const Q: usize = 100; // first 100 of the host's 500 queries
const M: usize = 256; // cross-polytope hash width, 2M buckets
const BITMAP: usize = N.div_ceil(32);

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

/// Full scan with early termination, the baseline `net_tree_bench` measured.
fn linear_nearest(items: &[Hv], q: &Hv) -> u32 {
    let mut best_d = items[0].hamming(q);
    for it in items.iter().skip(1) {
        let s = &it.0;
        let qq = &q.0;
        let mut d = 0u32;
        let mut w = 0;
        while w < WORDS {
            d += (s[w] ^ qq[w]).count_ones();
            if d >= best_d {
                break;
            }
            w += 1;
        }
        if w == WORDS {
            best_d = d;
        }
    }
    best_d
}

/// Hamming distance if it is at most `limit`, else `None`, stopping early.
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
    fn rebundle(&mut self, items: &[Hv], c: usize) {
        let mems: Vec<Hv> = self.members[c].iter().map(|&id| items[id as usize].clone()).collect();
        self.nets[c] = Hv::bundle(&mems);
    }

    fn insert(&mut self, items: &[Hv], id: u32) {
        let mut best = usize::MAX;
        let mut bd = u32::MAX;
        for (c, net) in self.nets.iter().enumerate() {
            if self.members[c].len() >= CAP {
                continue;
            }
            let d = net.hamming(&items[id as usize]);
            if d < bd {
                bd = d;
                best = c;
            }
        }
        if best == usize::MAX || bd > RADIUS {
            self.nets.push(items[id as usize].clone());
            let mut m = Vec::new();
            m.push(id);
            self.members.push(m);
            return;
        }
        self.members[best].push(id);
        self.rebundle(items, best);
    }

    /// Probe the `nprobe` (1 or 2) nearest cells. Returns (distance, id, first cell).
    fn query(&self, items: &[Hv], q: &Hv, nprobe: usize) -> (u32, u32, usize) {
        let mut chosen = [usize::MAX; 2];
        let mut chosen_d = [u32::MAX; 2];
        for (c, net) in self.nets.iter().enumerate() {
            let d = net.hamming(q);
            if d < chosen_d[0] {
                chosen_d[1] = chosen_d[0];
                chosen[1] = chosen[0];
                chosen_d[0] = d;
                chosen[0] = c;
            } else if d < chosen_d[1] {
                chosen_d[1] = d;
                chosen[1] = c;
            }
        }
        let (mut bd, mut bi) = (u32::MAX, 0u32);
        for &c in chosen.iter().take(nprobe) {
            if c == usize::MAX {
                continue;
            }
            for &id in &self.members[c] {
                let d = items[id as usize].hamming(q);
                if d < bd {
                    bd = d;
                    bi = id;
                }
            }
        }
        (bd, bi, chosen[0])
    }
}

/// Neighbour lists: every item within RADIUS, in K fixed u16 slots per item.
struct Lists {
    ids: Vec<u16>,
    cnt: Vec<u8>,
}

impl Lists {
    fn build(items: &[Hv]) -> Self {
        let mut ids: Vec<u16> = Vec::with_capacity(N * K);
        ids.resize(N * K, 0);
        let mut cnt: Vec<u8> = Vec::with_capacity(N);
        cnt.resize(N, 0);
        let push = |ids: &mut Vec<u16>, cnt: &mut Vec<u8>, i: usize, j: usize| {
            if cnt[i] == OVER {
                return;
            }
            if cnt[i] as usize == K {
                cnt[i] = OVER;
                return;
            }
            ids[i * K + cnt[i] as usize] = j as u16;
            cnt[i] += 1;
        };
        for i in 0..N {
            for j in (i + 1)..N {
                if within(&items[i], &items[j], RADIUS).is_some() {
                    push(&mut ids, &mut cnt, i, j);
                    push(&mut ids, &mut cnt, j, i);
                }
            }
            if i % 2000 == 0 {
                println!("  lists: row {}", i);
            }
        }
        Lists { ids, cnt }
    }

    /// Exact nearest given candidate `c` at distance `r`, or `None` when the
    /// certificate is unavailable. Ids marked in `seen` were already compared.
    fn certify(&self, items: &[Hv], q: &Hv, c: u32, r: u32, seen: &[u32; BITMAP]) -> Option<u32> {
        let n = self.cnt[c as usize];
        if 2 * r > RADIUS + 1 || n == OVER {
            return None;
        }
        let mut bd = r;
        let base = c as usize * K;
        for &id in &self.ids[base..base + n as usize] {
            if seen[id as usize / 32] >> (id % 32) & 1 == 1 {
                continue;
            }
            let d = items[id as usize].hamming(q);
            if d < bd {
                bd = d;
            }
        }
        Some(bd)
    }
}

fn mark(seen: &mut [u32; BITMAP], ids: &[u32], on: bool) {
    for &id in ids {
        let (w, b) = (id as usize / 32, id % 32);
        if on {
            seen[w] |= 1 << b;
        } else {
            seen[w] &= !(1 << b);
        }
    }
}

/// Same construction and seed as `CrossPolytope` in the host eval, so the
/// m=256 hash is bit-identical there and here.
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
        let mut y = [0i32; M];
        for (k, (&b, &sg)) in self.bits.iter().zip(&self.signs).enumerate() {
            y[k] = if (h.0[b / 64] >> (b % 64)) & 1 == 1 { sg } else { -sg };
        }
        let mut len = 1;
        while len < M {
            let mut i = 0;
            while i < M {
                for j in i..i + len {
                    let (a, b) = (y[j], y[j + len]);
                    y[j] = a + b;
                    y[j + len] = a - b;
                }
                i += 2 * len;
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
}

#[derive(Clone, Copy, Default)]
struct Row {
    us: u64,
    hit: usize,
    fell_back: usize,
}

/// Cells nprobe=1, then certify. Falls back to a full scan if unavailable.
fn cells_cert(items: &[Hv], tree: &NetTree, lists: &Lists, q: &Hv, seen: &mut [u32; BITMAP]) -> (u32, bool) {
    let (r, c, cell) = tree.query(items, q, 1);
    mark(seen, &tree.members[cell], true);
    let out = lists.certify(items, q, c, r, seen);
    mark(seen, &tree.members[cell], false);
    match out {
        Some(d) => (d, false),
        None => (linear_nearest(items, q), true),
    }
}

#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);
    let delay = Delay::new();
    // Give the serial reader time to attach, so an early hang or panic is not lost.
    delay.delay_millis(8000);
    println!("cert_bench: init psram...");

    let psram_config = esp_hal::psram::PsramConfig {
        mode: esp_hal::psram::PsramMode::OctalSpi,
        size: esp_hal::psram::PsramSize::AutoDetect,
        ..Default::default()
    };
    esp_alloc::psram_allocator!(peripherals.PSRAM, esp_hal::psram, psram_config);

    println!("cert_bench: heap free {} bytes; generating {} clustered items...", esp_alloc::HEAP.free(), N);

    let mut st: u64 = 0x000A_11CE_5EED;
    let bases: Vec<Hv> = (0..G).map(|_| rand_hv(&mut st)).collect();
    let mut items: Vec<Hv> = Vec::with_capacity(N);
    for i in 0..N {
        items.push(noisy(&bases[i % G], FLIP, &mut st));
    }
    let queries: Vec<Hv> = (0..Q)
        .map(|_| {
            let b = (xs(&mut st) as usize) % G;
            noisy(&bases[b], FLIP, &mut st)
        })
        .collect();
    drop(bases);

    println!("building net tree (cap={})...", CAP);
    let tb = Instant::now();
    let mut tree = NetTree { nets: Vec::with_capacity(N / CAP + 32), members: Vec::with_capacity(N / CAP + 32) };
    for i in 0..N {
        tree.insert(&items, i as u32);
    }
    let tree_ms = tb.elapsed().as_micros() / 1000;
    println!("built {} cells in {} ms, heap free {}", tree.nets.len(), tree_ms, esp_alloc::HEAP.free());

    println!("building neighbour lists (all pairs)...");
    let tl = Instant::now();
    let lists = Lists::build(&items);
    let lists_ms = tl.elapsed().as_micros() / 1000;
    let overflowed = lists.cnt.iter().filter(|&&c| c == OVER).count();
    let entries: usize = lists.cnt.iter().filter(|&&c| c != OVER).map(|&c| c as usize).sum();
    println!(
        "lists in {} ms, {} entries, {} overflowed, heap free {}",
        lists_ms,
        entries,
        overflowed,
        esp_alloc::HEAP.free()
    );

    let mut hs: u64 = 0xC0FF_EE11;
    let cp = CrossPolytope::new(&mut hs);
    let mut buckets: Vec<Vec<u32>> = (0..2 * M).map(|_| Vec::new()).collect();
    for (i, h) in items.iter().enumerate() {
        buckets[cp.hash(h)].push(i as u32);
    }

    let mut seen = [0u32; BITMAP];
    let mut truth: Vec<u32> = Vec::with_capacity(Q);
    let t0 = Instant::now();
    for q in queries.iter() {
        truth.push(linear_nearest(&items, black_box(q)));
    }
    let lin_us = t0.elapsed().as_micros() / Q as u64;

    // Two passes, second in reverse variant order, to expose layout or warm-up drift.
    let mut rows = [[Row::default(); 5]; 2];
    for pass in 0..2 {
        let order: [usize; 5] = if pass == 0 { [0, 1, 2, 3, 4] } else { [4, 3, 2, 1, 0] };
        for &v in &order {
            let mut row = Row::default();
            let t = Instant::now();
            for (qi, q) in queries.iter().enumerate() {
                let q = black_box(q);
                let d = match v {
                    0 => tree.query(&items, q, 1).0,
                    1 => tree.query(&items, q, 2).0,
                    2 => {
                        let (d, fb) = cells_cert(&items, &tree, &lists, q, &mut seen);
                        row.fell_back += fb as usize;
                        d
                    }
                    3 => {
                        let b = &buckets[cp.hash(q)];
                        let (mut r, mut c) = (u32::MAX, 0u32);
                        for &id in b {
                            let d = items[id as usize].hamming(q);
                            if d < r {
                                r = d;
                                c = id;
                            }
                        }
                        let out = if b.is_empty() {
                            None
                        } else {
                            mark(&mut seen, b, true);
                            let o = lists.certify(&items, q, c, r, &seen);
                            mark(&mut seen, b, false);
                            o
                        };
                        match out {
                            Some(d) => d,
                            None => {
                                row.fell_back += 1;
                                cells_cert(&items, &tree, &lists, q, &mut seen).0
                            }
                        }
                    }
                    _ => cp.hash(q) as u32,
                };
                if v != 4 && d == truth[qi] {
                    row.hit += 1;
                }
                black_box(d);
            }
            row.us = t.elapsed().as_micros() / Q as u64;
            rows[pass][v] = row;
        }
    }

    let names = [
        "cells nprobe=1        ",
        "cells nprobe=2        ",
        "cells -> certificate  ",
        "hash256 -> cert/cells ",
        "hash256 alone         ",
    ];
    loop {
        println!("cert_bench (ESP32-S3 @ 240 MHz)  N={} clusters={} cap={} queries={}", N, G, CAP, Q);
        println!(
            "cells={}  tree build={} ms  lists build={} ms  entries={} overflowed={}",
            tree.nets.len(),
            tree_ms,
            lists_ms,
            entries,
            overflowed
        );
        println!("linear scan           : {} us/query", lin_us);
        for (v, name) in names.iter().enumerate() {
            let (a, b) = (rows[0][v], rows[1][v]);
            if v == 4 {
                println!("{}: {} / {} us/query", name, a.us, b.us);
            } else {
                println!(
                    "{}: {} / {} us/query  recall {}/{}  fallback {}  speedup {}x",
                    name,
                    a.us,
                    b.us,
                    a.hit,
                    Q,
                    a.fell_back,
                    lin_us / a.us.max(1)
                );
            }
        }
        println!("--");
        delay.delay_millis(3000);
    }
}
