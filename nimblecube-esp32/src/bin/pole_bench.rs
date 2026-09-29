//! On-chip timing of the exact pole and cell index from `examples/pole_eval.rs`.
//! Host counts compares; the chip also pays for ranking every item (poles only)
//! or every cell (the ring variants), so the host ranking may not survive.
//!
//! Datasets: the synthetic clusters (generated here, as in `net_tree_eval`) and
//! ADBench rows packed by `bench/cert_chip_data.py` into `certdata.bin`, encoded
//! here by the real `FeatureEncoder`. 500 queries each; recall is checked
//! against a linear scan, which is also the baseline.
//!   python3 bench/cert_prep.py 2_annthyroid 23_mammography 28_pendigits 30_satellite 32_shuttle
//!   python3 bench/cert_chip_data.py annthyroid mammography satellite pendigits shuttle
//!   cd nimblecube-esp32 && cargo run --release --bin pole_bench
//!
//! Measured on chip 2026-09-29, 500 queries each, every variant 500/500 exact.
//! Time speedup over the linear scan (compare-count speedup in brackets):
//!
//! ```text
//!                 linear us  poles only   combined   rings+poles  tight<384  tight<best
//! synthetic          59511   0.5x (3x)   8.9x (54x)  0.7x (3x)   8.9x (54x)  8.9x (54x)
//! annthyroid         12343   0.8x (185x) 3.5x (51x)  3.3x (108x) 3.3x (108x) 3.3x (104x)
//! mammography        11478   0.5x (432x) 5.8x (137x) 5.5x (250x) 5.5x (250x) 5.5x (243x)
//! satellite          30805   1.6x (6x)   2.2x (5x)   2.2x (5x)   2.2x (5x)   2.3x (5x)
//! pendigits          32136   1.3x (9x)   2.9x (9x)   2.5x (8x)   2.5x (8x)   2.5x (8x)
//! shuttle (12k)      32856   0.6x (266x) 6.6x (115x) 6.1x (231x) 6.1x (231x) 6.1x (211x)
//! build: synthetic 79 s, real 5-22 s (the certificate's lists took 966 s)
//! ```
//!
//! Compare counts match the host. Time does not: 50-250x fewer compares became
//! 2-9x. Poles only is slower than linear on four of six: ranking every item on
//! every query costs more than the compares it saves. The rest goes to bound
//! work, each visited member's pole row read from a scattered place in PSRAM,
//! while the linear scan reads sequentially and stops early. Storing items and
//! pole rows in cell order would make those reads sequential; not yet tried.

#![no_std]
#![no_main]

extern crate alloc;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::hint::black_box;
use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::delay::Delay;
use esp_hal::main;
use esp_hal::time::Instant;
use esp_println::println;
use nimblecube_core::encode::FeatureEncoder;
use nimblecube_core::hv::{Hv, DIM_BITS, WORDS};

esp_bootloader_esp_idf::esp_app_desc!();

const K: usize = 16; // global poles (snapped bundles)
const CAP: usize = 32; // net tree cell cap
const RADIUS: u32 = (DIM_BITS / 4) as u32; // net tree join radius
const TIGHT: u32 = 384; // tight-cell threshold, bits
const Q: usize = 500;

static DATA: &[u8] = include_bytes!("../../certdata.bin");

fn xs(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn bit(h: &Hv, i: usize) -> u32 {
    ((h.0[i / 64] >> (i % 64)) & 1) as u32
}

/// Full scan with early termination, the baseline the other benches use.
fn linear_nearest(items: &[Hv], q: &Hv) -> u32 {
    let mut best_d = items[0].hamming(q);
    for it in items.iter().skip(1) {
        let mut d = 0u32;
        let mut w = 0;
        while w < WORDS {
            d += (it.0[w] ^ q.0[w]).count_ones();
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

/// Poles: k groups by one assignment pass to k random seeds, bundled and snapped
/// by majority (ties to a seeded coin). Same construction as the host eval.
fn bundle_poles(items: &[Hv], k: usize, seed: u64) -> Vec<Hv> {
    let mut s = seed;
    let seeds: Vec<usize> = (0..k).map(|_| (xs(&mut s) as usize) % items.len()).collect();
    let mut counts = vec![vec![0u32; DIM_BITS]; k];
    let mut sizes = vec![0u32; k];
    for x in items {
        let mut g = 0;
        let mut gd = u32::MAX;
        for (j, &si) in seeds.iter().enumerate() {
            let d = items[si].hamming(x);
            if d < gd {
                gd = d;
                g = j;
            }
        }
        sizes[g] += 1;
        for (i, c) in counts[g].iter_mut().enumerate() {
            *c += bit(x, i);
        }
    }
    let mut out = Vec::new();
    for (c, &m) in counts.iter().zip(&sizes) {
        if m == 0 {
            continue;
        }
        let mut w = [0u64; WORDS];
        for (i, &v) in c.iter().enumerate() {
            if 2 * v > m || (2 * v == m && xs(&mut s) & 1 == 1) {
                w[i / 64] |= 1 << (i % 64);
            }
        }
        out.push(Hv(w));
    }
    out
}

struct Index {
    poles: Vec<Hv>,
    table: Vec<u16>, // n x k distances to the poles
    nets: Vec<Hv>,
    members: Vec<Vec<u32>>,
    parent: Vec<Vec<u16>>,   // member distance to its cell centre
    radius: Vec<u16>,        // cell covering radius
    rings: Vec<(u16, u16)>,  // cells x k: min, max member distance to each pole
}

impl Index {
    fn build(items: &[Hv]) -> Self {
        let poles = bundle_poles(items, K, 0xB01E ^ K as u64);
        let k = poles.len();
        let mut table = Vec::with_capacity(items.len() * k);
        for x in items {
            for p in &poles {
                table.push(p.hamming(x) as u16);
            }
        }
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
        let parent: Vec<Vec<u16>> = members
            .iter()
            .zip(&nets)
            .map(|(m, c)| m.iter().map(|&i| items[i as usize].hamming(c) as u16).collect())
            .collect();
        let radius = parent.iter().map(|p| *p.iter().max().unwrap()).collect();
        let mut rings = Vec::with_capacity(members.len() * k);
        for m in &members {
            for j in 0..k {
                let (mut lo, mut hi) = (u16::MAX, 0u16);
                for &i in m {
                    let t = table[i as usize * k + j];
                    lo = lo.min(t);
                    hi = hi.max(t);
                }
                rings.push((lo, hi));
            }
        }
        Index { poles, table, nets, members, parent, radius, rings }
    }

    fn k(&self) -> usize {
        self.poles.len()
    }

    fn pole_lb(&self, x: usize, dq: &[u16]) -> u32 {
        let row = &self.table[x * self.k()..(x + 1) * self.k()];
        row.iter().zip(dq).map(|(&a, &b)| a.abs_diff(b)).max().unwrap() as u32
    }
}

#[derive(Clone, Copy)]
enum Rule {
    PolesOnly,
    Always,
    Never,
    Tight,
    TightBest,
}

/// One exact query. Returns (distance, compares).
fn query(ix: &Index, items: &[Hv], q: &Hv, rule: Rule, order: &mut Vec<(u32, u32)>, dq: &mut Vec<u16>) -> (u32, u32) {
    let k = ix.k();
    dq.clear();
    for p in &ix.poles {
        dq.push(p.hamming(q) as u16);
    }
    let mut cmp = k as u32;
    let mut best = u32::MAX;
    order.clear();
    if let Rule::PolesOnly = rule {
        for i in 0..items.len() {
            order.push((ix.pole_lb(i, dq), i as u32));
        }
        order.sort_unstable();
        for &(lb, i) in order.iter() {
            if lb >= best {
                break;
            }
            cmp += 1;
            best = best.min(items[i as usize].hamming(q));
        }
        return (best, cmp);
    }
    for c in 0..ix.nets.len() {
        let mut lb = 0u32;
        for j in 0..k {
            let (lo, hi) = ix.rings[c * k + j];
            let d = dq[j];
            lb = lb.max(d.saturating_sub(hi).max(lo.saturating_sub(d)) as u32);
        }
        order.push((lb, c as u32));
    }
    order.sort_unstable();
    for &(lb, c) in order.iter() {
        if lb >= best {
            break;
        }
        let c = c as usize;
        let r = ix.radius[c] as u32;
        let use_centre = match rule {
            Rule::Always => true,
            Rule::Never => false,
            Rule::Tight => r < TIGHT,
            Rule::TightBest => r < best,
            Rule::PolesOnly => unreachable!(),
        };
        let dc = if use_centre {
            cmp += 1;
            ix.nets[c].hamming(q)
        } else {
            0
        };
        if use_centre && dc.saturating_sub(r) >= best {
            continue;
        }
        for (m, &x) in ix.members[c].iter().enumerate() {
            if use_centre && dc.abs_diff(ix.parent[c][m] as u32) >= best {
                continue;
            }
            if ix.pole_lb(x as usize, dq) >= best {
                continue;
            }
            cmp += 1;
            best = best.min(items[x as usize].hamming(q));
        }
    }
    (best, cmp)
}

fn encode_rows<const CH: usize>(raw: &[i32], rows: usize) -> Vec<Hv> {
    let enc = FeatureEncoder::<CH, 16>::new(7, [(0, 1000); CH]);
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        out.push(enc.encode(&core::array::from_fn(|i| raw[r * CH + i])));
    }
    out
}

fn encode(cols: usize, raw: &[i32], rows: usize) -> Vec<Hv> {
    match cols {
        6 => encode_rows::<6>(raw, rows),
        9 => encode_rows::<9>(raw, rows),
        16 => encode_rows::<16>(raw, rows),
        36 => encode_rows::<36>(raw, rows),
        c => panic!("add {} columns", c),
    }
}

/// Run every variant on one dataset; returns the report lines.
fn run(name: &str, items: &[Hv], queries: &[Hv], anom: &[bool]) -> Vec<String> {
    let tb = Instant::now();
    let ix = Index::build(items);
    let build_ms = tb.elapsed().as_millis();
    let mut lines = Vec::new();
    let n_anom = anom.iter().filter(|&&a| a).count();
    lines.push(format!(
        "{}: {} stored, {} queries ({} anom), {} cells, {} poles, build {} ms, heap free {}",
        name,
        items.len(),
        queries.len(),
        n_anom,
        ix.nets.len(),
        ix.k(),
        build_ms,
        esp_alloc::HEAP.free()
    ));
    println!("{}", lines[0]);

    let mut truth = Vec::with_capacity(queries.len());
    let t = Instant::now();
    for q in queries {
        truth.push(linear_nearest(items, black_box(q)));
    }
    let lin_us = t.elapsed().as_micros() / queries.len() as u64;
    lines.push(format!("  {:<14} {:>8} us/query", "linear", lin_us));

    let mut order: Vec<(u32, u32)> = Vec::with_capacity(items.len());
    let mut dq: Vec<u16> = Vec::with_capacity(K);
    for (label, rule) in [
        ("poles only", Rule::PolesOnly),
        ("combined", Rule::Always),
        ("rings + poles", Rule::Never),
        ("tight < 384", Rule::Tight),
        ("tight < best", Rule::TightBest),
    ] {
        let (mut hit, mut cmp) = (0usize, 0u64);
        let t = Instant::now();
        for (qi, q) in queries.iter().enumerate() {
            let (d, c) = query(&ix, items, black_box(q), rule, &mut order, &mut dq);
            hit += (d == truth[qi]) as usize;
            cmp += c as u64;
        }
        let us = t.elapsed().as_micros() / queries.len() as u64;
        lines.push(format!(
            "  {:<14} {:>8} us/query  recall {}/{}  compares {}  speedup {}x (compares {}x)",
            label,
            us,
            hit,
            queries.len(),
            cmp / queries.len() as u64,
            lin_us / us.max(1),
            items.len() as u64 * queries.len() as u64 / cmp.max(1)
        ));
        println!("{}", lines.last().unwrap());
    }
    lines
}

#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);
    let delay = Delay::new();
    // Give the serial reader time to attach, so an early hang or panic is not lost.
    delay.delay_millis(8000);
    println!("pole_bench: init psram...");
    let psram_config = esp_hal::psram::PsramConfig {
        mode: esp_hal::psram::PsramMode::OctalSpi,
        size: esp_hal::psram::PsramSize::AutoDetect,
        ..Default::default()
    };
    esp_alloc::psram_allocator!(peripherals.PSRAM, esp_hal::psram, psram_config);
    println!("pole_bench: heap free {}", esp_alloc::HEAP.free());

    let mut report: Vec<String> = Vec::new();

    // synthetic set, identical to net_tree_eval
    {
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
        let mut items: Vec<Hv> = Vec::with_capacity(12000);
        for i in 0..12000 {
            items.push(noisy(&bases[i % 200], &mut s));
        }
        let queries: Vec<Hv> = (0..Q)
            .map(|_| {
                let b = (xs(&mut s) as usize) % 200;
                noisy(&bases[b], &mut s)
            })
            .collect();
        drop(bases);
        report.extend(run("synthetic", &items, &queries, &[false; Q]));
    }

    // real datasets from certdata.bin
    let mut p = 0usize;
    let mut take = |n: usize| {
        let s = &DATA[p..p + n];
        p += n;
        s
    };
    let u32_at = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
    let datasets = u32_at(take(4));
    for _ in 0..datasets {
        let len = take(1)[0] as usize;
        let name = String::from(core::str::from_utf8(take(len)).unwrap());
        let h = take(10);
        let cols = u16::from_le_bytes([h[0], h[1]]) as usize;
        let (stored, nq) = (u32_at(&h[2..6]), u32_at(&h[6..10]));
        let to_i32 = |b: &[u8]| -> Vec<i32> {
            b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        let train = to_i32(take(stored * cols * 4));
        let test = to_i32(take(nq * cols * 4));
        let anom: Vec<bool> = take(nq).iter().map(|&l| l == 1).collect();
        let items = encode(cols, &train, stored);
        drop(train);
        let queries = encode(cols, &test, nq);
        report.extend(run(&name, &items, &queries, &anom));
    }

    loop {
        println!("pole_bench (ESP32-S3 @ 240 MHz)  k={} cap={} tight<{}", K, CAP, TIGHT);
        for l in &report {
            println!("{}", l);
        }
        println!("--");
        delay.delay_millis(3000);
    }
}
