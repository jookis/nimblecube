//! Relative features: why the gas demo encodes per mille of the unit's own clean-air level
//! instead of absolute ADC counts. Synthetic MQ-2-like units that differ only in their
//! clean-air level, run through the same pipeline as `gas_anomaly` (32-reading windows,
//! 8 enrolled clean windows, threshold = worst clean spread + 70). Host/std, deterministic.
//!   cargo run --release --example relative_features
//!
//! Model: reading = level * gas * (1 + 1% noise) + 3 counts ADC noise; `gas` is 1.0 in
//! clean air, a steady factor in a plume, or a ramp from 1.0 during onset.
//!
//! Measured 2026-09-13, three units with clean-air levels 400 / 900 / 1600, 200 windows per
//! cell (share of windows over threshold):
//!
//! ```text
//!                               clean  x1.2  x1.5   x2   x3  onset  clean alarms (3 of 5)
//! own unit, absolute  A 400        0%    0%    0% 100% 100%     0%   0%
//!                     B/C          0%    0%  100% 100% 100%   100%   0%
//! own unit, relative  A, B, C      0%  100%  100% 100% 100%   100%   0%
//! A's store on B / C, absolute   100%  100%  100% 100% 100%   100%  100%
//! A's store on B / C, relative     0%  100%  100% 100% 100%   100%   0%
//! ```
//!
//! Absolute: sensitivity depends on which unit you got (A misses x1.5 and onset), and a
//! store from one unit flags clean air on every other. Relative: the same on all three,
//! and a store carries across. With last-minus-first slope, relative clean air false-
//! alarmed on 6-16% of windows; the quarter-mean slope in `WindowStats` fixed that.
//! Synthetic units that differ only in level; real units also differ in gain and drift.

use nimblecube_core::encode::FeatureEncoder;
use nimblecube_core::features::{relative, Baseline, WindowStats, RELATIVE_RANGES};
use nimblecube_core::store::FixedStore;

const W: usize = 32;
const ENROLL: usize = 8;
const TRIALS: usize = 200;
/// What `gas_anomaly` used before: mean, peak, slope over the raw 12-bit span.
const ABSOLUTE_RANGES: [(i32, i32); 3] = [(0, 4095), (0, 4095), (-4095, 4095)];
const UNITS: [(&str, f64); 3] = [("A", 400.0), ("B", 900.0), ("C", 1600.0)];

#[derive(Clone, Copy)]
enum Air {
    Steady(f64),
    Onset(f64), // ramps from 1.0 to the factor across the window
}
const AIRS: [(&str, Air); 6] = [
    ("clean", Air::Steady(1.0)),
    ("x1.2", Air::Steady(1.2)),
    ("x1.5", Air::Steady(1.5)),
    ("x2", Air::Steady(2.0)),
    ("x3", Air::Steady(3.0)),
    ("onset", Air::Onset(1.6)),
];

#[derive(Clone, Copy, PartialEq)]
enum Scheme {
    Absolute,
    Relative,
}

fn xs(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}
/// Roughly unit-variance noise: sum of four uniforms.
fn noise(s: &mut u64) -> f64 {
    (0..4).map(|_| 2.0 * (xs(s) >> 11) as f64 / (1u64 << 53) as f64 - 1.0).sum::<f64>() / 1.1547
}

fn window(level: f64, air: Air, s: &mut u64) -> WindowStats {
    let w: [i32; W] = core::array::from_fn(|i| {
        let gas = match air {
            Air::Steady(g) => g,
            Air::Onset(g) => 1.0 + (g - 1.0) * i as f64 / (W - 1) as f64,
        };
        let v = level * gas * (1.0 + 0.01 * noise(s)) + 3.0 * noise(s);
        v.round().clamp(0.0, 4095.0) as i32
    });
    WindowStats::of(&w)
}

struct Detector {
    enc: FeatureEncoder<3, 16>,
    scheme: Scheme,
    store: FixedStore<ENROLL>,
    threshold: u32,
}

impl Detector {
    fn features(&self, s: &WindowStats, level: i32) -> [i32; 3] {
        match self.scheme {
            Scheme::Absolute => [s.mean, s.peak, s.slope],
            Scheme::Relative => relative(s, level),
        }
    }
    /// Enroll clean windows exactly as `gas_anomaly` does.
    fn enroll(scheme: Scheme, clean: &[WindowStats], level: i32) -> Self {
        let ranges = if scheme == Scheme::Absolute { ABSOLUTE_RANGES } else { RELATIVE_RANGES };
        let enc = FeatureEncoder::new(7, ranges);
        let mut d = Detector { enc, scheme, store: FixedStore::new(), threshold: 0 };
        let mut worst = 0;
        for (i, c) in clean.iter().enumerate() {
            let hv = d.enc.encode(&d.features(c, level));
            if !d.store.is_empty() {
                worst = worst.max(d.store.nearest(&hv).unwrap().1);
            }
            d.store.insert(hv, i as u32).unwrap();
        }
        d.threshold = worst + 70;
        d
    }
    /// Over-threshold flag per window, as a stream, for a unit at `true_level` whose own
    /// measured baseline is `level`.
    fn flags(&self, true_level: f64, level: i32, air: Air, seed: u64) -> Vec<bool> {
        let mut s = seed;
        (0..TRIALS)
            .map(|_| {
                let f = self.features(&window(true_level, air, &mut s), level);
                self.store.nearest(&self.enc.encode(&f)).unwrap().1 > self.threshold
            })
            .collect()
    }
}

fn pct(n: usize, of: usize) -> f64 {
    100.0 * n as f64 / of as f64
}

/// Share of windows over threshold per air, then the share of the clean stream on which the
/// firmware's persistence rule (3 of the last 5 over) would sound the alarm.
fn row(label: &str, det: &Detector, true_level: f64, level: i32, u: usize) {
    print!("{label:30} thr {:4} ", det.threshold);
    let mut clean_alarms = 0.0;
    for (a, (_, air)) in AIRS.iter().enumerate() {
        let seed = 0x5EED_0000 + (u * 16 + a) as u64 * 7919; // same windows for both schemes
        let f = det.flags(true_level, level, *air, seed);
        print!("{:6.0}%", pct(f.iter().filter(|&&b| b).count(), f.len()));
        if a == 0 {
            let fired = f.windows(5).filter(|w| w.iter().filter(|&&b| b).count() >= 3).count();
            clean_alarms = pct(fired, f.len() - 4);
        }
    }
    println!("{clean_alarms:9.1}%");
}

fn main() {
    // Each unit's own start-up: 8 clean windows, and the baseline level measured from them.
    let clean: Vec<(Vec<WindowStats>, i32)> = UNITS
        .iter()
        .enumerate()
        .map(|(u, &(_, lvl))| {
            let mut s = 0xC1EA_0000 + u as u64 * 104729;
            let ws: Vec<WindowStats> =
                (0..ENROLL).map(|_| window(lvl, Air::Steady(1.0), &mut s)).collect();
            let mut b = Baseline::new();
            ws.iter().for_each(|w| b.add(w));
            (ws, b.level().unwrap())
        })
        .collect();

    print!("alarm rate per window, {TRIALS} windows per cell    ");
    AIRS.iter().for_each(|(name, _)| print!("{name:>7}"));
    println!("  alarm 3/5");
    println!("(clean = false alarms per window, the rest = detections, last = clean-air alarms)\n");

    println!("each unit enrolls itself:");
    for (u, &(name, lvl)) in UNITS.iter().enumerate() {
        let (ws, level) = &clean[u];
        for scheme in [Scheme::Absolute, Scheme::Relative] {
            let det = Detector::enroll(scheme, ws, *level);
            let tag = if scheme == Scheme::Absolute { "absolute" } else { "relative" };
            row(&format!("  unit {name} (clean {lvl:.0}) {tag}"), &det, lvl, *level, u);
        }
    }

    println!("\nstore enrolled on unit A, used on the others (each measures its own level):");
    for scheme in [Scheme::Absolute, Scheme::Relative] {
        let det = Detector::enroll(scheme, &clean[0].0, clean[0].1);
        for (u, &(name, lvl)) in UNITS.iter().enumerate().skip(1) {
            let tag = if scheme == Scheme::Absolute { "absolute" } else { "relative" };
            let label = format!("  A's store on {name} (clean {lvl:.0}) {tag}");
            row(&label, &det, lvl, clean[u].1, u);
        }
    }
}
