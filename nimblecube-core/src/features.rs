//! Window features for a slow analog sensor, relative to the unit's own clean-air level.
//!
//! Absolute ADC ranges tie every signature to one board: MQ-2 clean-air output differs
//! between units and drifts with age, temperature and humidity. Here all three features
//! are per mille of the unit's baseline level (the mean over its clean-air enrollment
//! windows), so a signature means the same on any unit. With the load resistor much
//! smaller than the sensor resistance, output over its clean-air value is about R0/Rs,
//! the ratio the MQ-2 datasheet curves are drawn in.

/// `FeatureEncoder` ranges for `relative`, in per mille of the baseline level. At 16
/// levels the channels step 250 / 50 / 50, and clean air (0) sits mid-level rather than on
/// a boundary where noise would flip it. Starting points: re-cut them from logged data.
pub const RELATIVE_RANGES: [(i32, i32); 3] = [(-375, 3625), (0, 800), (-425, 375)];

/// Raw statistics of one window of readings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowStats {
    pub mean: i32,
    pub peak: i32,
    /// Mean of the last quarter minus mean of the first quarter (last minus first below
    /// 4 readings). Averaging a quarter keeps one noisy sample from flipping a level, which
    /// matters at relative resolution: last-minus-first flipped 6-16% of clean windows.
    pub slope: i32,
}

impl WindowStats {
    pub const ZERO: WindowStats = WindowStats { mean: 0, peak: 0, slope: 0 };

    /// Mean, peak and quarter-to-quarter slope of `w`. An empty window gives all zero.
    pub fn of(w: &[i32]) -> Self {
        if w.is_empty() {
            return Self::ZERO;
        }
        let sum = |part: &[i32]| part.iter().map(|&v| v as i64).sum::<i64>();
        let q = (w.len() / 4).max(1);
        let slope = (sum(&w[w.len() - q..]) - sum(&w[..q])) / q as i64;
        WindowStats {
            mean: (sum(w) / w.len() as i64) as i32,
            peak: w.iter().copied().fold(i32::MIN, i32::max),
            slope: slope.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
        }
    }
}

/// Clean-air level of one unit: the mean of its enrollment windows' means.
#[derive(Clone, Copy, Debug, Default)]
pub struct Baseline {
    sum: i64,
    n: u32,
}

impl Baseline {
    pub const fn new() -> Self {
        Baseline { sum: 0, n: 0 }
    }

    /// Add one clean-air window.
    pub fn add(&mut self, s: &WindowStats) {
        self.sum += s.mean as i64;
        self.n += 1;
    }

    /// The baseline level, at least 1 so it can divide. `None` before any window.
    pub fn level(&self) -> Option<i32> {
        if self.n == 0 {
            return None;
        }
        Some(((self.sum / self.n as i64) as i32).max(1))
    }
}

/// `[mean - level, peak - mean, slope]`, each in per mille of `level` (clamped to >= 1).
/// Clean air gives about `[0, small, 0]` on any unit. Integer-only and panic-free.
pub fn relative(s: &WindowStats, level: i32) -> [i32; 3] {
    let level = level.max(1) as i64;
    let pm = |v: i64| (v * 1000 / level).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    let mean = s.mean as i64;
    [pm(mean - level), pm(s.peak as i64 - mean), pm(s.slope as i64)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::quantize;

    #[test]
    fn window_stats_basics() {
        let s = WindowStats::of(&[10, 30, 20, 16]);
        assert_eq!(s, WindowStats { mean: 19, peak: 30, slope: 6 });
        // 8 readings: quarters of 2, (10 + 12) / 2 - (0 + 2) / 2 = 10
        assert_eq!(WindowStats::of(&[0, 2, 4, 4, 4, 4, 10, 12]).slope, 10);
        assert_eq!(WindowStats::of(&[]), WindowStats::ZERO);
        assert_eq!(WindowStats::of(&[i32::MIN, i32::MAX]).slope, i32::MAX); // saturates
    }

    #[test]
    fn baseline_is_mean_of_means_and_never_zero() {
        let mut b = Baseline::new();
        assert_eq!(b.level(), None);
        b.add(&WindowStats { mean: 400, peak: 0, slope: 0 });
        b.add(&WindowStats { mean: 500, peak: 0, slope: 0 });
        assert_eq!(b.level(), Some(450));
        let mut z = Baseline::new();
        z.add(&WindowStats::ZERO);
        assert_eq!(z.level(), Some(1));
    }

    #[test]
    fn clean_air_is_zero_on_any_unit() {
        for level in [1, 300, 1600, 4095] {
            let s = WindowStats { mean: level, peak: level, slope: 0 };
            assert_eq!(relative(&s, level), [0, 0, 0]);
        }
    }

    #[test]
    fn same_ratio_same_features_across_units() {
        // 1.5x clean air, peak 10% over the mean, slope 5% of the level, on two units
        let a = WindowStats { mean: 600, peak: 660, slope: 20 };
        let b = WindowStats { mean: 2400, peak: 2640, slope: 80 };
        assert_eq!(relative(&a, 400), relative(&b, 1600));
        assert_eq!(relative(&a, 400), [500, 150, 50]);
    }

    #[test]
    fn extremes_do_not_panic() {
        let s = WindowStats { mean: i32::MAX, peak: i32::MIN, slope: i32::MIN };
        let r = relative(&s, 0);
        assert_eq!(r, [i32::MAX, i32::MIN, i32::MIN]);
    }

    #[test]
    fn clean_air_sits_mid_level() {
        // noise of up to just under half a step around 0 must not change the level
        let [lvl, _, slope] = RELATIVE_RANGES;
        assert_eq!(quantize(-124, lvl.0, lvl.1, 16), quantize(124, lvl.0, lvl.1, 16));
        assert_eq!(quantize(-24, slope.0, slope.1, 16), quantize(24, slope.0, slope.1, 16));
        assert_ne!(quantize(0, lvl.0, lvl.1, 16), quantize(126, lvl.0, lvl.1, 16));
    }
}
