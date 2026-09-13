//! Tabular anomaly benchmark driver: nimblecube's encoder and nearest-match score on
//! pre-scaled integer rows, for comparison with PyOD baselines on public datasets.
//! Host/std. The harness writes the rows; this writes one score line per test row.
//!   cargo run --release --example tabular_eval -- <train.txt> <test.txt> <scores.txt> [levels]
//!
//! `levels` is 16 (default, as in the firmware) or 64.
//!
//! Input: a first line "rows cols", then one row of integers per line, scaled so the
//! training normals span 0..1000 (values outside are clamped by `quantize`).
//! Output per test row: "<nearest> <centroid>", the Hamming distance to the nearest
//! training normal and to the bundle of all training normals. Higher = more anomalous.

use std::io::{BufRead, BufReader, BufWriter, Write};

use nimblecube_core::encode::FeatureEncoder;
use nimblecube_core::hv::Hv;

fn read(path: &str) -> (usize, Vec<Vec<i32>>) {
    let mut lines = BufReader::new(std::fs::File::open(path).expect("open input")).lines();
    let head = lines.next().expect("header").expect("read");
    let cols: usize = head.split_whitespace().nth(1).expect("cols").parse().expect("cols");
    let rows = lines
        .map(|l| l.expect("read").split_whitespace().map(|v| v.parse().expect("int")).collect())
        .collect();
    (cols, rows)
}

fn run<const CH: usize, const L: usize>(train: &[Vec<i32>], test: &[Vec<i32>]) -> Vec<(u32, u32)> {
    let enc = FeatureEncoder::<CH, L>::new(7, [(0, 1000); CH]);
    let hv = |r: &Vec<i32>| enc.encode(&core::array::from_fn(|i| r[i]));
    let normals: Vec<Hv> = train.iter().map(hv).collect();
    let centroid = Hv::bundle(&normals);
    test.iter()
        .map(|r| {
            let q = hv(r);
            (normals.iter().map(|n| n.hamming(&q)).min().unwrap(), centroid.hamming(&q))
        })
        .collect()
}

/// `FeatureEncoder` takes channels and levels as consts, so each column count is listed.
macro_rules! dispatch {
    ($cols:expr, $levels:expr, $train:expr, $test:expr, [$($n:literal),*]) => {
        match ($cols, $levels) {
            $(($n, 16) => run::<$n, 16>($train, $test), ($n, 64) => run::<$n, 64>($train, $test),)*
            (c, l) => panic!("add {c} columns / {l} levels to the dispatch list"),
        }
    };
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (cols, train) = read(&args[1]);
    let (_, test) = read(&args[2]);
    let levels: usize = args.get(4).map_or(16, |l| l.parse().expect("levels"));
    let scores =
        dispatch!(cols, levels, &train, &test, [5, 6, 8, 9, 10, 12, 13, 16, 21, 30, 32, 36]);
    let mut out = BufWriter::new(std::fs::File::create(&args[3]).expect("create output"));
    for (n, c) in scores {
        writeln!(out, "{n} {c}").expect("write");
    }
}
