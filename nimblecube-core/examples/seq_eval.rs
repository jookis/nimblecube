//! Sequence benchmark driver: nimblecube's own operations on variable-length symbolic sequences,
//! for comparison with the usual sequence baselines (n-gram cosine, TF-IDF, edit distance, compression).
//! Host/std. The harness writes the sequences; this writes one line of scores per test sequence.
//!   cargo run --release --example seq_eval -- <train.txt> <test.txt> <out.txt> [n]
//!
//! Encoding (the standard HDC n-gram scheme): every token gets a random 4096-bit vector, position i
//! inside an n-gram is `permute(i)`, an n-gram is the XOR (bind) of its permuted tokens, and a sequence
//! is the majority bundle of all its n-grams. Length drops out, so sequences of any length compare
//! directly, and a shift or an inserted token changes only the n-grams that touch it.
//!
//! Input format: first line "sequences vocab", then one sequence per line as "label tok tok tok ...".
//! Output per test sequence: "<nearest label> <nearest distance> <d_class0> <d_class1> ...", where the
//! class distances are to the bundle of all training sequences of that class (the HDC "centroid"
//! classifier). Higher distance = less similar.

use std::io::{BufRead, BufReader, BufWriter, Write};

use nimblecube_core::hv::{Hv, WORDS};

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

fn read(path: &str) -> (usize, Vec<(usize, Vec<usize>)>) {
    let mut lines = BufReader::new(std::fs::File::open(path).expect("open input")).lines();
    let head = lines.next().expect("header").expect("read");
    let vocab: usize = head.split_whitespace().nth(1).expect("vocab").parse().expect("vocab");
    let seqs = lines
        .map(|l| {
            let l = l.expect("read");
            let mut it = l.split_whitespace().map(|v| v.parse::<usize>().expect("int"));
            let label = it.next().expect("label");
            (label, it.collect())
        })
        .collect();
    (vocab, seqs)
}

/// bundle of all n-grams: each n-gram is the XOR of its tokens, token j permuted by j
fn encode(seq: &[usize], toks: &[Hv], n: usize) -> Hv {
    if seq.is_empty() {
        return Hv::zero();
    }
    let grams: Vec<Hv> = if seq.len() < n {
        vec![seq.iter().enumerate().fold(Hv::zero(), |a, (j, &t)| a.bind(&toks[t].permute(j)))]
    } else {
        seq.windows(n)
            .map(|w| w.iter().enumerate().fold(Hv::zero(), |a, (j, &t)| a.bind(&toks[t].permute(j))))
            .collect()
    };
    Hv::bundle(&grams)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (train_p, test_p, out_p) = (&a[1], &a[2], &a[3]);
    let n: usize = a.get(4).map(|v| v.parse().expect("n")).unwrap_or(3);

    let (vocab, train) = read(train_p);
    let (_, test) = read(test_p);
    let mut seed = 0x5eed_1234_u64;
    let toks: Vec<Hv> = (0..vocab).map(|_| rand_hv(&mut seed)).collect();

    let tr: Vec<(usize, Hv)> = train.iter().map(|(l, s)| (*l, encode(s, &toks, n))).collect();
    let nclass = tr.iter().map(|(l, _)| *l + 1).max().unwrap_or(0);
    let centroids: Vec<Hv> = (0..nclass)
        .map(|c| {
            let v: Vec<Hv> = tr.iter().filter(|(l, _)| *l == c).map(|(_, h)| h.clone()).collect();
            if v.is_empty() { Hv::zero() } else { Hv::bundle(&v) }
        })
        .collect();

    let mut out = BufWriter::new(std::fs::File::create(out_p).expect("create output"));
    let t0 = std::time::Instant::now();
    for (_, s) in test.iter() {
        let q = encode(s, &toks, n);
        let (mut best, mut bl) = (u32::MAX, 0usize);
        for (l, h) in tr.iter() {
            let d = h.hamming(&q);
            if d < best {
                best = d;
                bl = *l;
            }
        }
        let cd: Vec<String> = centroids.iter().map(|c| c.hamming(&q).to_string()).collect();
        writeln!(out, "{} {} {}", bl, best, cd.join(" ")).expect("write");
    }
    let per = t0.elapsed().as_secs_f64() / test.len() as f64 * 1e6;
    eprintln!(
        "n={} vocab={} train={} test={} classes={} : {:.0} us per test sequence (encode + {} comparisons)",
        n,
        vocab,
        tr.len(),
        test.len(),
        nclass,
        per,
        tr.len()
    );
}
