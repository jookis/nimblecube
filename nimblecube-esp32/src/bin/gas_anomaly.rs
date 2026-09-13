#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::analog::adc::{Adc, AdcConfig, Attenuation};
use esp_hal::clock::CpuClock;
use esp_hal::delay::Delay;
use esp_hal::main;
use esp_println::println;

use nimblecube_core::encode::FeatureEncoder;
use nimblecube_core::features::{relative, Baseline, WindowStats, RELATIVE_RANGES};
use nimblecube_core::store::FixedStore;

esp_bootloader_esp_idf::esp_app_desc!();

const W: usize = 32; // readings per window (~32 * 10 ms)
const WARMUP: usize = 30; // windows skipped for MQ-2 heater settling
const BASELINE: usize = 8; // clean-air windows enrolled as "normal"
const STORE: usize = 8;
const K: usize = 3; // persistence: alarm if >= K of the last M windows are over threshold
const M: usize = 5;

#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);

    let mut adc_config = AdcConfig::new();
    let mut adc_pin = adc_config.enable_pin(peripherals.GPIO3, Attenuation::_11dB);
    let mut adc = Adc::new(peripherals.ADC1, adc_config);
    let delay = Delay::new();

    // Features: level over clean air, peak over the window mean, slope; all per mille of
    // this unit's own clean-air level, so signatures carry across units (see features.rs).
    let enc = FeatureEncoder::<3, 16>::new(7, RELATIVE_RANGES);
    let mut store: FixedStore<STORE> = FixedStore::new();
    let mut clean = [WindowStats::ZERO; BASELINE];
    let mut baseline = Baseline::new();
    let mut level: i32 = 0;

    println!("mq2 gas_anomaly: warming up...");

    let mut t: usize = 0;
    let mut threshold: u32 = 0;
    let mut max_baseline: u32 = 0;
    let mut recent_over = [false; M];
    let mut ri = 0usize;

    loop {
        // --- sample one window ---
        let mut window = [0i32; W];
        for w in window.iter_mut() {
            let raw: u16 = nb::block!(adc.read_oneshot(&mut adc_pin)).unwrap();
            *w = raw as i32;
            delay.delay_millis(10);
        }
        let s = WindowStats::of(&window);

        if t < WARMUP {
            // MQ-2 heater settling - ignore.
        } else if t < WARMUP + BASELINE {
            // Collect clean air; it can only be encoded once the baseline level is known.
            clean[t - WARMUP] = s;
            baseline.add(&s);
            println!("enroll mean={}", s.mean);
        } else {
            if threshold == 0 {
                // Enroll the clean-air windows against this unit's level; track the worst spread.
                level = baseline.level().unwrap_or(1);
                for (i, c) in clean.iter().enumerate() {
                    let hv = enc.encode(&relative(c, level));
                    if !store.is_empty() {
                        let d = store.nearest(&hv).unwrap().1;
                        if d > max_baseline {
                            max_baseline = d;
                        }
                    }
                    store.insert(hv, i as u32).ok();
                }
                threshold = max_baseline + 70;
                println!("baseline level={} ceiling d={} -> threshold={}", level, max_baseline,
                    threshold);
            }
            let f = relative(&s, level);
            let d = store.nearest(&enc.encode(&f)).unwrap().1;
            recent_over[ri % M] = d > threshold;
            ri += 1;
            let count = recent_over.iter().filter(|&&b| b).count();
            if count >= K {
                println!("mean={} rel={:?} d={} ANOMALY: smoke/gas", s.mean, f, d);
            } else {
                println!("mean={} rel={:?} d={} ok", s.mean, f, d);
            }
        }
        t += 1;
    }
}
