# nimblecube-esp32

ESP32-S3 firmware (esp-hal, `no_std`) that runs `nimblecube-core` on real silicon. Standalone crate (its
own workspace; targets `xtensa-esp32s3-none-elf`).

## Binaries

| `--bin` | what it does |
|---|---|
| `nimblecube-esp32` | bring-up: boots, runs `FeatureEncoder` + `FixedStore` once, prints the result over serial |
| `sensor_read` | reads an analog sensor on ADC1 (GPIO3) and prints the raw value (the "is my wiring right?" tool) |
| **`gas_anomaly`** | the demo: windowed features → encode → auto-enroll "clean air" → detect → `ok` / `ANOMALY: smoke/gas` |
| `bench` | on-chip timing of `encode` / `hamming` / `nearest` |
| `psram_bench` | wires the 8 MB octal PSRAM as a heap and benchmarks a 12k-vector store |

## Toolchain

Install Espressif's Rust toolchain with [`espup`](https://github.com/esp-rs/espup), then source its
environment in your shell (the path is printed by `espup install`):

```bash
espup install
. ~/export-esp.sh          # adjust to wherever espup wrote it
```

`rust-toolchain.toml` pins the `esp` channel; `.cargo/config.toml` sets the target and link args.

## Build

```bash
cargo build --release --bin gas_anomaly
```

(No board needed to compile; it links to an `xtensa-esp32s3-none-elf` ELF.)

## Wire up the MQ-2 (for `gas_anomaly` / `sensor_read`)

| MQ-2 pin | ESP32-S3 |
|---|---|
| VCC | **3.3 V**: keeps `AOUT` ≤ 3.3 V, safe for the ADC |
| GND | GND |
| AOUT | **GPIO3** (ADC1) |
| DOUT | *(unused)* |

3.3 V slightly under-drives the heater but the **relative** clean-air → smoke change is all the detector
needs. (For full sensitivity: 5 V supply + a voltage divider on `AOUT` to keep it under 3.3 V.) The MQ-2
needs **~30-60 s warmup** before readings settle. Trigger it with any small smoke source (e.g. incense or a
just-extinguished flame) or breath, and watch the reading rise.

## Flash + monitor

Install [`espflash`](https://github.com/esp-rs/espflash), then:

```bash
cargo run --release --bin gas_anomaly       # builds, flashes, opens the serial monitor
```

The serial port (e.g. `/dev/ttyACM0` or `/dev/ttyUSB0`) must be accessible; on Linux add yourself to
the `dialout` group (`sudo usermod -aG dialout $USER`, then re-login). If your board exposes the native
USB-Serial-JTAG port rather than a UART bridge, build with `esp-println`'s `jtag-serial` feature instead
of `uart` (or vice-versa).

### Expected

- `sensor_read`: `raw=` jumps up when smoke reaches the sensor.
- `gas_anomaly`: `enroll …` for 8 clean-air windows after warmup, then `baseline level=…`, then
  `mean=… rel=[…] d=… ok` in clean air and `… ANOMALY: smoke/gas` under sustained smoke. `rel` is the
  encoded features in per mille of this unit's clean-air level (level over clean air, peak over the
  window mean, slope), so a recorded signature means the same on another board. Tune the threshold margin / persistence (`K`, `M`) in the source if it's too
  twitchy or too slow.

See [`../BENCHMARKS.md`](../BENCHMARKS.md) for measured on-chip numbers.
