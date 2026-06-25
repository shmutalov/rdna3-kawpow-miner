# rdna3-kawpow

A KawPow (ProgPoW 0.9.3 — used by Ravencoin and other coins) GPU miner for AMD
RDNA3 (gfx11, e.g. Radeon RX 7900 XT). The host is written in Rust; all hashing
runs on the GPU as Vulkan compute (SPIR-V) shaders. The host selects the Vulkan
device, generates and compiles the per-period shader, manages buffers, dispatches
the nonce search, and talks to the pool.

It carries the KawPow algorithm from
[kawpowminer](https://github.com/RavenCommunity/kawpowminer) and replaces the
C++/OpenCL/CUDA backends with a Rust host driving Vulkan shaders. A Python
implementation (`rdna3_kawpow/`) is retained as the CPU reference oracle the Rust
port is validated against; the GLSL shaders in `rdna3_kawpow/shaders/` are shared
by both.

## Repository layout

```
rust/                 Rust miner (primary)
  src/                constants, keccak, ethash, progpow (IR + GLSL renderer +
                      interpreter), reference, vkhost (Vulkan), shader_compiler,
                      dagcache, stratum, stats (HTTP JSON API), miner, main
  tests/              canonical vectors, Python differential, GPU tests
rdna3_kawpow/         Python reference implementation + shared shaders
  shaders/            progpow_search.comp.tmpl, ethash_dag.comp
hiveos/               HiveOS custom-miner package (h-manifest/config/run/stats)
docker/               Linux build (Dockerfile) + HiveOS package (Dockerfile.hiveos)
tests/                Python reference + GPU tests
```

## How it works

The per-period ProgPoW program is built as a backend-neutral intermediate
representation. The same IR is rendered to GLSL for the GPU shader and executed
directly on the CPU as the reference, so both run an identical program by
construction.

- Per epoch (7500 blocks for KawPow): the host computes a light cache and the GPU
  generates the multi-GB DAG from it. Both are cached to disk, so a restart within
  an epoch reloads them in well under a second (disable with `--no-dag-cache`).
- Per period (3 blocks): the search shader is regenerated and recompiled to SPIR-V.
- Per batch: the search is dispatched over a nonce range and solutions are read back.

## RDNA3 specifics

- The compute pipeline pins `requiredSubgroupSize = 32` (native wave32) via
  `VK_EXT_subgroup_size_control`; a 32-lane subgroup holds two 16-lane KawPow hash
  groups.
- Cross-lane exchange (global-load broadcast, digest reduction) uses
  `subgroupShuffle` from registers — no shared memory, no barrier.
- Compute-unit count is read from `VK_AMD_shader_core_properties2`.
- PCI bus id is read from `VK_EXT_pci_bus_info` for HiveOS per-GPU mapping.

## Build

Needs a Rust toolchain:

```
cargo build --release --manifest-path rust/Cargo.toml
```

At runtime the binary needs a GLSL→SPIR-V compiler — `glslc` (Vulkan SDK) or
`glslangValidator` (`glslang-tools`) — found on `PATH`, in the binary's own
directory, or under `$VULKAN_SDK/Bin`. `RDNA3_KAWPOW_COMPILER=glslc|glslangValidator`
forces a choice.

On Windows, `mine-rust.bat` builds on first run and launches the miner with an
editable pool/wallet block at the top of the file.

## Usage

```
rdna3-kawpow --list-devices                 # GPUs + RDNA3 features + PCI bus
rdna3-kawpow --device-count                 # number of discrete GPUs
rdna3-kawpow --benchmark                    # hashrate only (no pool)
rdna3-kawpow -a kawpow -o stratum+tcp://HOST:PORT -u WALLET.WORKER -p PASS
rdna3-kawpow stratum+tcp://WALLET.WORKER:PASS@HOST:PORT
```

Flags: `--device N`, `--local-size N` (wave32 multiple, default 256),
`--epoch-length N`, `--variant kawpow|vanilla`, `--no-recheck`, `--no-dag-cache`,
`--cache-dir DIR`, `--api-bind ADDR`.

`--api-bind 127.0.0.1:4068` serves a JSON snapshot (hashrate, accepted/rejected/
invalid shares, uptime, device name, PCI bus id) that the HiveOS stats script reads.

## HiveOS

`hiveos/` and `docker/Dockerfile.hiveos` build a self-contained custom-miner
package (binary + bundled `glslangValidator` + the four `h-*` scripts):

```
docker build -f docker/Dockerfile.hiveos --target export -o type=local,dest=dist .
```

This writes `dist/rdna3kawpow/`. The package launches one process per GPU and
aggregates per-GPU stats. It is built on Debian bullseye, so the binary and the
bundled `glslangValidator` require glibc ≤ 2.31 (HiveOS is Ubuntu 22.04, glibc
2.35). See [hiveos/README.md](hiveos/README.md) for install steps and flight-sheet
field mapping.

## Correctness

The algorithm is validated as a chain:

- The CPU reference reproduces the canonical ProgPoW 0.9.2 vector (block 30000) and
  the cpp-kawpow production vectors — `rust/tests/vectors.rs`.
- The Rust program IR and GLSL renderer match the Python reference byte-for-byte
  across many program seeds — `rust/tests/ir_differential.rs`.
- The GPU shaders match the CPU reference on an RX 7900 XT, for both `glslc` and
  `glslangValidator` output — `rust/tests/gpu_search.rs`; the disk-cached DAG
  reproduces the same hashes — `rust/tests/gpu_dag_cache.rs`.
- Live: shares accepted on a KawPow pool with 0 host-rejected (invalid) solutions.

Run the tests (GPU tests are marked `#[ignore]`; pass `--ignored` to run them on
hardware):

```
cargo test --release --manifest-path rust/Cargo.toml
cargo test --release --manifest-path rust/Cargo.toml -- --ignored   # needs a GPU
```

Measured hashrate: ~49 MH/s on the RX 7900 XT via `--benchmark`.

The Python reference and its tests run separately:

```
python tests/test_reference_vector.py   # CPU reference vs canonical vector
python tests/test_kawpow_vectors.py     # CPU reference vs cpp-kawpow vectors
```

## Notes

- Two distinct Keccaks: keccak-f800 (32-bit lanes, 22 rounds) for the ProgPoW
  seed/final hash, and original Keccak-256/512 (0x01 padding) for the ethash DAG —
  not NIST SHA3 (0x06 padding), which would produce wrong DAGs.
- Standard Ravencoin KawPow uses 512 DAG-item parents (classic ethash uses 256).
  This changes the DAG contents and must match the target coin.
- Endianness: the keccak seed is read big-endian for the 0.9.2 (vanilla) variant;
  KawPow reads it directly. The reference and the shader follow the same rule.
- GPU watchdog (TDR): a single GPU dispatch longer than the OS watchdog (~2 s on
  Windows) is killed and can hang the machine. The miner splits the DAG build and
  the nonce search into small, time-bounded dispatches and calibrates a ~200 ms
  batch at startup. Do not raise batch sizes past what calibration reports as safe.

## Requirements

- AMD RDNA3 GPU (gfx11) with `VK_EXT_subgroup_size_control` (wave32) and a Vulkan
  driver. Developed on an RX 7900 XT.
- A Rust toolchain to build; `glslc` or `glslangValidator` available at runtime.
- Optional: Python ≥ 3.9 for the reference implementation and its tests
  (`pip install -r requirements.txt`).

## License

GPL-3.0-or-later (inherited from kawpowminer). See `LICENSE`.
