# rdna3-kawpow

An **RDNA3-optimized KawPow** (ProgPoW 0.9.3 / Ravencoin) GPU miner.
**Python host orchestration + Vulkan compute (SPIR-V) shaders**, tuned for AMD
RDNA3 (gfx11, e.g. Radeon RX 7900 XT).

This is a focused fork of [kawpowminer](https://github.com/RavenCommunity/kawpowminer):
only the KawPow-relevant algorithm is carried over, re-implemented as a Python
host driving Vulkan compute shaders, replacing the original C++/OpenCL/CUDA
backends. No Boost / Hunter / C++ SDK build is required.

All KawPow hashing runs on the GPU as a SPIR-V compute shader. The host only sets
up Vulkan, (re)compiles the per-period shader, manages buffers, dispatches over
nonce batches, and reads back a small result buffer.

## RDNA3-specific optimizations

The upstream OpenCL kernel was written for GCN-era 64-wide wavefronts and treats
all AMD GPUs identically. This port targets RDNA3 directly:

| Optimization | How | Upstream (GCN) approach |
|---|---|---|
| **Native wave32** | Pipeline forces `requiredSubgroupSize = 32` via `VK_EXT_subgroup_size_control`; a 32-lane subgroup holds exactly two 16-lane KawPow hash groups | Implicit wave64; work-group sizing tuned to a hard-coded 36-CU (Polaris) baseline |
| **Register cross-lane exchange** | `subgroupShuffle` for the global-load broadcast and the digest reduction — straight from VGPRs, no LDS, no barrier (mirrors CUDA `__shfl`) | `share[]` in LDS + `barrier(CLK_LOCAL_MEM_FENCE)` |
| **CU-aware sizing** | `VK_AMD_shader_core_properties2` reports real CU count for dispatch sizing | Linear scale from the 36-CU constant |
| **Device-coherent abort** | abort flag in coherent host-visible memory for low-latency job switching | LDS/global flag |

Validated on the target hardware: a probe forces wave32 and runs `subgroupShuffle`
on an RX 7900 XT and verifies the shader observes subgroup size 32 with correct
shuffle results.

## Architecture

```
rdna3_kawpow/
  constants.py        ProgPoW/KawPow params + keccak tables
  keccak.py           keccak-f800 (seed/final) + original Keccak-256/512 (ethash)
  progpow.py          ProgPoW program generator as backend-neutral IR ->
                        renders GLSL/CUDA/CL AND executes in Python (one source
                        of truth, so GPU and CPU reference run the same program)
  ethash.py           light-cache + dataset (DAG) generation; serves DAG rows
  dagcache.py         on-disk cache of the per-epoch light cache + DAG
  reference.py        pure-Python KawPow hash (CPU correctness oracle)
  shaders/
    progpow_search.comp.tmpl   search shader (keccak + fill_mix + injected loop +
                                subgroupShuffle reductions); RDNA3 wave32
    ethash_dag.comp            GPU DAG generation (keccak-f1600 / Keccak-512)
  shader_compiler.py  assembles template + generated loop -> glslc -> SPIR-V (cached)
  vkhost.py           Vulkan device/memory/pipeline/dispatch
  miner.py            epoch DAG gen + per-period kernel + search loop
  stratum.py          pool client
  __main__.py         CLI
```

The host generates the GLSL search shader for each ProgPoW period (every 3 blocks),
compiles it to SPIR-V at runtime with `glslc`, and dispatches it; the multi-GB DAG
is generated on the GPU once per epoch from a host-computed light cache. The light
cache and DAG are cached to disk, so a restart within an epoch reloads them instead
of rebuilding (disable with `--no-dag-cache`).

## Validation status

The algorithm is validated **end-to-end** against the canonical ProgPoW test
vector (block 30000, header `ffeeddcc…ddeeff`, nonce `123456789abcdef0`) carried
in `tests/vectors/` — every intermediate value matches:

- ✅ `keccak_f800` seed
- ✅ ProgPoW program generator — byte-for-byte vs `kernel.cu` (prog_seed 600)
- ✅ ethash light cache + DAG: `cache_size`, `full_size`, `dag_elements`, and DAG
  cache words all match
- ✅ `fill_mix` (all lanes), all 64 per-loop DAG entries, all 16 per-lane digests
- ✅ 256-bit mix digest `11f19805…767d` and 64-bit result `5b7ccd472dbefdd9`
- ✅ All three GPU shaders compile to SPIR-V (kawpow search, vanilla search, DAG gen)
- ✅ Vulkan wave32 + `subgroupShuffle` verified on RX 7900 XT
- ✅ **Production KawPow == authoritative cpp-kawpow vectors** (blocks 0/49/50/99,
  mix hash + result) — `tests/test_kawpow_vectors.py`
- ✅ **GPU search shader == CPU reference on RX 7900 XT** (0 mismatches) —
  `tests/test_gpu_search.py`
- ✅ **GPU DAG generation == ethash/reference** on RX 7900 XT — `tests/test_gpu_dag.py`
- ✅ Shares accepted on the Gaelium (GAEL) KawPow pool at ~44–48 MH/s on the
  RX 7900 XT.

The correctness chain is closed end-to-end: CPU reference == cpp-kawpow vectors,
GPU == CPU reference on hardware, and the live pool accepts the resulting shares.

### Two non-obvious implementation details

1. **Standard Ravencoin KawPow uses 512 DAG-item parents, not the classic ethash
   256.** This changes the entire DAG and is the difference between "low
   difficulty" rejects and accepted shares. (The bundled Zing `result.log` uses
   256 — a different coin's DAG — which initially masked this.) See
   `ethash.DATASET_PARENTS`.
2. **Nonce submission (this pool):** the full 16-hex nonce, no `0x` prefix, with
   the extranonce in its high bits; the search must stay inside the miner's nonce
   region so every nonce still starts with the extranonce. A keepalive holds the
   connection open during the (idle) DAG build.

> ⚠️ **GPU watchdog (TDR) safety.** A single GPU dispatch that runs longer than
> the OS watchdog (~2 s on Windows) is killed and can hang the machine. The miner
> therefore splits the DAG build and the nonce search into small, time-bounded
> dispatches and auto-calibrates a safe batch (~200 ms) at startup. Do not raise
> `DAG_CHUNK_ITEMS` / batch sizes past what calibration reports as safe.

Run the regression test:

```bash
python tests/test_reference_vector.py
```

> The carried vector is **vanilla ProgPoW 0.9.2** (CNT_CACHE=12, CNT_MATH=20,
> period 50, zero keccak tail, 30000 epoch). Production **KawPow** differs only by
> configuration — Ravencoin keccak constants, 11/18 op counts, 7500 epoch — driven
> through the same validated code paths. Production parity is confirmed by live
> pool shares (and ideally a KawPow-specific vector); see "Remaining work".

### Remaining / nice-to-have

- **Hashrate** — ~49 MH/s on the RX 7900 XT (workgroup size tunable via
  `--local-size`; 256 is the gfx1100 default). That is at or above mature-miner
  figures for this card (~46 MH/s per minerstat); the larger RX 7900 **XTX**
  reaches ~58 MH/s. Occupancy is already saturated, so further gains would need
  ISA-level register/scheduling tuning not reachable through GLSL → SPIR-V.
- Other pools may use different stratum conventions (the nonce format above is
  what Gaelium's pool expects); a different pool may need a tweak.

## Usage

**Windows quick start:** just double-click **`mine.bat`** (pool/wallet are baked in
and editable at the top of the file; it installs deps on first run and
auto-restarts on exit). **`benchmark.bat`** measures hashrate without a pool.

```bash
pip install -r requirements.txt

# Identify the GPU and its RDNA3 features
python -m rdna3_kawpow --list-devices

# Measure hashrate (no pool; uninitialized DAG; watchdog-safe)
python -m rdna3_kawpow --benchmark

# Mine against a pool
python -m rdna3_kawpow stratum+tcp://YOUR_RVN_ADDRESS.worker:x@pool.host:port

# Run the test suite
python tests/test_reference_vector.py   # algorithm vs canonical vector (CPU)
python tests/test_stratum.py            # stratum/CLI pure logic
python tests/test_dagcache.py           # light cache + DAG disk cache (CPU)
python tests/test_gpu_search.py         # GPU search == CPU reference (needs GPU)
python tests/test_gpu_dag.py            # GPU DAG-gen == ethash (needs GPU)
```

## Requirements

- AMD RDNA3 GPU + recent driver (developed against RX 7900 XT / gfx1100)
- Vulkan SDK (for `glslc` and the loader) — set `VULKAN_SDK` or put `glslc` on PATH
- Python ≥ 3.9 and `pip install -r requirements.txt`

## License

GPL-3.0-or-later (inherited from kawpowminer). See `LICENSE`.
