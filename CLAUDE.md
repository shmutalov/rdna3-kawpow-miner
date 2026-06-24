# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An RDNA3-optimized **KawPow** (ProgPoW 0.9.3 / Ravencoin) GPU miner: a **Python host** that orchestrates **Vulkan compute (SPIR-V) shaders**. All hashing runs on the GPU; the Python host only sets up Vulkan, generates/compiles the per-period shader, manages buffers, and dispatches. It is a focused re-implementation of [kawpowminer](https://github.com/RavenCommunity/kawpowminer), carrying over only the KawPow algorithm and replacing the C++/OpenCL/CUDA backends with Python + Vulkan.

## Commands

```bash
pip install -r requirements.txt          # vulkan, numpy, pycryptodome

# Run / operate
python -m rdna3_kawpow --list-devices     # identify GPU + RDNA3 features
python -m rdna3_kawpow --benchmark        # hashrate only (no pool, uninitialized DAG)
python -m rdna3_kawpow stratum+tcp://WALLET.WORKER:x@pool.host:port   # mine
python -m rdna3_kawpow --variant vanilla ...     # ProgPoW 0.9.2 instead of KawPow

# Tests (each file runs standalone via __main__; they are also pytest-discoverable,
# but pytest is NOT a declared dependency — prefer running the file directly)
python tests/test_reference_vector.py    # CPU reference vs canonical vector  (no GPU)
python tests/test_stratum.py             # stratum/CLI pure logic             (no GPU)
python tests/test_gpu_search.py          # GPU search shader == CPU reference (needs GPU)
python tests/test_gpu_dag.py             # GPU DAG-gen == ethash              (needs GPU)
```

**External requirements not pip-installable:** the Vulkan SDK (provides `glslc` for runtime GLSL→SPIR-V and the loader; set `VULKAN_SDK` env var or put `glslc` on PATH) and an AMD RDNA3 GPU (developed on RX 7900 XT / gfx1100). Development/validation is on **Windows**.

## Architecture

### Single source of truth: the ProgPoW IR (most important concept)

[rdna3_kawpow/progpow.py](rdna3_kawpow/progpow.py) builds the per-period random "program" as a **backend-neutral list of op tuples** (`build_program`). That same IR is then:
- **rendered to GLSL/CUDA/OpenCL text** (`render_loop`) for the GPU shader, and
- **executed directly in Python** (`run_loop`) for the CPU reference.

So the GPU and the CPU oracle run the **identical** program by construction. When changing the algorithm, modify the IR and both consumers follow. The `kiss99()` draw order in `build_program` mirrors upstream `ProgPow.cpp` exactly — **do not reorder draws** or the program diverges.

### Correctness chain (how GPU output is trusted)

1. [tests/test_reference_vector.py](tests/test_reference_vector.py) pins the **CPU reference** ([reference.py](rdna3_kawpow/reference.py)) to the canonical ProgPoW 0.9.2 vector (block 30000) in [tests/vectors/](tests/vectors/) — keccak, program generator, ethash cache/DAG, fill_mix, 64-loop execution, digest reduction, final hash all checked.
2. [tests/test_gpu_search.py](tests/test_gpu_search.py) and [tests/test_gpu_dag.py](tests/test_gpu_dag.py) check that the **GPU shaders equal the CPU reference** on real hardware.

Therefore GPU correctness is established *transitively*: CPU == canonical vector, GPU == CPU. Preserve this chain when changing algorithm code.

### Module data flow

```
constants.py   ProgPoW/KawPow params + keccak tables
keccak.py      keccak-f800 (seed/final) + original Keccak-256/512 (ethash)
progpow.py     IR generator + GLSL/CUDA/CL renderer + Python interpreter  (source of truth)
ethash.py      host light-cache; serves DAG rows; GPU generates the full multi-GB DAG
dagcache.py    on-disk cache of the per-epoch light cache + full DAG (instant restarts)
reference.py   pure-Python KawPow hash (CPU oracle); consumes progpow IR + keccak + ethash
shaders/       progpow_search.comp.tmpl (search) + ethash_dag.comp (DAG gen)
shader_compiler.py   template + rendered IR -> glslc -> SPIR-V (cached on disk by content hash)
vkhost.py      Vulkan device/buffer/pipeline/dispatch (forces wave32)
miner.py       orchestration: per-epoch DAG, per-period kernel, per-batch search
stratum.py     pool client (mining.subscribe/authorize/notify/submit)
__main__.py    CLI + main mining loop
```

Mining lifecycle (`miner.py` + `__main__.py:mine`):
- **per-epoch** (every 7500 blocks for KawPow): host builds the light cache, GPU generates the full DAG. Both are cached to disk by [dagcache.py](rdna3_kawpow/dagcache.py) keyed by `(variant, epoch, parents, seed)`, so a restart within an epoch reloads them (light cache read + DAG streamed back to the GPU via a staging buffer) instead of rebuilding. Caching is best-effort and disabled with `--no-dag-cache`.
- **per-period** (every `PROGPOW_PERIOD`=3 blocks): regenerate + recompile the search kernel for the new program seed.
- **per-batch**: dispatch the search over a nonce range, collect solutions.

### Shader assembly by text substitution

[shaders/progpow_search.comp.tmpl](rdna3_kawpow/shaders/progpow_search.comp.tmpl) has placeholders the host fills at runtime: `__DEFINES__` (epoch sizing, `MAX_OUTPUTS`), `__PROGPOW_LOOP__` (the rendered IR `progPowLoop`), `__TAIL_INIT__` / `__FINAL_FILL__` (variant-specific keccak layout). `shader_compiler.py` substitutes these, runs `glslc`, and caches the SPIR-V keyed by source content hash under the system tempdir (`rdna3_kawpow_spv/`).

### RDNA3 (gfx11) specifics

- Pipeline forces `requiredSubgroupSize = 32` (native **wave32**) via `VK_EXT_subgroup_size_control` ([vkhost.py](rdna3_kawpow/vkhost.py) `ComputePipeline`). A 32-lane subgroup holds exactly two 16-lane KawPow hash groups.
- Cross-lane exchange (global-load broadcast, digest reduction) uses `subgroupShuffle` straight from registers — no LDS, no barrier (mirrors CUDA `__shfl`). This replaces the GCN-era shared-memory + `barrier()` pattern.

## Critical constraints & gotchas

- **GPU watchdog (TDR) safety.** Any single GPU dispatch longer than the OS watchdog (~2 s on Windows) is killed and can hang the machine. The miner therefore splits the DAG build and nonce search into many small, time-bounded dispatches and auto-calibrates a ~200 ms "safe batch" at startup (`WATCHDOG_TARGET_S`, `calibrate()`). **Do not raise `DAG_CHUNK_ITEMS` or the search batch past what calibration reports as safe.**

- **Two distinct Keccaks.** `keccak-f800` (25×32-bit lanes, 22 rounds) is hand-implemented in [keccak.py](rdna3_kawpow/keccak.py) for the ProgPoW seed/final hash. The ethash DAG uses **original Keccak-256/512** (keccak-f1600, `0x01` padding) via `pycryptodome`'s `Crypto.Hash.keccak` — **NOT** `hashlib.sha3_*`, which is NIST SHA3 with `0x06` padding and would silently produce wrong DAGs.

- **KawPow vs vanilla variant** (`keccak.KAWPOW` / `keccak.VANILLA`) is threaded through everywhere. KawPow absorbs the `RAVENCOIN_RNDC` constants into the keccak tail, uses 11/18 cache/math op counts, and a 7500-block epoch. Vanilla ProgPoW 0.9.2 (the carried test vector) uses a zero tail, 12/20 counts, and 30000-block epoch. Same code paths, different configuration — production parity ultimately needs confirmation from live pool shares.

- **Endianness of the seed.** ProgPoW reads the keccak seed big-endian: `seed0 = bswap(state2[1])`, `seed1 = bswap(state2[0])`. This bswap is replicated in both the reference and the shader — keep them in lockstep.

- **DAG buffer layout.** In GLSL the DAG is a flat `uint[]` SSBO; `dag_t` row `r` occupies words `[r*4 .. r*4+3]`. `PROGPOW_DAG_ELEMENTS = (full_size / MIX_BYTES) / 2`; each 64-byte ethash dataset item == 4 `dag_t` rows. `ethash.LightCache` must hand the kernel exactly these sizings.

- The host language genuinely does not cost hashrate: 100% of hashing is SPIR-V on the GPU. Keep host work to setup/dispatch and keep the GPU saturated with adequately-sized (but watchdog-safe) batches.
