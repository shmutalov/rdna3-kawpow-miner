# rdna3-kawpow on HiveOS

A self-contained HiveOS custom-miner package: the Rust KawPow miner binary, a
bundled `glslangValidator` (runtime SPIR-V compiler), and the four HiveOS
integration scripts. No dependencies are installed on the rig.

## Build the package

From the repo root (needs Docker):

```bash
docker build -f docker/Dockerfile.hiveos --target export -o type=local,dest=dist .
```

This produces `dist/rdna3kawpow/`:

```
rdna3kawpow/
  rdna3-kawpow        # the miner (glibc <= 2.31, runs on HiveOS 22.04+)
  glslangValidator    # bundled SPIR-V compiler (found via the binary's own dir)
  h-manifest.conf     # package metadata
  h-config.sh         # flight sheet -> runtime config
  h-run.sh            # launches one process per GPU
  h-stats.sh          # aggregates per-GPU JSON APIs -> HiveOS stats
  README.md
```

The build prints the max glibc each binary actually requires (both are <= 2.31).

## Install on the rig

**Option A — HiveOS UI (recommended):** zip the *contents* of `dist/rdna3kawpow/`
into `rdna3kawpow.tar.gz`, then in HiveOS create a **Flight Sheet** with a **custom
miner**, **Miner name = `rdna3kawpow`**, and use **Setup Miner Config → Install
the miner from a custom source** to upload the archive (or host it and give the
URL). HiveOS unpacks it to `/hive/miners/custom/rdna3kawpow/`.

**Option B — manual:** copy `dist/rdna3kawpow/` to
`/hive/miners/custom/rdna3kawpow/` on the rig and `chmod +x *.sh rdna3-kawpow
glslangValidator`.

## Flight Sheet fields

| HiveOS field                 | Maps to                                  |
|------------------------------|------------------------------------------|
| Pool URL                     | `-o` (e.g. `stratum+tcp://host:port`)    |
| Wallet and worker template   | `-u` (e.g. `WALLET.%WORKER_NAME%`)       |
| Pass                         | `-p`                                     |
| Hash algorithm               | `kawpow`                                 |
| Extra config arguments       | appended verbatim (e.g. `--local-size 128 --no-recheck`) |

Example (Gaelium): Pool `stratum+tcp://pool.gaelium.io:3638`, Wallet
`GQ...M6u.%WORKER_NAME%`, Pass `c=GAEL,mc=GAEL`.

## Multi-GPU

`h-run.sh` launches **one process per GPU** (`--device i`), each serving its stats
on `127.0.0.1:(4068+i)`. `h-stats.sh` queries all of them and reports the per-GPU
hashrate array plus summed accepted/rejected/invalid shares. If any GPU process
exits, the whole set is restarted so the rig never runs half-dead.

The first start of each epoch builds + caches the DAG to disk
(`~/.cache/rdna3_kawpow/dagcache` or `$RDNA3_KAWPOW_CACHE_DIR`); restarts within an
epoch reload it in well under a second.

## Notes / limitations

- temps, fans and `bus_numbers` are filled by the HiveOS agent from `gpu-stats`;
  the miner reports only hashrate + shares. On **heterogeneous** rigs the per-GPU
  hashrate array is in Vulkan device order, which may not match HiveOS slot order
  — a future refinement is to report each GPU's PCI bus id (via
  `VK_EXT_pci_bus_info`) in the stats JSON so HiveOS maps them exactly.
- Requires AMD RDNA3 (gfx11) with `VK_EXT_subgroup_size_control` (wave32). HiveOS
  ships the AMD Vulkan stack; no driver install needed.
