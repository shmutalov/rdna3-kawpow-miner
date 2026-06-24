"""KawPow miner orchestration on top of the Vulkan host.

Responsibilities:
  * per-epoch: build the light cache (host) and generate the full DAG on the GPU
    (chunked, to stay under the OS GPU watchdog),
  * per-period: regenerate + compile the ProgPoW search kernel,
  * per-batch: dispatch the search over a nonce range and collect solutions.

All hashing is on the GPU; this code only feeds it.
"""

import struct
import time

from . import dagcache, ethash, keccak, shader_compiler as sc
from .constants import PROGPOW_PERIOD, KAWPOW_EPOCH_LENGTH
from .vkhost import VulkanDevice, ComputePipeline

MAX_OUTPUTS = 16
SEARCH_LOCAL_SIZE = 128          # multiple of 16 lanes and of wave32
DAG_LOCAL_SIZE = 64
# Host-visible staging window for streaming the device-local DAG to/from disk.
DAG_STAGING_BYTES = 64 << 20     # 64 MiB
# IMPORTANT: keep every single GPU dispatch well under the OS GPU watchdog (TDR,
# ~2 s on Windows). A too-large dispatch is killed and can hang the machine. These
# are deliberately small; the miner splits big nonce ranges / the DAG into many
# such chunks. The search auto-calibrates a safe batch at startup.
DAG_CHUNK_ITEMS = 1 << 15        # ~32k items per DAG dispatch
SEARCH_CHUNK_NONCES = 1 << 20    # default safe nonce batch (auto-tuned by calibrate())
WATCHDOG_TARGET_S = 0.20         # aim for ~200 ms dispatches, far below the TDR limit


class Solution:
    def __init__(self, nonce, mix_words):
        self.nonce = nonce
        self.mix_words = mix_words            # 8 "my-domain" digest words

    def mix_hash(self):
        """Canonical 32-byte mix hash (big-endian word read)."""
        return struct.pack(">8I", *[keccak._swab32(w) for w in self.mix_words])


class VulkanMiner:
    def __init__(self, device_index=None, variant=keccak.KAWPOW,
                 epoch_length=KAWPOW_EPOCH_LENGTH, dag_cache=True, cache_dir=None):
        self.dev = VulkanDevice(device_index)
        self.variant = variant
        self.epoch_length = epoch_length
        # Disk cache of the light cache + full DAG, so restarts within an epoch
        # skip both the host mkcache and the GPU DAG build.
        self._cache = dagcache.Cache(cache_dir) if dag_cache else None
        # Standard KawPow uses 512 DAG parents; the 0.9.2 reference uses 256.
        self.parents = 512 if variant == keccak.KAWPOW else 256
        self._epoch = None
        self._light = None
        self._dag_buf = None
        self._light_buf = None
        self._dag_elements = None
        self._period = None
        self._search_pipe = None
        self._safe_batch = SEARCH_CHUNK_NONCES
        # Persistent small buffers.
        self._hdr_buf = self.dev.make_buffer(32)
        self._out_size = 16 + MAX_OUTPUTS * 9 * 4
        self._out_buf = self.dev.make_buffer(self._out_size)

    # --- per-epoch DAG ---
    def ensure_epoch(self, block, seed=None, log=print):
        epoch = block // self.epoch_length
        if epoch == self._epoch and self._dag_buf is not None and self._light is not None:
            return
        eff_seed = seed if seed is not None else ethash.seed_hash(epoch)
        cache = self._cache

        # 1) Light cache: reload from disk if present (and still needed for the
        #    host solution re-check), otherwise build it and cache it.
        light_bytes = cache.load_light(self.variant, epoch, self.parents,
                                       eff_seed) if cache else None
        self._light = None
        if light_bytes is not None:
            t0 = time.time()
            try:
                self._light = ethash.LightCache(
                    block, self.epoch_length, seed=eff_seed,
                    parents=self.parents, precomputed_cache=light_bytes)
                log(f"Epoch {epoch}: loaded {len(light_bytes)//(1<<20)} MiB light "
                    f"cache from disk in {time.time()-t0:.1f}s")
            except ValueError as e:
                log(f"Epoch {epoch}: cached light cache rejected ({e}); rebuilding")
                light_bytes = None
        if self._light is None:
            t0 = time.time()
            log(f"Epoch {epoch}: building light cache (host)...")
            self._light = ethash.LightCache(block, self.epoch_length, seed=eff_seed,
                                            parents=self.parents)
            light_bytes = b"".join(self._light.cache)
            log(f"  light={len(light_bytes)//(1<<20)} MiB cache in {time.time()-t0:.1f}s")
            if cache:
                # Caching is a pure optimization: never let a write failure
                # (full disk, etc.) abort an otherwise-good epoch setup.
                try:
                    cache.save_light(self.variant, epoch, self.parents,
                                     eff_seed, light_bytes)
                except Exception as e:
                    log(f"  (light cache save skipped: {e})")

        full_size = self._light.full_size

        # (Re)allocate the device-local DAG buffer. transfer=True lets us stage it
        # to/from disk for caching.
        if self._dag_buf:
            self._dag_buf.destroy()
        self._dag_buf = self.dev.make_buffer(full_size, host_visible=False,
                                             transfer=True)

        # 2) DAG: reload from disk if present, else generate on the GPU and cache it.
        loaded = False
        if cache and cache.has_dag(self.variant, epoch, self.parents,
                                   eff_seed, full_size):
            try:
                self._load_dag_from_disk(cache, epoch, log)
                loaded = True
            except Exception as e:
                # Fall back to GPU generation, which fully overwrites the buffer.
                log(f"  DAG cache load failed ({e}); regenerating")
        if not loaded:
            self._generate_dag(light_bytes, log)
            if cache:
                # Best-effort: the DAG is already on the GPU and usable; a save
                # failure must not abort epoch setup (or it would loop forever).
                try:
                    self._save_dag_to_disk(cache, epoch, eff_seed, log)
                except Exception as e:
                    log(f"  (DAG cache save skipped: {e})")

        self._dag_elements = self._light.dag_elements
        self._epoch = epoch
        self._period = None  # force kernel recompile (dag_elements may change)

    def _generate_dag(self, light_bytes, log):
        """Generate the full DAG on the GPU from the light cache (watchdog-safe)."""
        log(f"  generating DAG={self._light.full_size//(1<<20)} MiB on GPU...")
        if self._light_buf:
            self._light_buf.destroy()
        self._light_buf = self.dev.make_buffer(len(light_bytes))
        self._light_buf.write(light_bytes)

        spv = sc.compile_dag(len(self._light.cache), self._light.dag_items, self.parents)
        dag_pipe = ComputePipeline(self.dev, spv, num_bindings=2, push_const_size=4,
                                   local_size=DAG_LOCAL_SIZE, required_subgroup_size=32)
        dag_pipe.bind([self._light_buf, self._dag_buf])

        # Adaptive, watchdog-safe DAG generation: start with tiny dispatches and
        # grow the chunk only while each stays fast, so no dispatch ever runs long
        # enough to trip the OS GPU watchdog (which can hang the machine).
        t1 = time.time()
        items = self._light.dag_items
        start = 0
        chunk = 1 << 12                       # 4096 items -- definitely safe
        last_log = t1
        while start < items:
            n = min(chunk, items - start)
            groups = (n + DAG_LOCAL_SIZE - 1) // DAG_LOCAL_SIZE
            t0 = time.time()
            self.dev.dispatch(dag_pipe, groups, struct.pack("<I", start))
            dt = time.time() - t0
            start += n
            if dt < 0.12 and chunk < (1 << 19):
                chunk <<= 1
            elif dt > 0.30 and chunk > (1 << 12):
                chunk >>= 1
            if time.time() - last_log > 3.0:
                log(f"  DAG {100.0*start/items:5.1f}%  ({n} items, {dt*1000:.0f} ms/dispatch)")
                last_log = time.time()
        log(f"  DAG generated in {time.time()-t1:.1f}s")
        # The light cache on the GPU is only needed during generation.
        self._light_buf.destroy()
        self._light_buf = None

    def _load_dag_from_disk(self, cache, epoch, log):
        """Upload a cached DAG into the device-local buffer via a staging window."""
        full = self._light.full_size
        t0 = time.time()
        reader = stage = None
        try:
            reader = cache.open_dag_read(self.variant, epoch, self.parents)
            stage = self.dev.make_buffer(min(DAG_STAGING_BYTES, full),
                                         host_visible=True, storage=False, transfer=True)
            pos = 0
            while pos < full:
                n = min(DAG_STAGING_BYTES, full - pos)
                data = reader.read(n)
                if len(data) != n:
                    raise RuntimeError("short read from DAG cache")
                stage.write(data)
                self.dev.copy_buffer(stage, self._dag_buf, n, 0, pos)
                pos += n
        finally:
            if reader is not None:
                reader.close()
            if stage is not None:
                stage.destroy()
        log(f"  loaded DAG={full//(1<<20)} MiB from disk in {time.time()-t0:.1f}s")

    def _save_dag_to_disk(self, cache, epoch, seed, log):
        """Stream the device-local DAG out to disk via a staging window."""
        full = self._light.full_size
        t0 = time.time()
        w = stage = None
        try:
            w = cache.open_dag_write(self.variant, epoch, self.parents, seed, full)
            stage = self.dev.make_buffer(min(DAG_STAGING_BYTES, full),
                                         host_visible=True, storage=False, transfer=True)
            pos = 0
            while pos < full:
                n = min(DAG_STAGING_BYTES, full - pos)
                self.dev.copy_buffer(self._dag_buf, stage, n, pos, 0)
                w.write(stage.read(n))
                pos += n
            w.commit()
        except BaseException:
            if w is not None:
                w.abort()
            raise
        finally:
            if stage is not None:
                stage.destroy()
        cache.prune_dag(self.variant)
        log(f"  saved DAG={full//(1<<20)} MiB to disk in {time.time()-t0:.1f}s")

    def setup_benchmark_dag(self, block, log=print):
        """Allocate an UNINITIALIZED full-size DAG buffer for throughput testing.

        Hashes are garbage but timing is representative (same memory footprint and
        access pattern). Avoids the multi-second DAG build entirely.
        """
        epoch, full_size, dag_items, dag_elements = ethash.dag_sizing(
            block, self.epoch_length)
        if self._dag_buf is None or self._epoch != epoch:
            if self._dag_buf:
                self._dag_buf.destroy()
            log(f"Allocating {full_size//(1<<20)} MiB device-local DAG buffer "
                f"(uninitialized, benchmark only)...")
            self._dag_buf = self.dev.make_buffer(full_size, host_visible=False)
        self._dag_elements = dag_elements
        self._epoch = epoch
        self._period = None

    # --- per-period kernel ---
    def ensure_period(self, block, log=print):
        period = block // PROGPOW_PERIOD
        if period == self._period and self._search_pipe is not None:
            return
        from .constants import PROGPOW_CNT_CACHE as cc, PROGPOW_CNT_MATH as cm
        t0 = time.time()
        spv = sc.compile_search(period, self._dag_elements, self.variant,
                                MAX_OUTPUTS, cc, cm)
        self._search_pipe = ComputePipeline(
            self.dev, spv, num_bindings=3, push_const_size=16,
            local_size=SEARCH_LOCAL_SIZE, required_subgroup_size=32)
        self._search_pipe.bind([self._hdr_buf, self._dag_buf, self._out_buf])
        self._period = period
        log(f"Compiled period {period} kernel in {(time.time()-t0)*1000:.0f} ms")

    # --- search ---
    def search(self, header_bytes, target, start_nonce, num_nonces):
        """Dispatch the search over [start_nonce, start_nonce+num_nonces).

        target is the 64-bit boundary (upper bits). Returns (solutions, hashes).
        """
        self._hdr_buf.write(header_bytes)
        self._out_buf.write(b"\x00" * 16)  # zero count/hashCount/abort
        groups = (num_nonces + SEARCH_LOCAL_SIZE - 1) // SEARCH_LOCAL_SIZE
        push = struct.pack("<4I", start_nonce & 0xFFFFFFFF, start_nonce >> 32,
                           target & 0xFFFFFFFF, (target >> 32) & 0xFFFFFFFF)
        self.dev.dispatch(self._search_pipe, groups, push)

        raw = self._out_buf.read(self._out_size)
        count, hash_count, _abort, _pad = struct.unpack("<4I", raw[:16])
        sols = []
        for s in range(min(count, MAX_OUTPUTS)):
            base = 16 + s * 9 * 4
            v = struct.unpack("<9I", raw[base:base + 36])
            sols.append(Solution(start_nonce + v[0], v[1:9]))
        return sols, hash_count

    def calibrate(self, header, log=print):
        """Find a per-dispatch nonce batch that runs ~WATCHDOG_TARGET_S.

        Ramps up from a tiny batch, doubling and measuring, and STOPS as soon as a
        dispatch reaches the target -- so no single dispatch ever runs long enough
        to trip the GPU watchdog.
        """
        batch = 1 << 13
        per_nonce = None
        while batch <= (1 << 24):
            t0 = time.time()
            _, hashes = self.search(header, 0, 0, batch)
            dt = time.time() - t0
            if hashes:
                per_nonce = dt / hashes
            log(f"  calibrate batch={batch} -> {dt*1000:.1f} ms "
                f"({hashes/dt/1e6:.1f} MH/s)")
            if dt >= WATCHDOG_TARGET_S:
                break
            batch <<= 1
        if per_nonce:
            safe = int(WATCHDOG_TARGET_S / per_nonce)
            # round down to a multiple of the local size
            self._safe_batch = max(SEARCH_LOCAL_SIZE,
                                   (safe // SEARCH_LOCAL_SIZE) * SEARCH_LOCAL_SIZE)
        log(f"  safe batch = {self._safe_batch} nonces (~{WATCHDOG_TARGET_S*1000:.0f} ms/dispatch)")
        return self._safe_batch

    def benchmark(self, block, seconds=10.0, use_real_dag=False, log=print):
        """Measure sustained hashrate (MH/s) using watchdog-safe dispatches."""
        if use_real_dag:
            self.ensure_epoch(block, log=log)
        else:
            self.setup_benchmark_dag(block, log)
        self.ensure_period(block, log)

        header = bytes(range(32))
        log("Calibrating safe dispatch size...")
        self.calibrate(header, log)

        start = 0
        total = 0
        t0 = time.time()
        while time.time() - t0 < seconds:
            _, hashes = self.search(header, 0, start, self._safe_batch)
            total += hashes
            start += self._safe_batch
        dt = time.time() - t0
        mhs = total / dt / 1e6
        log(f"Benchmark: {total/1e6:.1f} Mhashes in {dt:.1f}s = {mhs:.2f} MH/s")
        return mhs
