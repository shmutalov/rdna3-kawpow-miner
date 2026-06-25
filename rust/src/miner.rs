//! KawPow miner orchestration on top of the Vulkan host (port of
//! `rdna3_kawpow/miner.py`). Per-epoch: build the light cache (host) + generate
//! the full DAG on the GPU (chunked, watchdog-safe). Per-period: regenerate +
//! compile the search kernel. Per-batch: dispatch the search over a nonce range.
//!
//! Phase 2 scope: epoch DAG, benchmark DAG, period kernel, search, calibration,
//! benchmark. The on-disk DAG cache and the live mining loop land in Phase 3.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Result};

use crate::constants::PROGPOW_PERIOD;
use crate::dagcache::Cache;
use crate::ethash::{self, LightCache};
use crate::keccak::{self, Variant};
use crate::shader_compiler as sc;
use crate::vkhost::{Buffer, ComputePipeline, VulkanDevice};

pub const MAX_OUTPUTS: u32 = 16;
pub const SEARCH_LOCAL_SIZE: u32 = 256; // wave32 multiple, tuned on gfx1100
const DAG_LOCAL_SIZE: u32 = 64;
// Keep every single GPU dispatch well under the OS GPU watchdog (TDR, ~2 s on
// Windows). The search auto-calibrates a safe batch at startup.
const SEARCH_CHUNK_NONCES: u32 = 1 << 20;
const WATCHDOG_TARGET_S: f64 = 0.20; // aim for ~200 ms dispatches
// Host-visible staging window for streaming the device-local DAG to/from disk.
const DAG_STAGING_BYTES: u64 = 64 << 20; // 64 MiB

fn variant_str(v: Variant) -> &'static str {
    match v {
        Variant::Kawpow => "kawpow",
        Variant::Vanilla => "vanilla",
    }
}

/// A found solution: nonce + the 8 "my-domain" digest words.
pub struct Solution {
    pub nonce: u64,
    pub mix_words: [u32; 8],
}

impl Solution {
    /// Canonical 32-byte mix hash (big-endian word read).
    pub fn mix_hash(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, &w) in self.mix_words.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&keccak::swab32(w).to_be_bytes());
        }
        out
    }
}

pub struct VulkanMiner {
    pub dev: VulkanDevice,
    pub variant: Variant,
    pub epoch_length: u64,
    pub local_size: u32,
    parents: u32,

    cache: Option<Cache>,
    epoch: Option<u64>,
    light: Option<LightCache>,
    dag_buf: Option<Buffer>,
    dag_elements: Option<u64>,
    period: Option<u64>,
    search_pipe: Option<ComputePipeline>,
    safe_batch: u32,

    hdr_buf: Buffer,
    out_buf: Buffer,
    out_size: usize,
}

impl VulkanMiner {
    pub fn new(
        device_index: Option<usize>,
        variant: Variant,
        epoch_length: u64,
        local_size: u32,
        dag_cache: bool,
        cache_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let dev = VulkanDevice::new(device_index)?;
        let parents = if variant == Variant::Kawpow { 512 } else { 256 };
        let out_size = 16 + (MAX_OUTPUTS as usize) * 9 * 4;
        let hdr_buf = dev.make_buffer(32, true, true, false)?;
        let out_buf = dev.make_buffer(out_size as u64, true, true, false)?;
        // Best-effort: a cache-dir failure degrades to "no cache", never aborts.
        let cache = if dag_cache {
            match Cache::new(cache_dir) {
                Ok(c) => Some(c),
                Err(e) => {
                    eprintln!("DAG cache disabled (cache dir error: {e})");
                    None
                }
            }
        } else {
            None
        };
        Ok(VulkanMiner {
            dev,
            variant,
            epoch_length,
            local_size,
            parents,
            cache,
            epoch: None,
            light: None,
            dag_buf: None,
            dag_elements: None,
            period: None,
            search_pipe: None,
            safe_batch: SEARCH_CHUNK_NONCES,
            hdr_buf,
            out_buf,
            out_size,
        })
    }

    pub fn light(&self) -> Option<&LightCache> {
        self.light.as_ref()
    }
    pub fn safe_batch(&self) -> u32 {
        self.safe_batch
    }

    // --- per-epoch DAG ---
    pub fn ensure_epoch(&mut self, block: u64, seed: Option<[u8; 32]>) -> Result<()> {
        let epoch = block / self.epoch_length;
        if self.epoch == Some(epoch) && self.dag_buf.is_some() && self.light.is_some() {
            return Ok(());
        }
        let eff_seed = seed.unwrap_or_else(|| ethash::seed_hash(epoch));
        let vstr = variant_str(self.variant);
        let e32 = epoch as u32;

        // 1) Light cache: reload from disk if present, else build it and cache it.
        let mut light: Option<LightCache> = None;
        if let Some(cache) = &self.cache {
            if let Some(bytes) = cache.load_light(vstr, e32, self.parents, &eff_seed) {
                let mib = bytes.len() / (1 << 20);
                match LightCache::from_precomputed(
                    block,
                    self.epoch_length,
                    eff_seed,
                    self.parents,
                    &bytes,
                ) {
                    Ok(lc) => {
                        println!("Epoch {epoch}: loaded {mib} MiB light cache from disk");
                        light = Some(lc);
                    }
                    Err(e) => println!("Epoch {epoch}: cached light cache rejected ({e}); rebuilding"),
                }
            }
        }
        let light = match light {
            Some(lc) => lc,
            None => {
                let t0 = Instant::now();
                println!("Epoch {epoch}: building light cache (host)...");
                let lc = LightCache::new(block, self.epoch_length, Some(eff_seed), self.parents);
                println!(
                    "  light={} MiB cache in {:.1}s",
                    lc.cache_size / (1 << 20),
                    t0.elapsed().as_secs_f64()
                );
                if let Some(cache) = &self.cache {
                    if let Err(e) = cache.save_light(vstr, e32, self.parents, &eff_seed, &lc.flatten_cache()) {
                        println!("  (light cache save skipped: {e})");
                    }
                }
                lc
            }
        };
        let full_size = light.full_size;

        // (Re)allocate the device-local DAG buffer (transfer=true for caching).
        self.dag_buf = None; // free the old one first
        self.dag_buf = Some(self.dev.make_buffer(full_size, false, true, true)?);

        // 2) DAG: reload from disk if present, else generate on the GPU and cache it.
        let mut loaded = false;
        if let Some(cache) = &self.cache {
            if cache.has_dag(vstr, e32, self.parents, &eff_seed, full_size) {
                match self.load_dag_from_disk(cache, e32, full_size) {
                    Ok(()) => loaded = true,
                    Err(e) => println!("  DAG cache load failed ({e}); regenerating"),
                }
            }
        }
        if !loaded {
            self.generate_dag(&light)?;
            if let Some(cache) = &self.cache {
                if let Err(e) = self.save_dag_to_disk(cache, e32, &eff_seed, full_size) {
                    println!("  (DAG cache save skipped: {e})");
                }
            }
        }

        self.dag_elements = Some(light.dag_elements);
        self.light = Some(light);
        self.epoch = Some(epoch);
        self.period = None; // force kernel recompile (dag_elements may change)
        Ok(())
    }

    /// Upload a cached DAG into the device-local buffer via a staging window.
    fn load_dag_from_disk(&self, cache: &Cache, epoch: u32, full: u64) -> Result<()> {
        use std::io::Read;
        let t0 = Instant::now();
        let mut reader = cache.open_dag_read(variant_str(self.variant), epoch, self.parents)?;
        let stage = self
            .dev
            .make_buffer(DAG_STAGING_BYTES.min(full), true, false, true)?;
        let dag = self.dag_buf.as_ref().unwrap();
        let mut buf = vec![0u8; DAG_STAGING_BYTES as usize];
        let mut pos = 0u64;
        while pos < full {
            let n = DAG_STAGING_BYTES.min(full - pos) as usize;
            reader.read_exact(&mut buf[..n])?;
            stage.write(&buf[..n], 0);
            self.dev.copy_buffer(&stage, dag, n as u64, 0, pos)?;
            pos += n as u64;
        }
        println!(
            "  loaded DAG={} MiB from disk in {:.1}s",
            full / (1 << 20),
            t0.elapsed().as_secs_f64()
        );
        Ok(())
    }

    /// Stream the device-local DAG out to disk via a staging window.
    fn save_dag_to_disk(&self, cache: &Cache, epoch: u32, seed: &[u8; 32], full: u64) -> Result<()> {
        let t0 = Instant::now();
        let mut w = cache.open_dag_write(variant_str(self.variant), epoch, self.parents, seed, full)?;
        let stage = self
            .dev
            .make_buffer(DAG_STAGING_BYTES.min(full), true, false, true)?;
        let dag = self.dag_buf.as_ref().unwrap();
        let res = (|| -> Result<()> {
            let mut pos = 0u64;
            while pos < full {
                let n = DAG_STAGING_BYTES.min(full - pos);
                self.dev.copy_buffer(dag, &stage, n, pos, 0)?;
                let data = stage.read(n as usize, 0);
                w.write_all(&data)?;
                pos += n;
            }
            Ok(())
        })();
        match res {
            Ok(()) => w.commit()?,
            Err(e) => {
                w.abort();
                return Err(e);
            }
        }
        cache.prune_dag(variant_str(self.variant));
        println!(
            "  saved DAG={} MiB to disk in {:.1}s",
            full / (1 << 20),
            t0.elapsed().as_secs_f64()
        );
        Ok(())
    }

    fn generate_dag(&self, light: &LightCache) -> Result<()> {
        println!("  generating DAG={} MiB on GPU...", light.full_size / (1 << 20));
        // Flatten the light cache to a byte buffer for upload.
        let mut light_bytes = Vec::with_capacity(light.cache.len() * 64);
        for item in &light.cache {
            light_bytes.extend_from_slice(item);
        }
        let light_buf = self.dev.make_buffer(light_bytes.len() as u64, true, true, false)?;
        light_buf.write(&light_bytes, 0);

        let spv = sc::compile_dag(
            light.cache.len() as u32,
            light.dag_items as u32,
            self.parents,
        )?;
        let dag_pipe = ComputePipeline::new(&self.dev, &spv, 2, 4, DAG_LOCAL_SIZE, 32)?;
        dag_pipe.bind(&[&light_buf, self.dag_buf.as_ref().unwrap()]);

        // Adaptive, watchdog-safe DAG generation: start tiny, grow only while each
        // dispatch stays fast, so no dispatch trips the OS GPU watchdog.
        let t1 = Instant::now();
        let items = light.dag_items;
        let mut start = 0u64;
        let mut chunk = 1u64 << 12; // 4096 items -- definitely safe
        let mut last_log = Instant::now();
        while start < items {
            let n = chunk.min(items - start);
            let groups = ((n + DAG_LOCAL_SIZE as u64 - 1) / DAG_LOCAL_SIZE as u64) as u32;
            let t0 = Instant::now();
            self.dev
                .dispatch(&dag_pipe, groups, &(start as u32).to_le_bytes())?;
            let dt = t0.elapsed().as_secs_f64();
            start += n;
            if dt < 0.12 && chunk < (1 << 19) {
                chunk <<= 1;
            } else if dt > 0.30 && chunk > (1 << 12) {
                chunk >>= 1;
            }
            if last_log.elapsed().as_secs_f64() > 3.0 {
                println!(
                    "  DAG {:5.1}%  ({} items, {:.0} ms/dispatch)",
                    100.0 * start as f64 / items as f64,
                    n,
                    dt * 1000.0
                );
                last_log = Instant::now();
            }
        }
        println!("  DAG generated in {:.1}s", t1.elapsed().as_secs_f64());
        Ok(())
    }

    /// Allocate an UNINITIALIZED full-size DAG buffer for throughput testing.
    pub fn setup_benchmark_dag(&mut self, block: u64) -> Result<()> {
        let (epoch, full_size, _dag_items, dag_elements) =
            ethash::dag_sizing(block, self.epoch_length);
        if self.dag_buf.is_none() || self.epoch != Some(epoch) {
            self.dag_buf = None;
            println!(
                "Allocating {} MiB device-local DAG buffer (uninitialized, benchmark only)...",
                full_size / (1 << 20)
            );
            self.dag_buf = Some(self.dev.make_buffer(full_size, false, true, false)?);
        }
        self.dag_elements = Some(dag_elements);
        self.epoch = Some(epoch);
        self.period = None;
        Ok(())
    }

    // --- per-period kernel ---
    pub fn ensure_period(&mut self, block: u64) -> Result<()> {
        let period = block / PROGPOW_PERIOD;
        if self.period == Some(period) && self.search_pipe.is_some() {
            return Ok(());
        }
        let de = self
            .dag_elements
            .ok_or_else(|| anyhow!("dag_elements unset (call ensure_epoch first)"))?;
        let t0 = Instant::now();
        let spv = sc::compile_search(period, de, self.variant, MAX_OUTPUTS)?;
        let pipe = ComputePipeline::new(&self.dev, &spv, 3, 16, self.local_size, 32)?;
        pipe.bind(&[
            &self.hdr_buf,
            self.dag_buf.as_ref().unwrap(),
            &self.out_buf,
        ]);
        self.search_pipe = Some(pipe);
        self.period = Some(period);
        println!(
            "Compiled period {period} kernel in {:.0} ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );
        Ok(())
    }

    // --- search ---
    pub fn search(
        &self,
        header_bytes: &[u8],
        target: u64,
        start_nonce: u64,
        num_nonces: u64,
    ) -> Result<(Vec<Solution>, u64)> {
        let pipe = self
            .search_pipe
            .as_ref()
            .ok_or_else(|| anyhow!("no search pipeline (call ensure_period first)"))?;
        self.hdr_buf.write(header_bytes, 0);
        self.out_buf.write(&[0u8; 16], 0); // zero count/hashCount/abort/pad

        let groups = ((num_nonces + self.local_size as u64 - 1) / self.local_size as u64) as u32;
        let mut push = [0u8; 16];
        push[0..4].copy_from_slice(&(start_nonce as u32).to_le_bytes());
        push[4..8].copy_from_slice(&((start_nonce >> 32) as u32).to_le_bytes());
        push[8..12].copy_from_slice(&(target as u32).to_le_bytes());
        push[12..16].copy_from_slice(&((target >> 32) as u32).to_le_bytes());
        self.dev.dispatch(pipe, groups, &push)?;

        let raw = self.out_buf.read(self.out_size, 0);
        let count = u32::from_le_bytes(raw[0..4].try_into().unwrap());
        let hash_count = u32::from_le_bytes(raw[4..8].try_into().unwrap());
        let mut sols = Vec::new();
        for s in 0..count.min(MAX_OUTPUTS) {
            let base = 16 + (s as usize) * 9 * 4;
            let rd = |i: usize| u32::from_le_bytes(raw[base + i * 4..base + i * 4 + 4].try_into().unwrap());
            let gid = rd(0);
            let mut mix = [0u32; 8];
            for (j, m) in mix.iter_mut().enumerate() {
                *m = rd(1 + j);
            }
            sols.push(Solution {
                nonce: start_nonce + gid as u64,
                mix_words: mix,
            });
        }
        Ok((sols, hash_count as u64))
    }

    /// Find a per-dispatch nonce batch that runs ~WATCHDOG_TARGET_S. Ramps up and
    /// STOPS as soon as a dispatch reaches the target, so none trips the watchdog.
    pub fn calibrate(&mut self, header: &[u8]) -> Result<u32> {
        let mut batch = 1u64 << 13;
        let mut per_nonce: Option<f64> = None;
        while batch <= (1 << 24) {
            let t0 = Instant::now();
            let (_, hashes) = self.search(header, 0, 0, batch)?;
            let dt = t0.elapsed().as_secs_f64();
            if hashes > 0 {
                per_nonce = Some(dt / hashes as f64);
            }
            println!(
                "  calibrate batch={batch} -> {:.1} ms ({:.1} MH/s)",
                dt * 1000.0,
                hashes as f64 / dt / 1e6
            );
            if dt >= WATCHDOG_TARGET_S {
                break;
            }
            batch <<= 1;
        }
        if let Some(pn) = per_nonce {
            let safe = (WATCHDOG_TARGET_S / pn) as u64;
            let ls = self.local_size as u64;
            self.safe_batch = ((safe / ls) * ls).max(ls) as u32;
        }
        println!(
            "  safe batch = {} nonces (~{:.0} ms/dispatch)",
            self.safe_batch,
            WATCHDOG_TARGET_S * 1000.0
        );
        Ok(self.safe_batch)
    }

    /// Measure sustained hashrate (MH/s) using watchdog-safe dispatches.
    pub fn benchmark(&mut self, block: u64, seconds: f64, use_real_dag: bool) -> Result<f64> {
        if use_real_dag {
            self.ensure_epoch(block, None)?;
        } else {
            self.setup_benchmark_dag(block)?;
        }
        self.ensure_period(block)?;

        let header: Vec<u8> = (0..32u8).collect();
        println!("Calibrating safe dispatch size...");
        self.calibrate(&header)?;

        let mut start = 0u64;
        let mut total = 0u64;
        let t0 = Instant::now();
        while t0.elapsed().as_secs_f64() < seconds {
            let (_, hashes) = self.search(&header, 0, start, self.safe_batch as u64)?;
            total += hashes;
            start += self.safe_batch as u64;
        }
        let dt = t0.elapsed().as_secs_f64();
        let mhs = total as f64 / dt / 1e6;
        println!(
            "Benchmark: {:.1} Mhashes in {:.1}s = {:.2} MH/s",
            total as f64 / 1e6,
            dt,
            mhs
        );
        Ok(mhs)
    }
}
