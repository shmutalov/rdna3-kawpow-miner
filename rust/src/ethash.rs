//! Ethash light-cache and dataset (DAG) sizing/generation for KawPow.
//!
//! Port of `rdna3_kawpow/ethash.py`. The light cache is generated on the host; the
//! full multi-GB DAG is generated on the GPU from it. `calc_dataset_item` lets
//! individual DAG rows be reproduced for the CPU reference without materialising
//! the whole DAG.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::constants::{FNV_PRIME, PROGPOW_CACHE_WORDS, PROGPOW_DAG_LOADS};
use crate::keccak::{keccak_256, keccak_512};

// Standard Ethash parameters.
const CACHE_BYTES_INIT: u64 = 1 << 24; // 16 MiB
const CACHE_BYTES_GROWTH: u64 = 1 << 17; // 128 KiB
const DATASET_BYTES_INIT: u64 = 1 << 30; // 1 GiB
const DATASET_BYTES_GROWTH: u64 = 1 << 23; // 8 MiB
const HASH_BYTES: u64 = 64;
const MIX_BYTES: u64 = 128;
/// Standard Ravencoin KawPow uses 512 dataset-item parents (Ethereum ethash: 256).
pub const DATASET_PARENTS: u32 = 512;
const CACHE_ROUNDS: usize = 3;

pub const ETHEREUM_EPOCH_LENGTH: u64 = 30000;
pub const KAWPOW_EPOCH_LENGTH: u64 = 7500;

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut i = 3u64;
    while i * i <= n {
        if n % i == 0 {
            return false;
        }
        i += 2;
    }
    true
}

pub fn epoch_number(block: u64, epoch_length: u64) -> u64 {
    block / epoch_length
}

pub fn get_cache_size(epoch: u64) -> u64 {
    let mut sz = CACHE_BYTES_INIT + CACHE_BYTES_GROWTH * epoch - HASH_BYTES;
    while !is_prime(sz / HASH_BYTES) {
        sz -= 2 * HASH_BYTES;
    }
    sz
}

pub fn get_full_size(epoch: u64) -> u64 {
    let mut sz = DATASET_BYTES_INIT + DATASET_BYTES_GROWTH * epoch - MIX_BYTES;
    while !is_prime(sz / MIX_BYTES) {
        sz -= 2 * MIX_BYTES;
    }
    sz
}

/// Cheap DAG sizing (no cache build): (epoch, full_size, dag_items, dag_elements).
pub fn dag_sizing(block: u64, epoch_length: u64) -> (u64, u64, u64, u64) {
    let epoch = epoch_number(block, epoch_length);
    let full_size = get_full_size(epoch);
    let dag_items = full_size / HASH_BYTES;
    let dag_elements = (full_size / MIX_BYTES) / 2;
    (epoch, full_size, dag_items, dag_elements)
}

pub fn seed_hash(epoch: u64) -> [u8; 32] {
    let mut s = [0u8; 32];
    for _ in 0..epoch {
        s = keccak_256(&s);
    }
    s
}

#[inline]
fn fnv(a: u32, b: u32) -> u32 {
    a.wrapping_mul(FNV_PRIME) ^ b
}

#[inline]
fn to_words(b: &[u8; 64]) -> [u32; 16] {
    let mut w = [0u32; 16];
    for (i, word) in w.iter_mut().enumerate() {
        *word = u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
    }
    w
}

#[inline]
fn from_words(w: &[u32; 16]) -> [u8; 64] {
    let mut b = [0u8; 64];
    for i in 0..16 {
        b[i * 4..i * 4 + 4].copy_from_slice(&w[i].to_le_bytes());
    }
    b
}

/// Generate the light cache as a vector of 64-byte items.
pub fn mkcache(cache_size: u64, seed: &[u8; 32]) -> Vec<[u8; 64]> {
    let n = (cache_size / HASH_BYTES) as usize;
    let mut cache: Vec<[u8; 64]> = Vec::with_capacity(n);
    cache.push(keccak_512(seed));
    for i in 1..n {
        cache.push(keccak_512(&cache[i - 1]));
    }
    // Three rounds of randmemohash.
    for _ in 0..CACHE_ROUNDS {
        for i in 0..n {
            let v = u32::from_le_bytes(cache[i][..4].try_into().unwrap()) as usize % n;
            let a = cache[(i + n - 1) % n];
            let b = cache[v];
            let mut x = [0u8; 64];
            for k in 0..64 {
                x[k] = a[k] ^ b[k];
            }
            cache[i] = keccak_512(&x);
        }
    }
    cache
}

/// Compute Ethash dataset item `i` (64 bytes) as 16 u32 words.
pub fn calc_dataset_item(cache: &[[u8; 64]], i: u32, parents: u32) -> [u32; 16] {
    let n = cache.len();
    let mut mix = to_words(&cache[(i as usize) % n]);
    mix[0] ^= i;
    mix = to_words(&keccak_512(&from_words(&mix)));
    for k in 0..parents {
        let parent = (fnv(i ^ k, mix[(k % 16) as usize]) as usize) % n;
        let pw = to_words(&cache[parent]);
        for w in 0..16 {
            mix[w] = fnv(mix[w], pw[w]);
        }
    }
    to_words(&keccak_512(&from_words(&mix)))
}

/// Holds an epoch's light cache plus sizing, and serves DAG rows on demand.
pub struct LightCache {
    pub epoch: u64,
    pub cache_size: u64,
    pub full_size: u64,
    pub parents: u32,
    pub seed: [u8; 32],
    pub cache: Vec<[u8; 64]>,
    pub dag_items: u64,    // 64-byte items
    pub dag_elements: u64, // matches the kernel's PROGPOW_DAG_ELEMENTS
    item_cache: RefCell<HashMap<u32, [u32; 16]>>,
}

impl LightCache {
    pub fn new(block: u64, epoch_length: u64, seed: Option<[u8; 32]>, parents: u32) -> Self {
        let epoch = epoch_number(block, epoch_length);
        let cache_size = get_cache_size(epoch);
        let full_size = get_full_size(epoch);
        let seed = seed.unwrap_or_else(|| seed_hash(epoch));
        let cache = mkcache(cache_size, &seed);
        LightCache {
            epoch,
            cache_size,
            full_size,
            parents,
            seed,
            cache,
            dag_items: full_size / HASH_BYTES,
            dag_elements: (full_size / MIX_BYTES) / 2,
            item_cache: RefCell::new(HashMap::new()),
        }
    }

    /// Reload a previously generated light cache (e.g. from disk) instead of the
    /// expensive `mkcache`. Rejects anything not sized exactly for this epoch.
    pub fn from_precomputed(
        block: u64,
        epoch_length: u64,
        seed: [u8; 32],
        parents: u32,
        precomputed: &[u8],
    ) -> Result<Self, String> {
        let epoch = epoch_number(block, epoch_length);
        let cache_size = get_cache_size(epoch);
        let full_size = get_full_size(epoch);
        if precomputed.len() as u64 != cache_size {
            return Err(format!(
                "light cache size mismatch: {} != {}",
                precomputed.len(),
                cache_size
            ));
        }
        let cache: Vec<[u8; 64]> = precomputed
            .chunks_exact(64)
            .map(|c| c.try_into().unwrap())
            .collect();
        Ok(LightCache {
            epoch,
            cache_size,
            full_size,
            parents,
            seed,
            cache,
            dag_items: full_size / HASH_BYTES,
            dag_elements: (full_size / MIX_BYTES) / 2,
            item_cache: RefCell::new(HashMap::new()),
        })
    }

    /// The light cache as one flat byte buffer (for upload / disk caching).
    pub fn flatten_cache(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.cache.len() * 64);
        for item in &self.cache {
            out.extend_from_slice(item);
        }
        out
    }

    pub fn dataset_item(&self, i: u32) -> [u32; 16] {
        if let Some(v) = self.item_cache.borrow().get(&i) {
            return *v;
        }
        let v = calc_dataset_item(&self.cache, i, self.parents);
        self.item_cache.borrow_mut().insert(i, v);
        v
    }

    /// The 4-word dag_t row `row` (each 64-byte dataset item == 4 rows).
    pub fn dag_row(&self, row: u32) -> [u32; 4] {
        let item = self.dataset_item(row >> 2);
        let off = ((row & 3) * 4) as usize;
        [item[off], item[off + 1], item[off + 2], item[off + 3]]
    }

    /// The 4096-word (PROGPOW_CACHE_BYTES) cache the kernel loads into LDS.
    pub fn progpow_cache(&self) -> Vec<u32> {
        let rows_needed = (PROGPOW_CACHE_WORDS / PROGPOW_DAG_LOADS) as u32; // 1024 rows
        let mut words = Vec::with_capacity(PROGPOW_CACHE_WORDS);
        for row in 0..rows_needed {
            words.extend_from_slice(&self.dag_row(row));
        }
        words
    }
}
