//! Pure-Rust KawPow / ProgPoW reference hash (CPU correctness oracle).
//!
//! Port of `rdna3_kawpow/reference.py`. Runs the exact same program IR that the
//! GPU shader executes (from `progpow`), so it doubles as the oracle for GPU
//! output. Exposes intermediates (fill_mix, per-loop DAG entry, per-lane digest)
//! to support validation against the canonical test vectors.

use crate::constants::{FNV_OFFSET_BASIS, FNV_PRIME, PROGPOW_CNT_DAG, PROGPOW_LANES, PROGPOW_REGS};
use crate::ethash::LightCache;
use crate::keccak::{self, Variant};
use crate::progpow::{self, Kiss99};

/// Optional intermediates captured during `hash_one` for vector validation.
#[derive(Default)]
pub struct Collected {
    pub seed64: u64,
    pub fill_mix: Vec<[u32; PROGPOW_REGS]>,
    pub loop_entry: Vec<u32>,
    pub digest_lane: [u32; 16],
}

#[inline]
fn fnv1a(h: u32, d: u32) -> u32 {
    (h ^ d).wrapping_mul(FNV_PRIME)
}

/// Expand the per-hash seed to this lane's 32 mix registers (KISS99).
pub fn fill_mix(seed0: u32, seed1: u32, lane_id: u32) -> [u32; PROGPOW_REGS] {
    let mut fnv_hash = FNV_OFFSET_BASIS;
    fnv_hash = fnv1a(fnv_hash, seed0);
    let z = fnv_hash;
    fnv_hash = fnv1a(fnv_hash, seed1);
    let w = fnv_hash;
    fnv_hash = fnv1a(fnv_hash, lane_id);
    let jsr = fnv_hash;
    fnv_hash = fnv1a(fnv_hash, lane_id);
    let jcong = fnv_hash;
    let mut rnd = Kiss99::new(z, w, jsr, jcong);
    let mut mix = [0u32; PROGPOW_REGS];
    for m in mix.iter_mut() {
        *m = rnd.next_u32();
    }
    mix
}

/// Compute `(mix_hash_words[8], result64)` for one (header, nonce). If `collect`
/// is `Some`, it is filled with intermediates.
pub fn hash_one(
    header_bytes: &[u8],
    nonce: u64,
    light: &LightCache,
    prog_seed: u64,
    variant: Variant,
    cnt_cache: usize,
    cnt_math: usize,
    mut collect: Option<&mut Collected>,
) -> ([u32; 8], u64) {
    let header_words = keccak::header_to_words(header_bytes);
    let (state2, seed64) = keccak::progpow_seed(&header_words, nonce, variant);
    let (seed0, seed1) = if variant == Variant::Kawpow {
        // KawPow (kawpowminer 0.9.3): fill_mix consumes state2[0], state2[1] directly.
        (state2[0], state2[1])
    } else {
        // ProgPoW 0.9.2 reads the seed big-endian.
        (keccak::swab32(state2[1]), keccak::swab32(state2[0]))
    };

    let lanes = PROGPOW_LANES;
    let mut mix_lanes: Vec<[u32; PROGPOW_REGS]> =
        (0..lanes).map(|lane| fill_mix(seed0, seed1, lane)).collect();
    if let Some(c) = collect.as_deref_mut() {
        c.seed64 = seed64;
        c.fill_mix = mix_lanes.clone();
    }

    let ops = progpow::build_program(prog_seed, cnt_cache, cnt_math);
    let de = light.dag_elements as u32;
    let cdag = light.progpow_cache();
    let mut loop_entry = Vec::with_capacity(PROGPOW_CNT_DAG as usize);

    for loop_idx in 0..PROGPOW_CNT_DAG {
        let src_lane = (loop_idx % lanes) as usize;
        loop_entry.push(mix_lanes[src_lane][0] % de);
        progpow::run_loop(
            &ops,
            loop_idx,
            &mut mix_lanes,
            &cdag,
            |row| light.dag_row(row),
            de,
        );
    }
    if let Some(c) = collect.as_deref_mut() {
        c.loop_entry = loop_entry;
    }

    // Reduce each lane's mix to a 32-bit digest.
    let mut digest_lane = [0u32; 16];
    for lane in 0..lanes as usize {
        let mut d = FNV_OFFSET_BASIS;
        for i in 0..PROGPOW_REGS {
            d = fnv1a(d, mix_lanes[lane][i]);
        }
        digest_lane[lane] = d;
    }
    if let Some(c) = collect.as_deref_mut() {
        c.digest_lane = digest_lane;
    }

    // Reduce all lanes to a 256-bit digest (the mix hash, "my-domain" words).
    let mut digest = [FNV_OFFSET_BASIS; 8];
    let mut i = 0usize;
    while i < lanes as usize {
        for j in 0..8 {
            digest[j] = fnv1a(digest[j], digest_lane[i + j]);
        }
        i += 8;
    }

    // The final keccak takes the canonical (big-endian) seed.
    let canon_seed =
        ((keccak::swab32(state2[0]) as u64) << 32) | keccak::swab32(state2[1]) as u64;
    let result = keccak::progpow_final(&state2, &digest, &header_words, canon_seed, variant);
    (digest, result)
}

/// Canonical 32-byte mix hash from the 8 my-domain digest words (big-endian read).
pub fn mix_hash_bytes(digest: &[u32; 8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, &d) in digest.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&keccak::swab32(d).to_be_bytes());
    }
    out
}
