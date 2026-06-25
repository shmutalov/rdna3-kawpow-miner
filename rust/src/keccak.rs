//! Keccak primitives for KawPow (port of `rdna3_kawpow/keccak.py`).
//!
//! - `keccak-f800` (25x32-bit lanes, 22 rounds): the ProgPoW/KawPow sponge for the
//!   initial seed hash and the final result hash. Hand-implemented (no library
//!   provides the 32-bit-lane variant).
//! - Original Keccak-256/512 (0x01 padding) for ethash, via `tiny-keccak` -- NOT
//!   SHA3 (0x06 padding), which would silently produce wrong DAGs.

use tiny_keccak::{Hasher, Keccak};

use crate::constants::{KECCAKF_PILN, KECCAKF_RNDC, KECCAKF_ROTC, RAVENCOIN_RNDC};

pub const KECCAK_F800_ROUNDS: usize = 22;

/// KawPow vs vanilla ProgPoW 0.9.2 -- threaded through keccak tail layout, op
/// counts, parent count and epoch length.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Variant {
    /// Ravencoin production: absorbs RAVENCOIN_RNDC into the keccak tail.
    Kawpow,
    /// ProgPoW 0.9.2 reference (the carried test/kernel.cu vector).
    Vanilla,
}

#[inline]
fn keccak_f800_round(st: &mut [u32; 25], r: usize) {
    let mut bc = [0u32; 5];
    // Theta
    for i in 0..5 {
        bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
    }
    for i in 0..5 {
        let t = bc[(i + 4) % 5] ^ bc[(i + 1) % 5].rotate_left(1);
        let mut j = 0;
        while j < 25 {
            st[j + i] ^= t;
            j += 5;
        }
    }
    // Rho Pi
    let mut t = st[1];
    for i in 0..24 {
        let j = KECCAKF_PILN[i];
        let tmp = st[j];
        st[j] = t.rotate_left(KECCAKF_ROTC[i]);
        t = tmp;
    }
    // Chi
    let mut j = 0;
    while j < 25 {
        for i in 0..5 {
            bc[i] = st[j + i];
        }
        for i in 0..5 {
            st[j + i] ^= (!bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        j += 5;
    }
    // Iota
    st[0] ^= KECCAKF_RNDC[r];
}

/// In-place keccak-f800 permutation (22 rounds).
pub fn keccak_f800(st: &mut [u32; 25]) {
    for r in 0..KECCAK_F800_ROUNDS {
        keccak_f800_round(st, r);
    }
}

#[inline]
pub fn swab32(x: u32) -> u32 {
    x.swap_bytes()
}

/// 32-byte header hash -> 8 little-endian u32 words.
pub fn header_to_words(header: &[u8]) -> [u32; 8] {
    assert_eq!(header.len(), 32, "header must be 32 bytes");
    let mut w = [0u32; 8];
    for (i, word) in w.iter_mut().enumerate() {
        *word = u32::from_le_bytes(header[i * 4..i * 4 + 4].try_into().unwrap());
    }
    w
}

/// Initial keccak. Returns `(state2[0..8], seed64)` where
/// `seed64 = (state2[1] << 32) | state2[0]` seeds fill_mix. KawPow fills the tail
/// with the Ravencoin constants; vanilla fills it with the (zero) digest.
pub fn progpow_seed(header_words: &[u32; 8], nonce: u64, variant: Variant) -> ([u32; 8], u64) {
    let mut st = [0u32; 25];
    st[0..8].copy_from_slice(header_words);
    st[8] = nonce as u32;
    st[9] = (nonce >> 32) as u32;
    if variant == Variant::Kawpow {
        for i in 10..25 {
            st[i] = RAVENCOIN_RNDC[i - 10];
        }
    }
    keccak_f800(&mut st);
    let mut state2 = [0u32; 8];
    state2.copy_from_slice(&st[0..8]);
    let seed64 = ((state2[1] as u64) << 32) | state2[0] as u64;
    (state2, seed64)
}

/// Final keccak. `digest` = 8 u32 (the 256-bit mix hash). Returns the 64-bit
/// result compared against the target.
///   KawPow:  state = state2(8) | digest(8) | RAVENCOIN_RNDC(9)
///   vanilla: state = header(8) | seed64(2) | digest(8) | zero(7)
pub fn progpow_final(
    state2: &[u32; 8],
    digest: &[u32; 8],
    header_words: &[u32; 8],
    seed64: u64,
    variant: Variant,
) -> u64 {
    let mut st = [0u32; 25];
    if variant == Variant::Kawpow {
        st[0..8].copy_from_slice(state2);
        st[8..16].copy_from_slice(digest);
        for i in 16..25 {
            st[i] = RAVENCOIN_RNDC[i - 16];
        }
    } else {
        st[0..8].copy_from_slice(header_words);
        st[8] = seed64 as u32;
        st[9] = (seed64 >> 32) as u32;
        st[10..18].copy_from_slice(digest);
    }
    keccak_f800(&mut st);
    ((swab32(st[0]) as u64) << 32) | swab32(st[1]) as u64
}

// --- Original Keccak (for ethash) ---

pub fn keccak_512(data: &[u8]) -> [u8; 64] {
    let mut k = Keccak::v512();
    k.update(data);
    let mut out = [0u8; 64];
    k.finalize(&mut out);
    out
}

pub fn keccak_256(data: &[u8]) -> [u8; 32] {
    let mut k = Keccak::v256();
    k.update(data);
    let mut out = [0u8; 32];
    k.finalize(&mut out);
    out
}
