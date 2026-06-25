//! KawPow / ProgPoW 0.9.3 (Ravencoin) algorithm constants.
//!
//! Direct port of `rdna3_kawpow/constants.py`. KawPow == ProgPoW 0.9.3 with a
//! keccak-f800 that absorbs the Ravencoin round constants ("RAVENCOINKAWPOW").
//! These values are the single source of truth shared by the keccak/ethash/IR
//! ports and must stay byte-identical to the Python reference during migration.

// --- ProgPoW tunables (libprogpow/ProgPow.h) ---
pub const PROGPOW_PERIOD: u64 = 3; // blocks before the random program changes
pub const PROGPOW_LANES: u32 = 16; // lanes cooperating on one hash
pub const PROGPOW_REGS: usize = 32; // u32 mix registers per lane
pub const PROGPOW_DAG_LOADS: usize = 4; // u32 loads from the DAG per lane
pub const PROGPOW_CACHE_BYTES: usize = 16 * 1024;
pub const PROGPOW_CACHE_WORDS: usize = PROGPOW_CACHE_BYTES / 4; // 4096
pub const PROGPOW_CNT_DAG: u32 = 64; // DAG accesses == inner loop count
pub const PROGPOW_CNT_CACHE: usize = 11; // random cache accesses per loop
pub const PROGPOW_CNT_MATH: usize = 18; // random math ops per loop

/// KawPow DAG epoch length (Ravencoin). NOTE: distinct from Ethereum's 30000.
pub const KAWPOW_EPOCH_LENGTH: u64 = 7500;
/// Vanilla ProgPoW / Ethereum epoch length (the carried 0.9.2 test vector).
pub const ETHEREUM_EPOCH_LENGTH: u64 = 30000;

pub const FNV_PRIME: u32 = 0x0100_0193;
pub const FNV_OFFSET_BASIS: u32 = 0x811C_9DC5;

// keccak-f800 round constants (24 rounds; only first 22 are run for mining).
pub const KECCAKF_RNDC: [u32; 24] = [
    0x0000_0001, 0x0000_8082, 0x0000_808A, 0x8000_8000, 0x0000_808B, 0x8000_0001,
    0x8000_8081, 0x0000_8009, 0x0000_008A, 0x0000_0088, 0x8000_8009, 0x8000_000A,
    0x8000_808B, 0x0000_008B, 0x0000_8089, 0x0000_8003, 0x0000_8002, 0x0000_0080,
    0x0000_800A, 0x8000_000A, 0x8000_8081, 0x0000_8080, 0x8000_0001, 0x8000_8008,
];

pub const KECCAKF_ROTC: [u32; 24] = [
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39,
    61, 20, 44,
];

pub const KECCAKF_PILN: [usize; 24] = [
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9,
    6, 1,
];

/// Ravencoin input constraints absorbed into keccak-f800 (spells "RAVENCOINKAWPOW").
pub const RAVENCOIN_RNDC: [u32; 15] = [
    0x0000_0072, // R
    0x0000_0041, // A
    0x0000_0056, // V
    0x0000_0045, // E
    0x0000_004E, // N
    0x0000_0043, // C
    0x0000_004F, // O
    0x0000_0049, // I
    0x0000_004E, // N
    0x0000_004B, // K
    0x0000_0041, // A
    0x0000_0057, // W
    0x0000_0050, // P
    0x0000_004F, // O
    0x0000_0057, // W
];
