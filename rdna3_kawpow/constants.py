"""KawPow / ProgPoW 0.9.3 algorithm constants (Ravencoin variant).

Mirrors libprogpow/ProgPow.h and the keccak tables in the CUDA/OpenCL kernels
of the upstream kawpowminer. KawPow == ProgPoW 0.9.3 with a keccak-f800 that
absorbs the Ravencoin round constants ("RAVENCOINKAWPOW").
"""

# --- ProgPoW tunables (libprogpow/ProgPow.h) ---
PROGPOW_PERIOD = 3          # blocks before the random program changes (Ravencoin)
PROGPOW_LANES = 16          # lanes cooperating on one hash
PROGPOW_REGS = 32           # uint32 mix registers per lane
PROGPOW_DAG_LOADS = 4       # uint32 loads from the DAG per lane
PROGPOW_CACHE_BYTES = 16 * 1024
PROGPOW_CACHE_WORDS = PROGPOW_CACHE_BYTES // 4   # 4096
PROGPOW_CNT_DAG = 64        # DAG accesses == inner loop count
PROGPOW_CNT_CACHE = 11      # random cache accesses per loop
PROGPOW_CNT_MATH = 18       # random math ops per loop

# KawPow DAG epoch length (Ravencoin). NOTE: distinct from Ethereum's 30000.
KAWPOW_EPOCH_LENGTH = 7500

FNV_PRIME = 0x01000193
FNV_OFFSET_BASIS = 0x811C9DC5

MASK32 = 0xFFFFFFFF

# keccak-f800 round constants (24 rounds; only first 22 are run for mining)
KECCAKF_RNDC = [
    0x00000001, 0x00008082, 0x0000808A, 0x80008000, 0x0000808B, 0x80000001,
    0x80008081, 0x00008009, 0x0000008A, 0x00000088, 0x80008009, 0x8000000A,
    0x8000808B, 0x0000008B, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
    0x0000800A, 0x8000000A, 0x80008081, 0x00008080, 0x80000001, 0x80008008,
]

KECCAKF_ROTC = [
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
]

KECCAKF_PILN = [
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
]

# Ravencoin input constraints absorbed into keccak-f800 (spells RAVENCOINKAWPOW)
RAVENCOIN_RNDC = [
    0x00000072,  # R
    0x00000041,  # A
    0x00000056,  # V
    0x00000045,  # E
    0x0000004E,  # N
    0x00000043,  # C
    0x0000004F,  # O
    0x00000049,  # I
    0x0000004E,  # N
    0x0000004B,  # K
    0x00000041,  # A
    0x00000057,  # W
    0x00000050,  # P
    0x0000004F,  # O
    0x00000057,  # W
]
