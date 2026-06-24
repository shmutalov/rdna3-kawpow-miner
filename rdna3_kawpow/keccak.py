"""Keccak primitives for KawPow.

- keccak-f800 (25 x 32-bit lanes, 22 rounds): the ProgPoW/KawPow sponge used for
  the initial seed hash and the final result hash. Implemented directly (no lib
  provides the 32-bit-lane variant).
- Original Keccak-256/512 (0x01 padding) for ethash light-cache / DAG generation,
  via pysha3's keccak_* (NOT hashlib.sha3_*, which is NIST SHA3 with 0x06 padding).

KawPow absorbs the Ravencoin round constants ("RAVENCOINKAWPOW") into the keccak
state, distinguishing it from vanilla ProgPoW.
"""

import struct

# Original Keccak (0x01 padding) for ethash, via pycryptodome -- NOT hashlib.sha3_*
# (NIST SHA3, 0x06 padding). Crypto.Hash.keccak is the original Keccak.
from Crypto.Hash import keccak as _keccak

from .constants import KECCAKF_RNDC, KECCAKF_ROTC, KECCAKF_PILN, RAVENCOIN_RNDC, MASK32

KECCAK_F800_ROUNDS = 22


def _rotl32(x, n):
    n &= 31
    return ((x << n) | (x >> (32 - n))) & MASK32 if n else x & MASK32


def keccak_f800_round(st, r):
    bc = [0] * 5
    # Theta
    for i in range(5):
        bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20]
    for i in range(5):
        t = bc[(i + 4) % 5] ^ _rotl32(bc[(i + 1) % 5], 1)
        for j in range(0, 25, 5):
            st[j + i] ^= t
    # Rho Pi
    t = st[1]
    for i in range(24):
        j = KECCAKF_PILN[i]
        bc[0] = st[j]
        st[j] = _rotl32(t, KECCAKF_ROTC[i])
        t = bc[0]
    # Chi
    for j in range(0, 25, 5):
        for i in range(5):
            bc[i] = st[j + i]
        for i in range(5):
            st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5]
            st[j + i] &= MASK32
    # Iota
    st[0] ^= KECCAKF_RNDC[r]
    st[0] &= MASK32


def keccak_f800(st):
    """In-place keccak-f800 permutation (22 rounds). `st` is a list of 25 uint32."""
    for r in range(KECCAK_F800_ROUNDS):
        keccak_f800_round(st, r)
    return st


def _swab32(x):
    return struct.unpack("<I", struct.pack(">I", x & MASK32))[0]


def header_to_words(header_bytes):
    """32-byte header hash -> 8 little-endian uint32 words."""
    assert len(header_bytes) == 32
    return list(struct.unpack("<8I", header_bytes))


KAWPOW = "kawpow"     # Ravencoin production: absorbs RAVENCOIN_RNDC
VANILLA = "vanilla"   # ProgPoW 0.9.2 reference (test/kernel.cu vector)


def progpow_seed(header_words, nonce, variant=KAWPOW):
    """Initial keccak: returns (state2[0..7], seed64).

    state2 carries into the final hash; seed64 = (state2[1] << 32) | state2[0]
    seeds fill_mix. KawPow fills the tail with the Ravencoin constants; vanilla
    ProgPoW 0.9.2 fills it with the (zero) digest.
    """
    st = [0] * 25
    st[0:8] = [w & MASK32 for w in header_words]
    st[8] = nonce & MASK32
    st[9] = (nonce >> 32) & MASK32
    if variant == KAWPOW:
        for i in range(10, 25):
            st[i] = RAVENCOIN_RNDC[i - 10]
    keccak_f800(st)
    state2 = st[0:8]
    seed64 = (state2[1] << 32) | state2[0]
    return state2, seed64


def progpow_final(state2, digest_words, header_words=None, seed64=None,
                  variant=KAWPOW):
    """Final keccak. digest_words = 8 uint32 (the 256-bit mix hash).

    Returns the 64-bit result compared against the target.
      KawPow:  state = state2(8) | digest(8) | RAVENCOIN_RNDC(9)
      vanilla: state = header(8) | seed64(2) | digest(8) | zero(7)
    """
    st = [0] * 25
    if variant == KAWPOW:
        st[0:8] = [w & MASK32 for w in state2]
        st[8:16] = [w & MASK32 for w in digest_words]
        for i in range(16, 25):
            st[i] = RAVENCOIN_RNDC[i - 16]
    else:
        st[0:8] = [w & MASK32 for w in header_words]
        st[8] = seed64 & MASK32
        st[9] = (seed64 >> 32) & MASK32
        st[10:18] = [w & MASK32 for w in digest_words]
    keccak_f800(st)
    return (_swab32(st[0]) << 32) | _swab32(st[1])


# --- Original Keccak (for ethash) ---

def keccak_512(data):
    return _keccak.new(digest_bits=512, data=data).digest()


def keccak_256(data):
    return _keccak.new(digest_bits=256, data=data).digest()
