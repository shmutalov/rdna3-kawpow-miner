"""Pure-Python KawPow / ProgPoW reference hash (CPU correctness oracle).

Runs the exact same program IR that the GPU shader executes (from progpow.py),
so it doubles as the oracle for GPU output. Exposes intermediates (fill_mix,
per-loop DAG entry, per-lane digest) to support validation against test vectors.
"""

from . import keccak, progpow
from .constants import (
    PROGPOW_LANES, PROGPOW_REGS, PROGPOW_CNT_DAG, PROGPOW_CNT_CACHE,
    PROGPOW_CNT_MATH, FNV_OFFSET_BASIS, MASK32,
)


def _fnv1a(h, d):
    return ((h ^ d) * 0x01000193) & MASK32


def fill_mix(seed0, seed1, lane_id):
    """Expand the per-hash seed to this lane's 32 mix registers (KISS99)."""
    fnv_hash = FNV_OFFSET_BASIS
    fnv_hash = _fnv1a(fnv_hash, seed0); z = fnv_hash
    fnv_hash = _fnv1a(fnv_hash, seed1); w = fnv_hash
    fnv_hash = _fnv1a(fnv_hash, lane_id); jsr = fnv_hash
    fnv_hash = _fnv1a(fnv_hash, lane_id); jcong = fnv_hash
    rnd = progpow._Kiss99(z, w, jsr, jcong)
    return [rnd() for _ in range(PROGPOW_REGS)]


def hash_one(header_bytes, nonce, light, prog_seed, variant=keccak.KAWPOW,
             cnt_cache=PROGPOW_CNT_CACHE, cnt_math=PROGPOW_CNT_MATH,
             collect=None):
    """Compute (mix_hash_words[8], result64) for one (header, nonce).

    `light` is an ethash.LightCache. If `collect` is a dict it is filled with
    intermediates: 'seed64', 'mix' (16x32), 'loop_entry' (64), 'digest_lane' (16).
    """
    header_words = keccak.header_to_words(header_bytes)
    state2, seed64 = keccak.progpow_seed(header_words, nonce, variant)
    if variant == keccak.KAWPOW:
        # KawPow (kawpowminer 0.9.3): fill_mix consumes state2[0], state2[1] directly.
        seed0, seed1 = state2[0], state2[1]
    else:
        # ProgPoW 0.9.2 reads the seed big-endian (validated vs result.log):
        # hash_seed[0]=lo=swab(st[1]), [1]=hi=swab(st[0]).
        seed0 = keccak._swab32(state2[1])
        seed1 = keccak._swab32(state2[0])

    mix_lanes = [fill_mix(seed0, seed1, lane) for lane in range(PROGPOW_LANES)]
    if collect is not None:
        collect["seed64"] = seed64
        collect["fill_mix"] = [row[:] for row in mix_lanes]

    ops = progpow.build_program(prog_seed, cnt_cache, cnt_math)
    de = light.dag_elements
    cdag = light.progpow_cache()
    loop_entry = []

    for loop in range(PROGPOW_CNT_DAG):
        src_lane = loop % PROGPOW_LANES
        loop_entry.append(mix_lanes[src_lane][0] % de)
        progpow.run_loop(ops, loop, mix_lanes, cdag, light.dag_row, de)

    if collect is not None:
        collect["loop_entry"] = loop_entry

    # Reduce each lane's mix to a 32-bit digest.
    digest_lane = []
    for lane in range(PROGPOW_LANES):
        d = FNV_OFFSET_BASIS
        for i in range(PROGPOW_REGS):
            d = _fnv1a(d, mix_lanes[lane][i])
        digest_lane.append(d)
    if collect is not None:
        collect["digest_lane"] = digest_lane

    # Reduce all lanes to a 256-bit digest (the mix hash, "my-domain" words).
    digest = [FNV_OFFSET_BASIS] * 8
    for i in range(0, PROGPOW_LANES, 8):
        for j in range(8):
            digest[j] = _fnv1a(digest[j], digest_lane[i + j])

    # The final keccak takes the canonical (big-endian) seed.
    canon_seed = (keccak._swab32(state2[0]) << 32) | keccak._swab32(state2[1])
    result = keccak.progpow_final(state2, digest, header_words, canon_seed, variant)
    return digest, result


def mix_hash_bytes(digest):
    """Canonical 32-byte mix hash from the 8 my-domain digest words (big-endian read)."""
    import struct
    return struct.pack(">8I", *[keccak._swab32(d) for d in digest])
