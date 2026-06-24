"""Ethash light-cache and dataset (DAG) generation for KawPow.

KawPow draws its DAG from the standard Ethash dataset, differing only in epoch
length (KawPow: 7500; Ethereum/vanilla-ProgPoW test vectors: 30000). The light
cache is generated on the host; the full multi-GB DAG is generated on the GPU
from this cache (see shaders/ethash_dag.comp). The host also exposes
calc_dataset_item() so individual DAG rows can be reproduced for validation
without materialising the whole DAG.
"""

import struct

from .keccak import keccak_512, keccak_256
from .constants import FNV_PRIME, MASK32

# Standard Ethash parameters.
CACHE_BYTES_INIT = 2 ** 24       # 16 MiB
CACHE_BYTES_GROWTH = 2 ** 17     # 128 KiB
DATASET_BYTES_INIT = 2 ** 30     # 1 GiB
DATASET_BYTES_GROWTH = 2 ** 23   # 8 MiB
HASH_BYTES = 64
MIX_BYTES = 128
# Standard Ravencoin KawPow (cpp-kawpow) uses 512 dataset-item parents. NOTE: this
# differs from classic Ethereum ethash (256) and from the Zing kawpowminer fork
# (256) -- it changes the DAG contents, so it must match the target coin.
DATASET_PARENTS = 512
CACHE_ROUNDS = 3

ETHEREUM_EPOCH_LENGTH = 30000     # vanilla ProgPoW / Ethereum
KAWPOW_EPOCH_LENGTH = 7500        # Ravencoin KawPow


def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def epoch_number(block, epoch_length=KAWPOW_EPOCH_LENGTH):
    return block // epoch_length


def get_cache_size(epoch):
    sz = CACHE_BYTES_INIT + CACHE_BYTES_GROWTH * epoch - HASH_BYTES
    while not _is_prime(sz // HASH_BYTES):
        sz -= 2 * HASH_BYTES
    return sz


def get_full_size(epoch):
    sz = DATASET_BYTES_INIT + DATASET_BYTES_GROWTH * epoch - MIX_BYTES
    while not _is_prime(sz // MIX_BYTES):
        sz -= 2 * MIX_BYTES
    return sz


def dag_sizing(block, epoch_length=KAWPOW_EPOCH_LENGTH):
    """Cheap DAG sizing (no cache build): (epoch, full_size, dag_items, dag_elements)."""
    epoch = epoch_number(block, epoch_length)
    full_size = get_full_size(epoch)
    dag_items = full_size // HASH_BYTES
    dag_elements = (full_size // MIX_BYTES) // 2
    return epoch, full_size, dag_items, dag_elements


def seed_hash(epoch):
    s = b"\x00" * 32
    for _ in range(epoch):
        s = keccak_256(s)
    return s


def _fnv(a, b):
    return ((a * FNV_PRIME) ^ b) & MASK32


def _to_words(b):
    return list(struct.unpack("<16I", b))


def _from_words(w):
    return struct.pack("<16I", *[x & MASK32 for x in w])


def mkcache(cache_size, seed):
    """Generate the light cache as a list of byte-strings (64 bytes each)."""
    n = cache_size // HASH_BYTES
    cache = [b""] * n
    cache[0] = keccak_512(seed)
    for i in range(1, n):
        cache[i] = keccak_512(cache[i - 1])
    # Three rounds of randmemohash.
    for _ in range(CACHE_ROUNDS):
        for i in range(n):
            v = struct.unpack("<I", cache[i][:4])[0] % n
            a = cache[(i - 1 + n) % n]
            b = cache[v]
            cache[i] = keccak_512(bytes(x ^ y for x, y in zip(a, b)))
    return cache


def calc_dataset_item(cache, i, parents=DATASET_PARENTS):
    """Compute Ethash dataset item `i` (64 bytes) as 16 uint32 words."""
    n = len(cache)
    mix = _to_words(cache[i % n])
    mix[0] ^= i
    mix = _to_words(keccak_512(_from_words(mix)))
    for k in range(parents):
        parent = _fnv(i ^ k, mix[k % 16]) % n
        pw = _to_words(cache[parent])
        for w in range(16):
            mix[w] = _fnv(mix[w], pw[w])
    return _to_words(keccak_512(_from_words(mix)))


class LightCache:
    """Holds an epoch's light cache plus sizing, and serves DAG rows on demand."""

    def __init__(self, block, epoch_length=KAWPOW_EPOCH_LENGTH, seed=None,
                 parents=DATASET_PARENTS, precomputed_cache=None):
        self.epoch = epoch_number(block, epoch_length)
        self.cache_size = get_cache_size(self.epoch)
        self.full_size = get_full_size(self.epoch)
        self.parents = parents
        # Prefer the pool-provided seed hash when available; it must equal
        # seed_hash(epoch) for a correctly-sized cache.
        self.seed = seed if seed is not None else seed_hash(self.epoch)
        if precomputed_cache is not None:
            # Reload a previously generated cache (e.g. from disk) instead of the
            # expensive mkcache. Reject anything not sized exactly for this epoch.
            if len(precomputed_cache) != self.cache_size:
                raise ValueError(
                    f"light cache size mismatch: {len(precomputed_cache)} "
                    f"!= {self.cache_size}")
            self.cache = [bytes(precomputed_cache[i:i + HASH_BYTES])
                          for i in range(0, self.cache_size, HASH_BYTES)]
        else:
            self.cache = mkcache(self.cache_size, self.seed)
        # The kernel's PROGPOW_DAG_ELEMENTS = (full_size / MIX_BYTES) / 2, i.e. the
        # number of 128-byte mix items halved. dag_t rows (4 words / 16 bytes) total
        # full_size/16; each 64-byte dataset item == 4 dag_t rows.
        self.dag_items = self.full_size // HASH_BYTES               # 64-byte items
        self.dag_elements = (self.full_size // MIX_BYTES) // 2      # matches kernel
        self._item_cache = {}

    def dataset_item(self, i):
        v = self._item_cache.get(i)
        if v is None:
            v = calc_dataset_item(self.cache, i, self.parents)
            self._item_cache[i] = v
        return v

    def dag_row(self, row):
        """Return the 4-word dag_t row `row` (each dataset item == 4 rows)."""
        item = self.dataset_item(row >> 2)
        off = (row & 3) * 4
        return item[off:off + 4]

    def progpow_cache(self):
        """The 4096-word (PROGPOW_CACHE_BYTES) cache loaded into LDS by the kernel."""
        from .constants import PROGPOW_CACHE_WORDS, PROGPOW_DAG_LOADS
        words = []
        rows_needed = PROGPOW_CACHE_WORDS // PROGPOW_DAG_LOADS  # 1024 dag_t rows
        for row in range(rows_needed):
            words.extend(self.dag_row(row))
        return words
