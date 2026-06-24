"""Offline unit tests for the on-disk light-cache + DAG cache (no GPU)."""

import glob
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdna3_kawpow import dagcache, ethash  # noqa: E402

VARIANT, EPOCH, PARENTS = "kawpow", 123, 512
SEED = bytes(range(32))


def _cache():
    return dagcache.Cache(tempfile.mkdtemp(prefix="dagcache_test_"))


def test_light_round_trip():
    c = _cache()
    light = os.urandom(64 * 1000)
    assert c.load_light(VARIANT, EPOCH, PARENTS, SEED) is None    # miss before write
    c.save_light(VARIANT, EPOCH, PARENTS, SEED, light)
    assert c.load_light(VARIANT, EPOCH, PARENTS, SEED) == light


def test_light_key_mismatches_miss():
    c = _cache()
    light = os.urandom(64 * 16)
    c.save_light(VARIANT, EPOCH, PARENTS, SEED, light)
    assert c.load_light(VARIANT, EPOCH, PARENTS, os.urandom(32)) is None   # seed
    assert c.load_light(VARIANT, EPOCH + 1, PARENTS, SEED) is None         # epoch
    assert c.load_light(VARIANT, EPOCH, 256, SEED) is None                 # parents
    assert c.load_light("vanilla", EPOCH, PARENTS, SEED) is None           # variant


def test_dag_round_trip_chunked():
    c = _cache()
    full = 64 * 4096
    payload = os.urandom(full)
    assert not c.has_dag(VARIANT, EPOCH, PARENTS, SEED, full)
    w = c.open_dag_write(VARIANT, EPOCH, PARENTS, SEED, full)
    step = 7777                          # deliberately not a divisor of `full`
    for i in range(0, full, step):
        w.write(payload[i:i + step])
    w.commit()
    assert c.has_dag(VARIANT, EPOCH, PARENTS, SEED, full)
    r = c.open_dag_read(VARIANT, EPOCH, PARENTS)
    out = b""
    while True:
        chunk = r.read(step)
        if not chunk:
            break
        out += chunk
    r.close()
    assert out == payload


def test_dag_size_and_seed_mismatch_miss():
    c = _cache()
    full = 64 * 256
    w = c.open_dag_write(VARIANT, EPOCH, PARENTS, SEED, full)
    w.write(os.urandom(full))
    w.commit()
    assert not c.has_dag(VARIANT, EPOCH, PARENTS, SEED, full + 64)     # size
    assert not c.has_dag(VARIANT, EPOCH, PARENTS, os.urandom(32), full)  # seed


def test_corrupt_header_is_a_miss():
    c = _cache()
    full = 64 * 64
    w = c.open_dag_write(VARIANT, EPOCH, PARENTS, SEED, full)
    w.write(os.urandom(full))
    w.commit()
    path = glob.glob(os.path.join(c.dir, "dag-*.bin"))[0]
    with open(path, "r+b") as f:
        f.seek(0)
        f.write(b"XXXX")                 # clobber the magic
    assert not c.has_dag(VARIANT, EPOCH, PARENTS, SEED, full)


def test_prune_keeps_newest():
    c = _cache()
    full = 64 * 16
    for e in range(EPOCH, EPOCH + dagcache.KEEP_DAG + 3):
        w = c.open_dag_write(VARIANT, e, PARENTS, SEED, full)
        w.write(os.urandom(full))
        w.commit()
        c.prune_dag(VARIANT)
    remaining = glob.glob(os.path.join(c.dir, f"dag-{VARIANT}-e*.bin"))
    assert len(remaining) == dagcache.KEEP_DAG


def test_precomputed_lightcache_matches_built():
    """A reloaded (precomputed) cache must reproduce identical dataset items."""
    seed = b"\x11" * 32
    cache = ethash.mkcache(64 * 211, seed)        # small reference cache
    split = [bytes(b"".join(cache)[i:i + 64])
             for i in range(0, 64 * 211, 64)]
    assert split == cache
    for i in (0, 1, 7, 100, 210):
        assert (ethash.calc_dataset_item(cache, i, parents=512)
                == ethash.calc_dataset_item(split, i, parents=512))


def test_precomputed_wrong_size_rejected():
    size = ethash.get_cache_size(0)
    try:
        ethash.LightCache(0, ethash.KAWPOW_EPOCH_LENGTH, seed=b"\x00" * 32,
                          parents=512, precomputed_cache=b"\x00" * (size - 64))
    except ValueError:
        return
    raise AssertionError("wrong-size precomputed_cache was accepted")


def run_all():
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
    print(f"PASS - {n} dagcache unit tests")


if __name__ == "__main__":
    run_all()
