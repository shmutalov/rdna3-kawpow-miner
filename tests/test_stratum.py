"""Offline unit tests for stratum/CLI pure logic (no socket / no GPU)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdna3_kawpow import stratum  # noqa: E402
from rdna3_kawpow.__main__ import parse_pool_url  # noqa: E402


def test_difficulty_target():
    assert stratum.difficulty_to_target(1) == (1 << 256) - 1      # capped, not 0
    assert stratum.difficulty_to_target(2) == (1 << 256) // 2     # diff 2 halves it
    assert stratum.boundary64(stratum.difficulty_to_target(1)) == (1 << 64) - 1
    t = stratum.difficulty_to_target(256)
    assert stratum.boundary64(t) == (t >> 192) & ((1 << 64) - 1)


def test_target_hex():
    h = "00000000ffff0000000000000000000000000000000000000000000000000000"
    t = stratum.target_hex_to_int(h)
    assert stratum.boundary64(t) == 0x00000000ffff0000


def test_nonce_hex():
    assert stratum.nonce_hex(0x123456789abcdef0) == "123456789abcdef0"
    assert stratum.nonce_hex(0) == "0000000000000000"


def test_extranonce_start_nonce():
    j = stratum.Job("j", b"\x00" * 32, b"\x00" * 32, 1 << 255,
                    extranonce=0xABCD, extranonce_bits=16)
    n = j.start_nonce(0x1234)
    assert (n >> 48) == 0xABCD           # extranonce in the top 16 bits
    assert (n & 0xFFFFFFFFFFFF) == 0x1234


def test_parse_pool_url():
    p = parse_pool_url("stratum+tcp://RAVENADDR.rig1:x@pool.example:4444")
    assert p == dict(host="pool.example", port=4444, wallet="RAVENADDR",
                     worker="rig1", password="x")
    p2 = parse_pool_url("RAVENADDR@host:1234")
    assert p2["wallet"] == "RAVENADDR" and p2["worker"] == "rdna3" and p2["port"] == 1234


def test_notify_parsing():
    c = stratum.StratumClient("h", 1, "w", log=lambda *_: None)
    seen = []
    c.on_new_job = seen.append
    c._handle({"method": "mining.set_difficulty", "params": [2]})
    hdr = "ab" * 32
    seed = "cd" * 32
    tgt = "00000000ffff0000000000000000000000000000000000000000000000000000"
    c._handle({"method": "mining.notify",
               "params": ["job1", hdr, seed, tgt, True, "0x7530"]})
    assert len(seen) == 1
    job = seen[0]
    assert job.job_id == "job1"
    assert job.header == bytes.fromhex(hdr)
    assert job.height == 0x7530
    assert job.boundary64 == 0x00000000ffff0000


def run_all():
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
    print(f"PASS - {n} stratum/CLI unit tests")


if __name__ == "__main__":
    run_all()
