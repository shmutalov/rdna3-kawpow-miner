"""Regression test: the CPU reference reproduces the canonical ProgPoW 0.9.2
test vector (block 30000, from upstream kawpowminer test/kernel.cu + result.log).

This pins the entire algorithm engine -- keccak_f800, the program generator,
ethash cache+DAG, fill_mix, the 64-loop execution, digest reduction and the
final hash. KawPow production differs only by configuration (Ravencoin keccak
tail, 11/18 op counts, 7500 epoch length), all exercised by the same code paths.

Run: python -m pytest tests/  (or: python tests/test_reference_vector.py)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdna3_kawpow import ethash, reference, keccak  # noqa: E402

HEADER = bytes.fromhex(
    "ffeeddccbbaa9988776655443322110000112233445566778899aabbccddeeff")
NONCE = 0x123456789ABCDEF0
EXP_DIGEST = "11f19805c58ab46610ff9c719dcf0a5f18fa2f1605798eef770c47219274767d"
EXP_RESULT = 0x5B7CCD472DBEFDD9
EXP_DIGEST_LANES = [
    0x5883883E, 0x2FB0FD2E, 0xEADB7563, 0x4A171075, 0xAC2758F5, 0xAA5B06CF,
    0x52156E93, 0x4F7A7FFF, 0xFE91E36A, 0x9964C8B6, 0x6A3D93E2, 0x3C6D641F,
    0xE90DA618, 0x80CD8AB9, 0xCE72386F, 0x95517D28,
]


def _run():
    # The result.log/kernel.cu reference is vanilla ProgPoW 0.9.2 (256 DAG parents).
    lc = ethash.LightCache(30000, epoch_length=ethash.ETHEREUM_EPOCH_LENGTH, parents=256)
    col = {}
    digest, result = reference.hash_one(
        HEADER, NONCE, lc, prog_seed=600, variant=keccak.VANILLA,
        cnt_cache=12, cnt_math=20, collect=col)
    return lc, col, digest, result


def test_vector():
    lc, col, digest, result = _run()
    assert lc.cache_size == 16907456
    assert lc.full_size == 1082130304
    assert lc.dag_elements == 4227071
    assert col["fill_mix"][0][0] == 0x10C02F0D
    assert col["fill_mix"][15][31] == 0x20201012
    assert col["loop_entry"][:4] == [2043727, 1878577, 1972818, 4192557]
    assert col["loop_entry"][63] == 574671
    assert col["digest_lane"] == EXP_DIGEST_LANES
    assert reference.mix_hash_bytes(digest).hex() == EXP_DIGEST
    assert result == EXP_RESULT


if __name__ == "__main__":
    test_vector()
    print("PASS - reference reproduces canonical ProgPoW test vector")
