"""Production KawPow regression test against the authoritative cpp-kawpow vectors
(RavenCommunity/cpp-kawpow test/unittests/progpow_test_vectors.hpp).

These pin the *standard Ravencoin KawPow* path (512 DAG parents, Ravencoin keccak,
direct seed, 11/18). All four blocks here are epoch 0, so one light cache suffices.
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdna3_kawpow import ethash, reference, keccak  # noqa: E402
from rdna3_kawpow.constants import PROGPOW_PERIOD  # noqa: E402

# (block, header_hex, nonce, mix_hex, final_hex)
VECTORS = [
    (0, "00" * 32, 0x0000000000000000,
     "6e97b47b134fda0c7888802988e1a373affeb28bcd813b6e9a0fc669c935d03a",
     "e601a7257a70dc48fccc97a7330d704d776047623b92883d77111fb36870f3d1"),
    (49, "63155f732f2bf556967f906155b510c917e48e99685ead76ea83f4eca03ab12b",
     0x0000000007073c07,
     "d36f7e815ee09e74eceb9c96993a3d681edf2bf0921fc7bb710364042db99777",
     "e7ced124598fd2500a55ad9f9f48e3569327fe50493c77a4ac9799b96efb9463"),
    (50, "9e7248f20914913a73d80a70174c331b1d34f260535ac3631d770e656b5dd922",
     0x00000000076e482e,
     "d6dc634ae837e2785b347648ea515e25e5d8821ae0b95e1c2a9c2d497e0dcfbd",
     "ab0ad7ef8d8ee317dd12d10310aceed7321d34fb263791c2de5776a6658d177e"),
    (99, "de37e1824c86d35d154cf65a88de6d9286aec4f7f10c3fc9f0fa1bcc2687188d",
     0x000000003917afab,
     "fa706860e5e0e830d5d1d7157e5bea7f5f8a350c7c8612ac1d1fcf2974d64244",
     "aa85340690f2e907054324a5021937910e15edfd1ef1577231843e7d32ec3a61"),
]


def test_vectors():
    lc = ethash.LightCache(0, epoch_length=7500)  # epoch 0, 512 parents (default)
    for block, hdr_hex, nonce, exp_mix, exp_final in VECTORS:
        header = bytes.fromhex(hdr_hex)
        prog_seed = block // PROGPOW_PERIOD
        digest, result = reference.hash_one(
            header, nonce, lc, prog_seed=prog_seed, variant=keccak.KAWPOW)
        got_mix = struct.pack("<8I", *digest).hex()
        exp_result64 = int.from_bytes(bytes.fromhex(exp_final)[:8], "big")
        assert got_mix == exp_mix, f"block {block} mix: {got_mix} != {exp_mix}"
        assert result == exp_result64, f"block {block} result: {result:016x} != {exp_result64:016x}"


if __name__ == "__main__":
    test_vectors()
    print("PASS - all cpp-kawpow vectors reproduced (standard Ravencoin KawPow)")
