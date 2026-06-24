"""GPU correctness: the Vulkan search shader must reproduce the validated CPU
reference, bit-for-bit, on real hardware.

Uses a synthetic DAG shared between GPU and reference so the search logic
(keccak_f800, fill_mix, the injected per-period loop, subgroupShuffle exchange,
digest reduction, final keccak) is exercised independently of ethash. The CPU
reference is itself pinned to the canonical vector by test_reference_vector.py.
"""

import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdna3_kawpow import reference, keccak, shader_compiler as sc  # noqa: E402
from rdna3_kawpow.vkhost import VulkanDevice  # noqa: E402

DAG_ELEMENTS = 2048           # tiny synthetic DAG
LOCAL_SIZE = 64
N = 64                        # nonces (one workgroup)
MAX_OUTPUTS = 128
START_NONCE = 0x123456789ABCDEF0
HEADER = bytes.fromhex(
    "ffeeddccbbaa9988776655443322110000112233445566778899aabbccddeeff")


class SyntheticLight:
    """Stand-in for ethash.LightCache backed by a fixed word array."""

    def __init__(self, dag_elements, words):
        self.dag_elements = dag_elements
        self._w = words

    def dag_row(self, row):
        b = row * 4
        return self._w[b:b + 4]

    def progpow_cache(self):
        return self._w[:4096]


def run(variant, prog_seed, cnt_cache, cnt_math):
    words = np.random.default_rng(0xC0FFEE).integers(
        0, 2 ** 32, size=DAG_ELEMENTS * 64, dtype=np.uint32)
    words_list = words.tolist()
    light = SyntheticLight(DAG_ELEMENTS, words_list)

    # CPU reference digests for every nonce.
    ref = {}
    for gid in range(N):
        digest, _ = reference.hash_one(
            HEADER, START_NONCE + gid, light, prog_seed, variant,
            cnt_cache=cnt_cache, cnt_math=cnt_math)
        ref[gid] = tuple(digest)

    # GPU.
    dev = VulkanDevice()
    print(dev.summary())
    spv = sc.compile_search(prog_seed, DAG_ELEMENTS, variant, MAX_OUTPUTS,
                            cnt_cache, cnt_math)
    from rdna3_kawpow.vkhost import ComputePipeline
    pipe = ComputePipeline(dev, spv, num_bindings=3, push_const_size=16,
                           local_size=LOCAL_SIZE, required_subgroup_size=32)

    hdr_buf = dev.make_buffer(32)
    hdr_buf.write(HEADER)
    dag_buf = dev.make_buffer(DAG_ELEMENTS * 64 * 4)
    dag_buf.write(words.tobytes())
    out_size = 16 + MAX_OUTPUTS * 9 * 4
    out_buf = dev.make_buffer(out_size)
    out_buf.write(b"\x00" * out_size)

    pipe.bind([hdr_buf, dag_buf, out_buf])
    push = struct.pack("<4I", START_NONCE & 0xFFFFFFFF, START_NONCE >> 32,
                       0xFFFFFFFF, 0xFFFFFFFF)  # target = max -> every nonce writes
    dev.dispatch(pipe, group_count_x=1, push_constants=push)

    raw = out_buf.read(out_size)
    count, hashCount, abort, _pad = struct.unpack("<4I", raw[:16])
    gpu = {}
    for s in range(min(count, MAX_OUTPUTS)):
        base = 16 + s * 9 * 4
        vals = struct.unpack("<9I", raw[base:base + 36])
        gpu[vals[0]] = vals[1:9]

    mism = 0
    for gid in range(N):
        if gpu.get(gid) != ref[gid]:
            mism += 1
            if mism <= 3:
                print(f"  gid {gid}: gpu={gpu.get(gid)} ref={ref[gid]}")
    print(f"[{variant}] count={count} hashCount={hashCount} nonces={N} mismatches={mism}")
    return mism == 0 and count >= N


def main():
    ok_v = run(keccak.VANILLA, 600, 12, 20)
    ok_k = run(keccak.KAWPOW, 12345, 11, 18)
    print("RESULT:", "PASS - GPU search matches CPU reference" if (ok_v and ok_k) else "FAIL")
    return 0 if (ok_v and ok_k) else 1


if __name__ == "__main__":
    sys.exit(main())
