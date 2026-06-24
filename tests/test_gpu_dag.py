"""GPU correctness: the DAG-generation shader must reproduce ethash dataset items.

Generates the epoch-1 light cache on the host, uploads it, runs ethash_dag.comp
for items 0..255 on the GPU, and checks the resulting cache words against the
canonical result.log values (cdag[0..15], cdag[4080..4095]).
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdna3_kawpow import ethash, shader_compiler as sc  # noqa: E402
from rdna3_kawpow.vkhost import VulkanDevice, ComputePipeline  # noqa: E402

EXP_HEAD = [0xb3e35467, 0xae7402e3, 0x8522a782, 0xa2d8353b, 0xff4723bd, 0xbfbc05ee,
            0xde6944de, 0xf0d2b5b8, 0xc74cbad3, 0xb100f797, 0x05bc60be, 0x4f40840b,
            0x35e47268, 0x9cd6f993, 0x6a0e4659, 0xb838e46e]
EXP_TAIL = [0xbde0c650, 0x57cba482, 0x54877c9d, 0xf9fdc423, 0xfb65141b, 0x55074ca4,
            0xc7dd116e, 0xbc1737d1, 0x126e8847, 0xb16983b2, 0xf80c058e, 0xe0ad53b5,
            0xd5f3e840, 0xff1bdd89, 0x35660a19, 0x73244193]


def main():
    print("Generating epoch-1 light cache (host)...")
    # result.log is vanilla ProgPoW 0.9.2 (256 DAG parents).
    lc = ethash.LightCache(30000, epoch_length=ethash.ETHEREUM_EPOCH_LENGTH, parents=256)
    light_items = len(lc.cache)
    light_bytes = b"".join(lc.cache)              # item i -> 64 bytes
    print(f"light_items={light_items} ({len(light_bytes)//1024//1024} MiB), dag_items={lc.dag_items}")

    n_items = 256                                  # enough for cdag[0..4095]
    dev = VulkanDevice()
    print(dev.summary())
    spv = sc.compile_dag(light_items, lc.dag_items, parents=256)
    pipe = ComputePipeline(dev, spv, num_bindings=2, push_const_size=4,
                           local_size=64, required_subgroup_size=32)

    light_buf = dev.make_buffer(len(light_bytes))
    light_buf.write(light_bytes)
    dag_buf = dev.make_buffer(n_items * 64)
    dag_buf.write(b"\x00" * (n_items * 64))

    pipe.bind([light_buf, dag_buf])
    groups = (n_items + 63) // 64
    dev.dispatch(pipe, group_count_x=groups, push_constants=struct.pack("<I", 0))

    raw = dag_buf.read(n_items * 64)
    cdag = list(struct.unpack("<%dI" % (n_items * 16), raw))
    head_ok = cdag[0:16] == EXP_HEAD
    tail_ok = cdag[4080:4096] == EXP_TAIL
    print("cdag[0]=%08x (exp b3e35467) | cdag[4080]=%08x (exp bde0c650)" % (cdag[0], cdag[4080]))
    print("head match:", head_ok, "| tail match:", tail_ok)
    ok = head_ok and tail_ok
    print("RESULT:", "PASS - GPU DAG generation matches ethash" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
