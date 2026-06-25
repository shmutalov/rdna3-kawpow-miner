"""Generate the IR + GLSL differential fixture from the Python reference.

Dumps, for a spread of program seeds and both op-count configs, the exact
`build_program` IR (tagged op arrays) and the rendered `progPowLoop` GLSL. The
Rust differential test regenerates these and asserts byte-identical output, which
pins the KISS99 draw order AND the GLSL renderer against the Python source of
truth across hundreds of seeds.

Run from the repo root (no GPU / numpy / pycryptodome needed -- only progpow.py):
    python rust/tools/gen_fixtures.py
Writes: rust/tests/fixtures/ir.json
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO_ROOT)

from rdna3_kawpow import progpow  # noqa: E402

# A spread that exercises many distinct draw sequences, including seeds whose
# high 32 bits (seed1) are non-zero.
SEEDS = (
    list(range(0, 64))
    + [100, 200, 255, 256, 600, 1000, 4096, 12345, 65535, 0xFFFFFFFF]
    + [0x1_0000_0000, 0xDEADBEEF, 0x1234_5678_9ABC_DEF0, 0xFFFF_FFFF_FFFF_FFFF]
)

CONFIGS = [
    {"name": "kawpow", "cnt_cache": 11, "cnt_math": 18},   # production KawPow
    {"name": "vanilla", "cnt_cache": 12, "cnt_math": 20},  # ProgPoW 0.9.2 reference
]


def op_to_array(op):
    # Python IR tuples are already ("global",) / ("cache",s,d,r) /
    # ("math",s1,s2,dst,r1,r2) / ("dagmerge",dst,i,r). Emit as JSON arrays.
    return [op[0]] + [int(x) for x in op[1:]]


def main():
    configs = []
    for cfg in CONFIGS:
        seeds_out = []
        for seed in SEEDS:
            ops = progpow.build_program(seed, cfg["cnt_cache"], cfg["cnt_math"])
            glsl = progpow.render_loop(seed, progpow.GLSL, ops,
                                       cfg["cnt_cache"], cfg["cnt_math"])
            seeds_out.append({
                "seed": seed,
                "ops": [op_to_array(op) for op in ops],
                "glsl": glsl,
            })
        configs.append({**cfg, "seeds": seeds_out})

    out_dir = os.path.join(REPO_ROOT, "rust", "tests", "fixtures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ir.json")
    with open(out_path, "w", newline="\n") as f:
        json.dump({"configs": configs}, f, indent=1)
    n = len(SEEDS) * len(CONFIGS)
    print(f"wrote {out_path}: {n} (seed,config) entries")


if __name__ == "__main__":
    main()
