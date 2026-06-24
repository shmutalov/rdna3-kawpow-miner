"""ProgPoW random-program generator (port of libprogpow/ProgPow.cpp).

For a program seed this builds the per-period random sequence ("the program") as
a backend-neutral intermediate representation (IR): a list of operations. The IR
is the single source of truth -- it is both:
  - rendered to GLSL (the GPU compute shader) / CUDA / OpenCL, and
  - executed directly in Python (the CPU correctness reference),
so the GPU and the reference are guaranteed to run the same program.

The exact ordering of kiss99() draws is what makes the program deterministic, so
the draw order here mirrors ProgPow.cpp precisely. Do not reorder.

For a given seed the rendered CUDA matches test/kernel.cu byte-for-byte (modulo
CUDA-equivalent cosmetics), which is the generator correctness check.
"""

from .constants import (
    PROGPOW_REGS, PROGPOW_DAG_LOADS, PROGPOW_CNT_CACHE, PROGPOW_CNT_MATH,
    PROGPOW_CACHE_WORDS, FNV_PRIME, FNV_OFFSET_BASIS, MASK32,
)

CUDA = "cuda"
CL = "cl"
GLSL = "glsl"


class _Kiss99:
    """KISS99 PRNG, matching ProgPow.cpp (32-bit wraparound)."""

    __slots__ = ("z", "w", "jsr", "jcong")

    def __init__(self, z, w, jsr, jcong):
        self.z, self.w, self.jsr, self.jcong = z, w, jsr, jcong

    def __call__(self):
        self.z = (36969 * (self.z & 0xFFFF) + (self.z >> 16)) & MASK32
        self.w = (18000 * (self.w & 0xFFFF) + (self.w >> 16)) & MASK32
        mwc = ((self.z << 16) + self.w) & MASK32
        self.jsr ^= (self.jsr << 17) & MASK32
        self.jsr ^= (self.jsr >> 13)
        self.jsr ^= (self.jsr << 5) & MASK32
        self.jsr &= MASK32
        self.jcong = (69069 * self.jcong + 1234567) & MASK32
        return ((mwc ^ self.jcong) + self.jsr) & MASK32


def _fnv1a(h, d):
    return ((h ^ d) * FNV_PRIME) & MASK32


def build_program(prog_seed, cnt_cache=PROGPOW_CNT_CACHE, cnt_math=PROGPOW_CNT_MATH):
    """Build the IR for `prog_seed`.

    Returns a list of op tuples:
      ("global",)                         global DAG load (shuffle + index)
      ("cache", src, dst, r)              cache load then merge(mix[dst], data, r)
      ("math",  s1, s2, dst, r1, r2)      data=math(mix[s1],mix[s2],r1); merge(dst,r2)
      ("dagmerge", dst, i, r)             merge(mix[dst], data_dag.s[i], r)

    cnt_cache/cnt_math default to production KawPow (11/18); pass 12/20 to
    reproduce the ProgPoW-0.9.2 reference in test/kernel.cu.
    """
    seed0 = prog_seed & MASK32
    seed1 = (prog_seed >> 32) & MASK32
    fnv_hash = FNV_OFFSET_BASIS
    fnv_hash = _fnv1a(fnv_hash, seed0); z = fnv_hash
    fnv_hash = _fnv1a(fnv_hash, seed1); w = fnv_hash
    fnv_hash = _fnv1a(fnv_hash, seed0); jsr = fnv_hash
    fnv_hash = _fnv1a(fnv_hash, seed1); jcong = fnv_hash
    rnd = _Kiss99(z, w, jsr, jcong)

    mix_seq_dst = list(range(PROGPOW_REGS))
    mix_seq_cache = list(range(PROGPOW_REGS))
    for i in range(PROGPOW_REGS - 1, 0, -1):
        j = rnd() % (i + 1)
        mix_seq_dst[i], mix_seq_dst[j] = mix_seq_dst[j], mix_seq_dst[i]
        j = rnd() % (i + 1)
        mix_seq_cache[i], mix_seq_cache[j] = mix_seq_cache[j], mix_seq_cache[i]

    dst_cnt = cache_cnt = 0
    ops = [("global",)]

    for i in range(max(cnt_cache, cnt_math)):
        if i < cnt_cache:
            src = mix_seq_cache[cache_cnt % PROGPOW_REGS]; cache_cnt += 1
            dst = mix_seq_dst[dst_cnt % PROGPOW_REGS]; dst_cnt += 1
            r = rnd()
            ops.append(("cache", src, dst, r))
        if i < cnt_math:
            src_rnd = rnd() % ((PROGPOW_REGS - 1) * PROGPOW_REGS)
            src1 = src_rnd % PROGPOW_REGS
            src2 = src_rnd // PROGPOW_REGS
            if src2 >= src1:
                src2 += 1
            r1 = rnd()
            dst = mix_seq_dst[dst_cnt % PROGPOW_REGS]; dst_cnt += 1
            r2 = rnd()
            ops.append(("math", src1, src2, dst, r1, r2))

    ops.append(("dagmerge", 0, 0, rnd()))
    for i in range(1, PROGPOW_DAG_LOADS):
        dst = mix_seq_dst[dst_cnt % PROGPOW_REGS]; dst_cnt += 1
        ops.append(("dagmerge", dst, i, rnd()))
    return ops


# --- math / merge op selection (shared by renderer and interpreter) --------------

def _math_sel(r):
    return r % 11


def _merge_sel(r):
    return r % 4


def _merge_rot(r):
    return ((r >> 16) % 31) + 1


# ================================ TEXT RENDERER ==================================

def _math_text(d, a, b, r):
    sel = _math_sel(r)
    return [
        f"{d} = {a} + {b};",
        f"{d} = {a} * {b};",
        f"{d} = mul_hi({a}, {b});",
        f"{d} = min({a}, {b});",
        f"{d} = ROTL32({a}, {b} % 32);",
        f"{d} = ROTR32({a}, {b} % 32);",
        f"{d} = {a} & {b};",
        f"{d} = {a} | {b};",
        f"{d} = {a} ^ {b};",
        f"{d} = clz({a}) + clz({b});",
        f"{d} = popcount({a}) + popcount({b});",
    ][sel]


def _merge_text(a, b, r):
    sel = _merge_sel(r)
    if sel == 0:
        return f"{a} = ({a} * 33) + {b};"
    if sel == 1:
        return f"{a} = ({a} ^ {b}) * 33;"
    if sel == 2:
        return f"{a} = ROTL32({a}, {_merge_rot(r)}) ^ {b};"
    return f"{a} = ROTR32({a}, {_merge_rot(r)}) ^ {b};"


def render_loop(prog_seed, kern, ops=None, cnt_cache=PROGPOW_CNT_CACHE,
                cnt_math=PROGPOW_CNT_MATH):
    """Render the progPowLoop source for backend `kern` from IR `ops`."""
    if ops is None:
        ops = build_program(prog_seed, cnt_cache, cnt_math)
    out = [_signature(prog_seed, kern), "{\n", "dag_t data_dag;\n",
           "uint32_t offset, data;\n"]
    if kern == CL:
        out.append("uint32_t mix[PROGPOW_REGS];\n"
                   "for(int i=0; i<PROGPOW_REGS; i++)\n    mix[i] = mix_arg[i];\n")
    out.append(_lane_id(kern))

    cache_i = math_i = 0
    for op in ops:
        if op[0] == "global":
            out.append("// global load\n")
            out.append(_global_shuffle(kern))
            out.append("offset %= PROGPOW_DAG_ELEMENTS;\n")
            out.append("offset = offset * PROGPOW_LANES + (lane_id ^ loop) % PROGPOW_LANES;\n")
            if kern == GLSL:
                # g_dag is a flat uint[] SSBO: dag_t row r occupies words [r*4 .. r*4+3].
                for i in range(PROGPOW_DAG_LOADS):
                    out.append(f"data_dag.s[{i}] = g_dag[offset * PROGPOW_DAG_LOADS + {i}u];\n")
            else:
                out.append("data_dag = g_dag[offset];\n")
            out.append("// hack to prevent compiler from reordering LD and usage\n")
            out.append(_fence(kern))
        elif op[0] == "cache":
            _, src, dst, r = op
            out.append(f"// cache load {cache_i}\n"); cache_i += 1
            out.append(f"offset = mix[{src}] % PROGPOW_CACHE_WORDS;\n")
            out.append("data = c_dag[offset];\n")
            out.append(_merge_text(f"mix[{dst}]", "data", r) + "\n")
        elif op[0] == "math":
            _, s1, s2, dst, r1, r2 = op
            out.append(f"// random math {math_i}\n"); math_i += 1
            out.append(_math_text("data", f"mix[{s1}]", f"mix[{s2}]", r1) + "\n")
            out.append(_merge_text(f"mix[{dst}]", "data", r2) + "\n")
        elif op[0] == "dagmerge":
            _, dst, i, r = op
            if i == 0:
                out.append("// consume global load data\n")
                out.append("// hack to prevent compiler from reordering LD and usage\n")
                out.append(_fence(kern))
            out.append(_merge_text(f"mix[{dst}]", f"data_dag.s[{i}]", r) + "\n")

    if kern == CL:
        out.append("for(int i=0; i<PROGPOW_REGS; i++)\n    mix_arg[i] = mix[i];\n")
    out.append("}\n")
    return "".join(out)


# Backwards-compatible alias used by earlier validation
def generate_loop(prog_seed, kern, cnt_cache=PROGPOW_CNT_CACHE, cnt_math=PROGPOW_CNT_MATH):
    return render_loop(prog_seed, kern, None, cnt_cache, cnt_math)


def _signature(prog_seed, kern):
    if kern == CUDA:
        return (f"// Inner loop for prog_seed {prog_seed}\n"
                "__device__ __forceinline__ void progPowLoop(const uint32_t loop,\n"
                "    uint32_t mix[PROGPOW_REGS],\n"
                "    const dag_t *g_dag,\n"
                "    const uint32_t c_dag[PROGPOW_CACHE_WORDS],\n"
                "    const bool hack_false)\n")
    if kern == CL:
        return (f"// Inner loop for prog_seed {prog_seed}\n"
                "inline void progPowLoop(const uint32_t loop,\n"
                "        volatile uint32_t mix_arg[PROGPOW_REGS],\n"
                "        __global const dag_t *g_dag,\n"
                "        __local const uint32_t c_dag[PROGPOW_CACHE_WORDS],\n"
                "        __local uint64_t share[GROUP_SHARE],\n"
                "        const bool hack_false)\n")
    return (f"// Inner loop for prog_seed {prog_seed}\n"
            "void progPowLoop(const uint32_t loop, inout uint32_t mix[PROGPOW_REGS],\n"
            "        uint32_t lane_id, uint32_t sg_base, bool hack_false)\n")


def _lane_id(kern):
    if kern == CUDA:
        return "const uint32_t lane_id = threadIdx.x & (PROGPOW_LANES-1);\n"
    if kern == CL:
        return ("const uint32_t lane_id = get_local_id(0) & (PROGPOW_LANES-1);\n"
                "const uint32_t group_id = get_local_id(0) / PROGPOW_LANES;\n")
    return ""  # GLSL: lane_id / sg_base passed in by caller


def _global_shuffle(kern):
    if kern == CUDA:
        return "offset = SHFL(mix[0], loop%PROGPOW_LANES, PROGPOW_LANES);\n"
    if kern == CL:
        return ("if(lane_id == (loop % PROGPOW_LANES))\n    share[group_id] = mix[0];\n"
                "barrier(CLK_LOCAL_MEM_FENCE);\noffset = share[group_id];\n")
    # GLSL/RDNA3: subgroupShuffle straight from registers -- no LDS, no barrier.
    return "offset = subgroupShuffle(mix[0], sg_base + (loop % PROGPOW_LANES));\n"


def _fence(kern):
    if kern == CUDA:
        return "if (hack_false) __threadfence_block();\n"
    if kern == CL:
        return "if (hack_false) barrier(CLK_LOCAL_MEM_FENCE);\n"
    return "if (hack_false) subgroupBarrier();\n"


# ================================ PYTHON INTERPRETER =============================

def _rotl32(x, n):
    n &= 31
    return ((x << n) | (x >> (32 - n))) & MASK32 if n else x & MASK32


def _rotr32(x, n):
    n &= 31
    return ((x >> n) | (x << (32 - n))) & MASK32 if n else x & MASK32


def _mul_hi(a, b):
    return ((a * b) >> 32) & MASK32


def _clz(x):
    return 32 if x == 0 else 31 - x.bit_length() + 1  # 32 - bit_length


def _popcount(x):
    return bin(x & MASK32).count("1")


def _math_exec(a, b, r):
    sel = _math_sel(r)
    a &= MASK32; b &= MASK32
    if sel == 0:
        return (a + b) & MASK32
    if sel == 1:
        return (a * b) & MASK32
    if sel == 2:
        return _mul_hi(a, b)
    if sel == 3:
        return min(a, b)
    if sel == 4:
        return _rotl32(a, b % 32)
    if sel == 5:
        return _rotr32(a, b % 32)
    if sel == 6:
        return a & b
    if sel == 7:
        return a | b
    if sel == 8:
        return a ^ b
    if sel == 9:
        return (_clz(a) + _clz(b)) & MASK32
    return (_popcount(a) + _popcount(b)) & MASK32


def _merge_exec(a, b, r):
    sel = _merge_sel(r)
    a &= MASK32; b &= MASK32
    if sel == 0:
        return ((a * 33) + b) & MASK32
    if sel == 1:
        return ((a ^ b) * 33) & MASK32
    if sel == 2:
        return _rotl32(a, _merge_rot(r)) ^ b
    return _rotr32(a, _merge_rot(r)) ^ b


def run_loop(ops, loop, mix_lanes, c_dag, dag_entry_for_loop, dag_elements):
    """Execute one progPowLoop iteration across all 16 lanes (CPU reference).

    mix_lanes: list[16] of list[32] uint32 (modified in place).
    c_dag: the 4096-word cache.
    dag_entry_for_loop(entry_index) -> list[PROGPOW_DAG_LOADS] uint32 (one DAG row).
    """
    from .constants import PROGPOW_LANES, PROGPOW_DAG_LOADS

    # Global load: lane (loop % LANES) broadcasts mix[0]; each lane reads its DAG row.
    src_lane = loop % PROGPOW_LANES
    base_offset = mix_lanes[src_lane][0] % dag_elements
    data_dag = []
    for lane in range(PROGPOW_LANES):
        off = base_offset * PROGPOW_LANES + ((lane ^ loop) % PROGPOW_LANES)
        data_dag.append(dag_entry_for_loop(off))

    for lane in range(PROGPOW_LANES):
        mix = mix_lanes[lane]
        dd = data_dag[lane]
        for op in ops:
            if op[0] == "global":
                continue
            if op[0] == "cache":
                _, src, dst, r = op
                data = c_dag[mix[src] % PROGPOW_CACHE_WORDS]
                mix[dst] = _merge_exec(mix[dst], data, r)
            elif op[0] == "math":
                _, s1, s2, dst, r1, r2 = op
                data = _math_exec(mix[s1], mix[s2], r1)
                mix[dst] = _merge_exec(mix[dst], data, r2)
            elif op[0] == "dagmerge":
                _, dst, i, r = op
                mix[dst] = _merge_exec(mix[dst], dd[i], r)
