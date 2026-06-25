//! ProgPoW random-program generator (port of `rdna3_kawpow/progpow.py`).
//!
//! For a program seed this builds the per-period random sequence ("the program")
//! as a backend-neutral IR (`Vec<Op>`). That IR is the single source of truth: it
//! is both rendered to GLSL (the GPU shader) and executed directly here (the CPU
//! reference), so the GPU and the reference run the identical program.
//!
//! The exact ordering of `Kiss99` draws is what makes the program deterministic,
//! so the draw order here mirrors `ProgPow.cpp` precisely. DO NOT REORDER DRAWS.

use std::fmt::Write as _;

use crate::constants::{
    FNV_OFFSET_BASIS, FNV_PRIME, PROGPOW_CNT_CACHE, PROGPOW_CNT_MATH, PROGPOW_DAG_LOADS,
    PROGPOW_REGS,
};

/// One operation in the per-period program IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// Global DAG load (shuffle + index).
    Global,
    /// Cache load then merge(mix[dst], data, r).
    Cache { src: usize, dst: usize, r: u32 },
    /// data = math(mix[s1], mix[s2], r1); merge(mix[dst], data, r2).
    Math {
        s1: usize,
        s2: usize,
        dst: usize,
        r1: u32,
        r2: u32,
    },
    /// merge(mix[dst], data_dag.s[i], r).
    DagMerge { dst: usize, i: usize, r: u32 },
}

/// KISS99 PRNG, matching `ProgPow.cpp` (32-bit wraparound).
#[derive(Clone, Copy)]
pub struct Kiss99 {
    z: u32,
    w: u32,
    jsr: u32,
    jcong: u32,
}

impl Kiss99 {
    pub fn new(z: u32, w: u32, jsr: u32, jcong: u32) -> Self {
        Kiss99 { z, w, jsr, jcong }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.z = (36969u32.wrapping_mul(self.z & 0xFFFF)).wrapping_add(self.z >> 16);
        self.w = (18000u32.wrapping_mul(self.w & 0xFFFF)).wrapping_add(self.w >> 16);
        let mwc = (self.z << 16).wrapping_add(self.w);
        self.jsr ^= self.jsr << 17;
        self.jsr ^= self.jsr >> 13;
        self.jsr ^= self.jsr << 5;
        self.jcong = 69069u32.wrapping_mul(self.jcong).wrapping_add(1234567);
        (mwc ^ self.jcong).wrapping_add(self.jsr)
    }
}

#[inline]
fn fnv1a(h: u32, d: u32) -> u32 {
    (h ^ d).wrapping_mul(FNV_PRIME)
}

/// Build the IR for `prog_seed`. `cnt_cache`/`cnt_math` default to production
/// KawPow (11/18); pass 12/20 to reproduce the ProgPoW-0.9.2 reference.
pub fn build_program(prog_seed: u64, cnt_cache: usize, cnt_math: usize) -> Vec<Op> {
    let seed0 = prog_seed as u32;
    let seed1 = (prog_seed >> 32) as u32;
    let mut fnv_hash = FNV_OFFSET_BASIS;
    fnv_hash = fnv1a(fnv_hash, seed0);
    let z = fnv_hash;
    fnv_hash = fnv1a(fnv_hash, seed1);
    let w = fnv_hash;
    fnv_hash = fnv1a(fnv_hash, seed0);
    let jsr = fnv_hash;
    fnv_hash = fnv1a(fnv_hash, seed1);
    let jcong = fnv_hash;
    let mut rnd = Kiss99::new(z, w, jsr, jcong);

    let mut mix_seq_dst: Vec<usize> = (0..PROGPOW_REGS).collect();
    let mut mix_seq_cache: Vec<usize> = (0..PROGPOW_REGS).collect();
    for i in (1..PROGPOW_REGS).rev() {
        let j = (rnd.next_u32() as usize) % (i + 1);
        mix_seq_dst.swap(i, j);
        let j = (rnd.next_u32() as usize) % (i + 1);
        mix_seq_cache.swap(i, j);
    }

    let mut dst_cnt = 0usize;
    let mut cache_cnt = 0usize;
    let mut ops = vec![Op::Global];

    for i in 0..cnt_cache.max(cnt_math) {
        if i < cnt_cache {
            let src = mix_seq_cache[cache_cnt % PROGPOW_REGS];
            cache_cnt += 1;
            let dst = mix_seq_dst[dst_cnt % PROGPOW_REGS];
            dst_cnt += 1;
            let r = rnd.next_u32();
            ops.push(Op::Cache { src, dst, r });
        }
        if i < cnt_math {
            let regs = PROGPOW_REGS as u32;
            let src_rnd = rnd.next_u32() % ((regs - 1) * regs);
            let src1 = (src_rnd % regs) as usize;
            let mut src2 = (src_rnd / regs) as usize;
            if src2 >= src1 {
                src2 += 1;
            }
            let r1 = rnd.next_u32();
            let dst = mix_seq_dst[dst_cnt % PROGPOW_REGS];
            dst_cnt += 1;
            let r2 = rnd.next_u32();
            ops.push(Op::Math {
                s1: src1,
                s2: src2,
                dst,
                r1,
                r2,
            });
        }
    }

    ops.push(Op::DagMerge {
        dst: 0,
        i: 0,
        r: rnd.next_u32(),
    });
    for i in 1..PROGPOW_DAG_LOADS {
        let dst = mix_seq_dst[dst_cnt % PROGPOW_REGS];
        dst_cnt += 1;
        ops.push(Op::DagMerge {
            dst,
            i,
            r: rnd.next_u32(),
        });
    }
    ops
}

/// Build with production-default op counts.
pub fn build_program_default(prog_seed: u64) -> Vec<Op> {
    build_program(prog_seed, PROGPOW_CNT_CACHE, PROGPOW_CNT_MATH)
}

// --- math / merge op selection (shared by renderer and interpreter) ---

#[inline]
fn math_sel(r: u32) -> u32 {
    r % 11
}
#[inline]
fn merge_sel(r: u32) -> u32 {
    r % 4
}
#[inline]
fn merge_rot(r: u32) -> u32 {
    ((r >> 16) % 31) + 1
}

// ================================ GLSL RENDERER ==================================

fn math_text(d: &str, a: &str, b: &str, r: u32) -> String {
    match math_sel(r) {
        0 => format!("{d} = {a} + {b};"),
        1 => format!("{d} = {a} * {b};"),
        2 => format!("{d} = mul_hi({a}, {b});"),
        3 => format!("{d} = min({a}, {b});"),
        4 => format!("{d} = ROTL32({a}, {b} % 32);"),
        5 => format!("{d} = ROTR32({a}, {b} % 32);"),
        6 => format!("{d} = {a} & {b};"),
        7 => format!("{d} = {a} | {b};"),
        8 => format!("{d} = {a} ^ {b};"),
        9 => format!("{d} = clz({a}) + clz({b});"),
        _ => format!("{d} = popcount({a}) + popcount({b});"),
    }
}

fn merge_text(a: &str, b: &str, r: u32) -> String {
    match merge_sel(r) {
        0 => format!("{a} = ({a} * 33) + {b};"),
        1 => format!("{a} = ({a} ^ {b}) * 33;"),
        2 => format!("{a} = ROTL32({a}, {}) ^ {b};", merge_rot(r)),
        _ => format!("{a} = ROTR32({a}, {}) ^ {b};", merge_rot(r)),
    }
}

/// Render the `progPowLoop` GLSL source for `ops` (which must come from
/// `build_program(prog_seed, ...)`). Byte-identical to the Python GLSL renderer.
pub fn render_loop_glsl(prog_seed: u64, ops: &[Op]) -> String {
    let mut out = String::new();
    let _ = write!(
        out,
        "// Inner loop for prog_seed {prog_seed}\n\
         void progPowLoop(const uint32_t loop, inout uint32_t mix[PROGPOW_REGS],\n        \
         uint32_t lane_id, uint32_t sg_base, bool hack_false)\n"
    );
    out.push_str("{\n");
    out.push_str("dag_t data_dag;\n");
    out.push_str("uint32_t offset, data;\n");

    let mut cache_i = 0usize;
    let mut math_i = 0usize;
    for op in ops {
        match *op {
            Op::Global => {
                out.push_str("// global load\n");
                out.push_str(
                    "offset = subgroupShuffle(mix[0], sg_base + (loop % PROGPOW_LANES));\n",
                );
                out.push_str("offset %= PROGPOW_DAG_ELEMENTS;\n");
                out.push_str(
                    "offset = offset * PROGPOW_LANES + (lane_id ^ loop) % PROGPOW_LANES;\n",
                );
                for i in 0..PROGPOW_DAG_LOADS {
                    let _ = writeln!(
                        out,
                        "data_dag.s[{i}] = g_dag[offset * PROGPOW_DAG_LOADS + {i}u];"
                    );
                }
                out.push_str("// hack to prevent compiler from reordering LD and usage\n");
                out.push_str("if (hack_false) subgroupBarrier();\n");
            }
            Op::Cache { src, dst, r } => {
                let _ = writeln!(out, "// cache load {cache_i}");
                cache_i += 1;
                let _ = writeln!(out, "offset = mix[{src}] % PROGPOW_CACHE_WORDS;");
                out.push_str("data = c_dag[offset];\n");
                let _ = writeln!(out, "{}", merge_text(&format!("mix[{dst}]"), "data", r));
            }
            Op::Math { s1, s2, dst, r1, r2 } => {
                let _ = writeln!(out, "// random math {math_i}");
                math_i += 1;
                let _ = writeln!(
                    out,
                    "{}",
                    math_text("data", &format!("mix[{s1}]"), &format!("mix[{s2}]"), r1)
                );
                let _ = writeln!(out, "{}", merge_text(&format!("mix[{dst}]"), "data", r2));
            }
            Op::DagMerge { dst, i, r } => {
                if i == 0 {
                    out.push_str("// consume global load data\n");
                    out.push_str("// hack to prevent compiler from reordering LD and usage\n");
                    out.push_str("if (hack_false) subgroupBarrier();\n");
                }
                let _ = writeln!(
                    out,
                    "{}",
                    merge_text(&format!("mix[{dst}]"), &format!("data_dag.s[{i}]"), r)
                );
            }
        }
    }
    out.push_str("}\n");
    out
}

// ================================ INTERPRETER ===================================

#[inline]
fn mul_hi(a: u32, b: u32) -> u32 {
    (((a as u64) * (b as u64)) >> 32) as u32
}

fn math_exec(a: u32, b: u32, r: u32) -> u32 {
    match math_sel(r) {
        0 => a.wrapping_add(b),
        1 => a.wrapping_mul(b),
        2 => mul_hi(a, b),
        3 => a.min(b),
        4 => a.rotate_left(b % 32),
        5 => a.rotate_right(b % 32),
        6 => a & b,
        7 => a | b,
        8 => a ^ b,
        9 => a.leading_zeros().wrapping_add(b.leading_zeros()),
        _ => a.count_ones().wrapping_add(b.count_ones()),
    }
}

fn merge_exec(a: u32, b: u32, r: u32) -> u32 {
    match merge_sel(r) {
        0 => a.wrapping_mul(33).wrapping_add(b),
        1 => (a ^ b).wrapping_mul(33),
        2 => a.rotate_left(merge_rot(r)) ^ b,
        _ => a.rotate_right(merge_rot(r)) ^ b,
    }
}

/// Execute one `progPowLoop` iteration across all 16 lanes (CPU reference).
///
/// `mix_lanes`: 16 lanes x 32 u32 (modified in place). `cdag`: the 4096-word cache.
/// `dag_row(entry_index)` returns one 4-word DAG row.
pub fn run_loop<F: Fn(u32) -> [u32; 4]>(
    ops: &[Op],
    loop_idx: u32,
    mix_lanes: &mut [[u32; crate::constants::PROGPOW_REGS]],
    cdag: &[u32],
    dag_row: F,
    dag_elements: u32,
) {
    use crate::constants::{PROGPOW_CACHE_WORDS, PROGPOW_LANES};
    let lanes = PROGPOW_LANES;

    let src_lane = (loop_idx % lanes) as usize;
    let base_offset = (mix_lanes[src_lane][0] as u64 % dag_elements as u64) as u32;
    let mut data_dag = [[0u32; 4]; 16];
    for lane in 0..lanes {
        let off = base_offset as u64 * lanes as u64 + ((lane ^ loop_idx) % lanes) as u64;
        data_dag[lane as usize] = dag_row(off as u32);
    }

    let cache_words = PROGPOW_CACHE_WORDS as u32;
    for lane in 0..lanes as usize {
        let dd = data_dag[lane];
        let mix = &mut mix_lanes[lane];
        for op in ops {
            match *op {
                Op::Global => {}
                Op::Cache { src, dst, r } => {
                    let data = cdag[(mix[src] % cache_words) as usize];
                    mix[dst] = merge_exec(mix[dst], data, r);
                }
                Op::Math { s1, s2, dst, r1, r2 } => {
                    let data = math_exec(mix[s1], mix[s2], r1);
                    mix[dst] = merge_exec(mix[dst], data, r2);
                }
                Op::DagMerge { dst, i, r } => {
                    mix[dst] = merge_exec(mix[dst], dd[i], r);
                }
            }
        }
    }
}
