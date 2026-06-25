//! Assemble GLSL compute shaders and compile them to SPIR-V.
//!
//! Port of `rdna3_kawpow/shader_compiler.py`. Phase 0 uses the same architecture
//! as the Python host: shell out to `glslc` (Vulkan SDK / bundled on the rig) and
//! cache the resulting SPIR-V on disk keyed by source content. The shaders are
//! embedded at build time via `include_str!` so the binary is self-contained.
//!
//! A future hardening step may switch to in-process `shaderc` (statically linked)
//! to drop the external `glslc` dependency; the disk-cache + assembly logic here
//! stays the same either way.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};

use crate::constants::{PROGPOW_CNT_CACHE, PROGPOW_CNT_MATH, RAVENCOIN_RNDC};
use crate::keccak::Variant;
use crate::progpow;

/// The GLSL DAG-generation shader (host injects `__DEFINES__`). The template path
/// reaches into the shared `rdna3_kawpow/shaders` tree so the Rust and Python
/// hosts compile byte-identical GLSL during migration (single source of truth).
pub const ETHASH_DAG_GLSL: &str =
    include_str!("../../rdna3_kawpow/shaders/ethash_dag.comp");
/// The per-period search shader template (placeholders filled by the IR renderer
/// in Phase 1: `__DEFINES__`, `__PROGPOW_LOOP__`, `__SEED_DERIVE__`,
/// `__TAIL_INIT__`, `__FINAL_FILL__`).
pub const SEARCH_GLSL_TMPL: &str =
    include_str!("../../rdna3_kawpow/shaders/progpow_search.comp.tmpl");

fn cache_dir() -> PathBuf {
    std::env::temp_dir().join("rdna3_kawpow_spv")
}

/// A located runtime GLSL->SPIR-V compiler. `glslc` (Vulkan SDK) is preferred for
/// dev; `glslangValidator` (apt `glslang-tools`) is the bundled rig compiler.
enum Compiler {
    Glslc(PathBuf),
    Glslang(PathBuf),
}

impl Compiler {
    fn key(&self) -> &'static str {
        match self {
            Compiler::Glslc(_) => "glslc",
            Compiler::Glslang(_) => "glslang",
        }
    }
}

fn exe_suffix(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

/// Locate a compiler: bundled next to our binary first (HiveOS package), then
/// PATH, then `$VULKAN_SDK/Bin`. `glslc` wins over `glslangValidator` when both
/// are present (it optimizes the SPIR-V).
fn find_compiler() -> Result<Compiler> {
    // RDNA3_KAWPOW_COMPILER=glslangValidator (or glslc) forces a preference;
    // otherwise glslc wins when both are present (it optimizes the SPIR-V).
    let prefer_glslang = std::env::var("RDNA3_KAWPOW_COMPILER")
        .map(|v| v.to_ascii_lowercase().starts_with("glslang"))
        .unwrap_or(false);
    let names: [(&str, bool); 2] = if prefer_glslang {
        [("glslangValidator", false), ("glslc", true)]
    } else {
        [("glslc", true), ("glslangValidator", false)]
    };

    // 1) Same directory as the running executable (bundled compiler).
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for (name, is_glslc) in names {
                let cand = dir.join(exe_suffix(name));
                if cand.is_file() {
                    return Ok(wrap(cand, is_glslc));
                }
            }
        }
    }
    // 2) PATH.
    if let Some(path_var) = std::env::var_os("PATH") {
        for (name, is_glslc) in names {
            for dir in std::env::split_paths(&path_var) {
                let cand = dir.join(exe_suffix(name));
                if cand.is_file() {
                    return Ok(wrap(cand, is_glslc));
                }
            }
        }
    }
    // 3) Vulkan SDK.
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        for (name, is_glslc) in names {
            let cand = PathBuf::from(&sdk).join("Bin").join(exe_suffix(name));
            if cand.exists() {
                return Ok(wrap(cand, is_glslc));
            }
        }
    }
    Err(anyhow!(
        "no SPIR-V compiler found (need glslc or glslangValidator on PATH, in the \
         binary's directory, or in $VULKAN_SDK/Bin)"
    ))
}

fn wrap(path: PathBuf, is_glslc: bool) -> Compiler {
    if is_glslc {
        Compiler::Glslc(path)
    } else {
        Compiler::Glslang(path)
    }
}

/// Back-compat shim for older callers/tests.
pub fn find_glslc() -> Result<PathBuf> {
    match find_compiler()? {
        Compiler::Glslc(p) | Compiler::Glslang(p) => Ok(p),
    }
}

fn content_key(compiler: &str, stage: &str, optimize: bool, src: &str) -> String {
    // Stable (fixed-seed) hash -- only needs to be a unique cache key, not crypto.
    let mut h = DefaultHasher::new();
    compiler.hash(&mut h);
    stage.hash(&mut h);
    optimize.hash(&mut h);
    src.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Compile GLSL `src` to SPIR-V bytes, caching the result on disk.
pub fn compile_glsl(src: &str, stage: &str, optimize: bool) -> Result<Vec<u8>> {
    let compiler = find_compiler()?;
    let dir = cache_dir();
    fs::create_dir_all(&dir).with_context(|| format!("creating cache dir {dir:?}"))?;
    let key = content_key(compiler.key(), stage, optimize, src);
    let spv_path = dir.join(format!("{key}.spv"));
    if spv_path.exists() {
        return fs::read(&spv_path).context("reading cached SPIR-V");
    }

    let src_path = dir.join(format!("{key}.{stage}"));
    fs::write(&src_path, src).context("writing GLSL source")?;

    let mut cmd = match &compiler {
        Compiler::Glslc(p) => {
            let mut c = Command::new(p);
            c.arg("--target-env=vulkan1.3")
                .arg(format!("-fshader-stage={stage}"));
            if optimize {
                c.arg("-O");
            }
            c.arg(&src_path).arg("-o").arg(&spv_path);
            c
        }
        Compiler::Glslang(p) => {
            // glslangValidator: -V emits a Vulkan SPIR-V binary; -S sets the stage.
            // target vulkan1.2 (SPIR-V 1.5): covers our subgroup + int64 use and is
            // accepted by older glslang (the apt build); valid on the 1.3 device.
            // (No -O; the AMD driver optimizes the SPIR-V on ingestion anyway.)
            let mut c = Command::new(p);
            c.arg("-V")
                .arg("--target-env")
                .arg("vulkan1.2")
                .arg("-S")
                .arg(stage)
                .arg(&src_path)
                .arg("-o")
                .arg(&spv_path);
            c
        }
    };

    let out = cmd.output().context("failed to run the SPIR-V compiler")?;
    if !out.status.success() {
        bail!(
            "shader compile failed:\n{}{}\n--- source ---\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
            numbered(src)
        );
    }
    fs::read(&spv_path).context("reading compiled SPIR-V")
}

// --- variant-specific keccak fills injected into the search template ---

fn tail_init(variant: Variant) -> String {
    match variant {
        Variant::Vanilla => String::new(), // tail already zero
        Variant::Kawpow => (0..15)
            .map(|i| format!("st[{}] = 0x{:08x}u;", 10 + i, RAVENCOIN_RNDC[i]))
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn seed_derive(variant: Variant) -> &'static str {
    match variant {
        // kawpowminer 0.9.3: direct.
        Variant::Kawpow => "uint seed0 = s0; uint seed1 = s1;",
        // 0.9.2 reads the seed big-endian.
        Variant::Vanilla => "uint seed0 = bswap32(s1); uint seed1 = bswap32(s0);",
    }
}

fn final_fill(variant: Variant) -> String {
    match variant {
        Variant::Kawpow => {
            let mut lines = vec![
                "for (int i = 0; i < 8; i++) st[i] = state2[i];".to_string(),
                "for (int i = 0; i < 8; i++) st[8 + i] = digest[i];".to_string(),
            ];
            for i in 0..9 {
                lines.push(format!("st[{}] = 0x{:08x}u;", 16 + i, RAVENCOIN_RNDC[i]));
            }
            lines.join("\n")
        }
        // vanilla: header(8) | seed(2, big-endian) | digest(8) | zero(7)
        Variant::Vanilla => "for (int i = 0; i < 8; i++) st[i] = g_header[i];\n\
             st[8] = bswap32(state2[1]); st[9] = bswap32(state2[0]);\n\
             for (int i = 0; i < 8; i++) st[10 + i] = digest[i];"
            .to_string(),
    }
}

/// Assemble the full GLSL source for the search shader of this period.
pub fn assemble_search_glsl(
    prog_seed: u64,
    dag_elements: u64,
    variant: Variant,
    max_outputs: u32,
    cnt_cache: usize,
    cnt_math: usize,
) -> String {
    let ops = progpow::build_program(prog_seed, cnt_cache, cnt_math);
    let loop_glsl = progpow::render_loop_glsl(prog_seed, &ops);
    let defines = format!(
        "#define PROGPOW_DAG_ELEMENTS {dag_elements}u\n\
         #define MAX_OUTPUTS {max_outputs}u\n"
    );
    SEARCH_GLSL_TMPL
        .replace("__DEFINES__", &defines)
        .replace("__PROGPOW_LOOP__", &loop_glsl)
        .replace("__SEED_DERIVE__", seed_derive(variant))
        .replace("__TAIL_INIT__", &tail_init(variant))
        .replace("__FINAL_FILL__", &final_fill(variant))
}

/// Assemble + compile the per-period search shader.
pub fn compile_search(
    prog_seed: u64,
    dag_elements: u64,
    variant: Variant,
    max_outputs: u32,
) -> Result<Vec<u8>> {
    let src = assemble_search_glsl(
        prog_seed,
        dag_elements,
        variant,
        max_outputs,
        PROGPOW_CNT_CACHE,
        PROGPOW_CNT_MATH,
    );
    compile_glsl(&src, "comp", true)
}

/// Assemble + compile the DAG-gen shader for a given epoch sizing.
pub fn compile_dag(light_items: u32, dag_items: u32, parents: u32) -> Result<Vec<u8>> {
    let defines = format!(
        "#define LIGHT_ITEMS {light_items}u\n\
         #define DAG_ITEMS {dag_items}u\n\
         #define DATASET_PARENTS {parents}u\n"
    );
    let src = ETHASH_DAG_GLSL.replace("__DEFINES__", &defines);
    compile_glsl(&src, "comp", true)
}

fn numbered(src: &str) -> String {
    src.lines()
        .enumerate()
        .map(|(i, l)| format!("{:4}| {l}", i + 1))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Phase 0 proof-of-life: assemble the real DAG shader with representative defines
/// and run it through `glslc`. Confirms the embed + substitution + compile + cache
/// path end to end. Returns the SPIR-V size in bytes.
pub fn selftest() -> Result<usize> {
    let spv = compile_dag(262_139, 16_777_213, 512)?;
    if spv.len() < 4 || spv[..4] != [0x03, 0x02, 0x23, 0x07] {
        bail!("compiled output is not SPIR-V (bad magic)");
    }
    Ok(spv.len())
}
