"""Assemble GLSL compute shaders from the template + generated program and
compile them to SPIR-V with glslc (shipped in the Vulkan SDK).

The search shader is regenerated whenever the ProgPoW period changes (every
PROGPOW_PERIOD blocks); compiled SPIR-V is cached on disk keyed by content.
"""

import hashlib
import os
import shutil
import subprocess
import tempfile

from . import progpow
from .constants import RAVENCOIN_RNDC, PROGPOW_CNT_CACHE, PROGPOW_CNT_MATH

_HERE = os.path.dirname(os.path.abspath(__file__))
_SHADER_DIR = os.path.join(_HERE, "shaders")
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "rdna3_kawpow_spv")

KAWPOW = "kawpow"
VANILLA = "vanilla"


def find_glslc():
    exe = shutil.which("glslc")
    if exe:
        return exe
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        cand = os.path.join(sdk, "Bin", "glslc.exe" if os.name == "nt" else "glslc")
        if os.path.exists(cand):
            return cand
    raise RuntimeError("glslc not found (install the Vulkan SDK or set VULKAN_SDK)")


def _tail_init(variant):
    if variant != KAWPOW:
        return ""  # vanilla: tail already zero
    return "\n".join(f"st[{10 + i}] = 0x{RAVENCOIN_RNDC[i]:08x}u;"
                     for i in range(15))


def _seed_derive(variant):
    if variant == KAWPOW:
        return "uint seed0 = s0; uint seed1 = s1;"      # direct (kawpowminer 0.9.3)
    return "uint seed0 = bswap32(s1); uint seed1 = bswap32(s0);"  # 0.9.2 big-endian


def _final_fill(variant):
    if variant == KAWPOW:
        lines = ["for (int i = 0; i < 8; i++) st[i] = state2[i];",
                 "for (int i = 0; i < 8; i++) st[8 + i] = digest[i];"]
        lines += [f"st[{16 + i}] = 0x{RAVENCOIN_RNDC[i]:08x}u;" for i in range(9)]
        return "\n".join(lines)
    # vanilla: header(8) | seed(2, big-endian) | digest(8) | zero(7)
    return ("for (int i = 0; i < 8; i++) st[i] = g_header[i];\n"
            "st[8] = bswap32(state2[1]); st[9] = bswap32(state2[0]);\n"
            "for (int i = 0; i < 8; i++) st[10 + i] = digest[i];")


def assemble_search_glsl(prog_seed, dag_elements, variant=KAWPOW,
                         max_outputs=4, cnt_cache=PROGPOW_CNT_CACHE,
                         cnt_math=PROGPOW_CNT_MATH):
    """Return the full GLSL source for the search shader of this period."""
    with open(os.path.join(_SHADER_DIR, "progpow_search.comp.tmpl")) as f:
        tmpl = f.read()
    loop = progpow.render_loop(prog_seed, progpow.GLSL, None, cnt_cache, cnt_math)
    defines = (f"#define PROGPOW_DAG_ELEMENTS {dag_elements}u\n"
               f"#define MAX_OUTPUTS {max_outputs}u\n")
    src = (tmpl.replace("__DEFINES__", defines)
               .replace("__PROGPOW_LOOP__", loop)
               .replace("__SEED_DERIVE__", _seed_derive(variant))
               .replace("__TAIL_INIT__", _tail_init(variant))
               .replace("__FINAL_FILL__", _final_fill(variant)))
    return src


def compile_glsl(src, stage="comp", optimize=True):
    """Compile GLSL `src` to SPIR-V bytes via glslc, with on-disk caching."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    key = hashlib.sha256((stage + str(optimize) + src).encode()).hexdigest()[:24]
    spv_path = os.path.join(_CACHE_DIR, key + ".spv")
    if os.path.exists(spv_path):
        with open(spv_path, "rb") as f:
            return f.read()

    src_path = os.path.join(_CACHE_DIR, key + "." + stage)
    with open(src_path, "w") as f:
        f.write(src)
    cmd = [find_glslc(), "--target-env=vulkan1.3", f"-fshader-stage={stage}"]
    if optimize:
        cmd.append("-O")
    cmd += [src_path, "-o", spv_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"glslc failed:\n{proc.stderr}\n--- source ---\n" +
                           _numbered(src))
    with open(spv_path, "rb") as f:
        return f.read()


def compile_search(prog_seed, dag_elements, variant=KAWPOW, max_outputs=4,
                   cnt_cache=PROGPOW_CNT_CACHE, cnt_math=PROGPOW_CNT_MATH):
    src = assemble_search_glsl(prog_seed, dag_elements, variant, max_outputs,
                               cnt_cache, cnt_math)
    return compile_glsl(src)


def compile_dag(light_items, dag_items, parents=512):
    with open(os.path.join(_SHADER_DIR, "ethash_dag.comp")) as f:
        tmpl = f.read()
    defines = (f"#define LIGHT_ITEMS {light_items}u\n"
               f"#define DAG_ITEMS {dag_items}u\n"
               f"#define DATASET_PARENTS {parents}u\n")
    return compile_glsl(tmpl.replace("__DEFINES__", defines))


def _numbered(src):
    return "\n".join(f"{i+1:4d}| {ln}" for i, ln in enumerate(src.splitlines()))
