"""On-disk caching of the per-epoch light cache and full DAG.

A fresh start spends ~40-80 s building the epoch's light cache (pure-Python
``mkcache``) and then generating the multi-GB DAG on the GPU. Both are a pure
function of ``(epoch, seed, parents)``, so once produced they can be persisted
and reloaded on the next start, making restarts within an epoch effectively
instant.

Two artifacts are cached side by side, keyed by ``(variant, epoch, parents)``:

* the **light cache** (tens of MiB) -- needed both to regenerate the DAG and for
  host-side solution re-checking, and
* the **full DAG** (multi-GB) -- so the GPU generation pass can be skipped too.

Each file carries a small validated header (magic, format version, epoch,
parents, payload length and the 32-byte seed). Any mismatch or short/corrupt
file is silently treated as a miss and regenerated, so a stale cache can never
feed the miner the wrong data. Writes go to a ``.tmp`` sibling and are atomically
renamed, and old epochs are pruned so the directory does not grow without bound.

All disk access here is best-effort: callers wrap save/load so a full disk or a
permission error degrades to "regenerate", never a crash.
"""

import glob
import os
import struct

MAGIC_LIGHT = b"RKLC"        # rdna3-kawpow light cache
MAGIC_DAG = b"RKDG"          # rdna3-kawpow full DAG
VERSION = 1                  # bump to invalidate every existing cache file

# magic, format version, epoch, parents, payload length, 32-byte seed
_HEADER_FMT = "<4sIIIQ32s"
HEADER_SIZE = struct.calcsize(_HEADER_FMT)

# Retention: epochs advance rarely (every 7500 blocks), so a couple of DAGs and a
# few light caches is plenty to cover an epoch flip without unbounded growth.
KEEP_DAG = 2
KEEP_LIGHT = 4


def default_dir():
    """Persistent cache location (NOT the system tempdir, which may be wiped)."""
    base = os.environ.get("RDNA3_KAWPOW_CACHE_DIR")
    if base:
        return base
    root = os.environ.get("LOCALAPPDATA") or os.path.join(
        os.path.expanduser("~"), ".cache")
    return os.path.join(root, "rdna3_kawpow", "dagcache")


def _seed32(seed):
    return (bytes(seed or b"") + b"\x00" * 32)[:32]


def _pack_header(magic, epoch, parents, payload_len, seed):
    return struct.pack(_HEADER_FMT, magic, VERSION, epoch, parents,
                       payload_len, _seed32(seed))


def _check_header(raw, magic, epoch, parents, seed):
    """Return the payload length if the header matches expectations, else None."""
    if len(raw) != HEADER_SIZE:
        return None
    m, ver, ep, par, payload_len, sd = struct.unpack(_HEADER_FMT, raw)
    if (m != magic or ver != VERSION or ep != epoch or par != parents
            or sd != _seed32(seed)):
        return None
    return payload_len


class _Writer:
    """Write to a ``.tmp`` sibling and atomically rename on commit()."""

    def __init__(self, final_path):
        self.final = final_path
        self.tmp = final_path + ".tmp"
        self.f = open(self.tmp, "wb")

    def write(self, data):
        self.f.write(data)

    def commit(self):
        self.f.close()
        os.replace(self.tmp, self.final)

    def abort(self):
        try:
            self.f.close()
        finally:
            try:
                os.remove(self.tmp)
            except OSError:
                pass


class Cache:
    """Disk cache rooted at a directory; one instance per miner."""

    def __init__(self, directory=None):
        self.dir = directory or default_dir()
        os.makedirs(self.dir, exist_ok=True)

    # --- paths ---
    def _path(self, prefix, variant, epoch, parents):
        return os.path.join(self.dir, f"{prefix}-{variant}-e{epoch}-p{parents}.bin")

    def _prune(self, prefix, variant, keep):
        # Best-effort: a file vanishing mid-scan must never mask a successful save.
        try:
            pattern = os.path.join(self.dir, f"{prefix}-{variant}-e*.bin")
            stamped = []
            for f in glob.glob(pattern):
                try:
                    stamped.append((os.path.getmtime(f), f))
                except OSError:
                    pass
            stamped.sort(reverse=True)
            for _, old in stamped[keep:]:
                try:
                    os.remove(old)
                except OSError:
                    pass
        except OSError:
            pass

    # --- light cache (small; read/written whole) ---
    def load_light(self, variant, epoch, parents, seed):
        """Return the cached light-cache bytes, or None on miss/mismatch."""
        try:
            with open(self._path("light", variant, epoch, parents), "rb") as f:
                payload_len = _check_header(f.read(HEADER_SIZE), MAGIC_LIGHT,
                                            epoch, parents, seed)
                if payload_len is None:
                    return None
                data = f.read(payload_len)
                return data if len(data) == payload_len else None
        except OSError:
            return None

    def save_light(self, variant, epoch, parents, seed, data):
        w = _Writer(self._path("light", variant, epoch, parents))
        try:
            w.write(_pack_header(MAGIC_LIGHT, epoch, parents, len(data), seed))
            w.write(data)
            w.commit()
        except BaseException:
            w.abort()
            raise
        self._prune("light", variant, KEEP_LIGHT)

    # --- full DAG (multi-GB; streamed in chunks by the caller) ---
    def has_dag(self, variant, epoch, parents, seed, full_size):
        """True iff a DAG file exists whose header matches and payload == full_size."""
        try:
            with open(self._path("dag", variant, epoch, parents), "rb") as f:
                payload_len = _check_header(f.read(HEADER_SIZE), MAGIC_DAG,
                                            epoch, parents, seed)
            return payload_len == full_size
        except OSError:
            return False

    def open_dag_read(self, variant, epoch, parents):
        """Open the DAG file positioned at the payload start (call has_dag first)."""
        f = open(self._path("dag", variant, epoch, parents), "rb")
        f.seek(HEADER_SIZE)
        return f

    def open_dag_write(self, variant, epoch, parents, seed, full_size):
        """Return a _Writer with the DAG header already written; caller streams payload."""
        w = _Writer(self._path("dag", variant, epoch, parents))
        try:
            w.write(_pack_header(MAGIC_DAG, epoch, parents, full_size, seed))
        except BaseException:
            w.abort()
            raise
        return w

    def prune_dag(self, variant):
        self._prune("dag", variant, KEEP_DAG)
