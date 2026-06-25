//! On-disk caching of the per-epoch light cache and full DAG (port of
//! `rdna3_kawpow/dagcache.py`).
//!
//! A fresh start spends ~40-80 s building the light cache + generating the multi-GB
//! DAG. Both are a pure function of `(variant, epoch, parents, seed)`, so they are
//! persisted and reloaded to make restarts within an epoch near-instant.
//!
//! Each file carries a small validated header (magic, version, epoch, parents,
//! payload length, 32-byte seed). Any mismatch or short/corrupt file is treated as
//! a miss and regenerated, so a stale cache can never feed the miner wrong data.
//! Writes go to a `.tmp` sibling and are atomically renamed; old epochs are pruned.

use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MAGIC_LIGHT: &[u8; 4] = b"RKLC";
const MAGIC_DAG: &[u8; 4] = b"RKDG";
const VERSION: u32 = 1; // bump to invalidate every existing cache file

// magic(4) | version(4) | epoch(4) | parents(4) | payload_len(8) | seed(32)
pub const HEADER_SIZE: usize = 4 + 4 + 4 + 4 + 8 + 32;

pub const KEEP_DAG: usize = 2;
pub const KEEP_LIGHT: usize = 4;

/// Persistent cache location (NOT the system tempdir, which may be wiped).
pub fn default_dir() -> PathBuf {
    if let Ok(base) = std::env::var("RDNA3_KAWPOW_CACHE_DIR") {
        return PathBuf::from(base);
    }
    let root = std::env::var("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(".cache")
        });
    root.join("rdna3_kawpow").join("dagcache")
}

fn seed32(seed: &[u8]) -> [u8; 32] {
    let mut s = [0u8; 32];
    let n = seed.len().min(32);
    s[..n].copy_from_slice(&seed[..n]);
    s
}

fn pack_header(magic: &[u8; 4], epoch: u32, parents: u32, payload_len: u64, seed: &[u8]) -> [u8; HEADER_SIZE] {
    let mut h = [0u8; HEADER_SIZE];
    h[0..4].copy_from_slice(magic);
    h[4..8].copy_from_slice(&VERSION.to_le_bytes());
    h[8..12].copy_from_slice(&epoch.to_le_bytes());
    h[12..16].copy_from_slice(&parents.to_le_bytes());
    h[16..24].copy_from_slice(&payload_len.to_le_bytes());
    h[24..56].copy_from_slice(&seed32(seed));
    h
}

/// Return the payload length iff the header matches expectations.
fn check_header(raw: &[u8], magic: &[u8; 4], epoch: u32, parents: u32, seed: &[u8]) -> Option<u64> {
    if raw.len() != HEADER_SIZE {
        return None;
    }
    if &raw[0..4] != magic {
        return None;
    }
    let ver = u32::from_le_bytes(raw[4..8].try_into().unwrap());
    let ep = u32::from_le_bytes(raw[8..12].try_into().unwrap());
    let par = u32::from_le_bytes(raw[12..16].try_into().unwrap());
    let payload_len = u64::from_le_bytes(raw[16..24].try_into().unwrap());
    let sd = &raw[24..56];
    if ver != VERSION || ep != epoch || par != parents || sd != seed32(seed) {
        return None;
    }
    Some(payload_len)
}

/// Write to a `.tmp` sibling and atomically rename on `commit()`.
pub struct Writer {
    final_path: PathBuf,
    tmp: PathBuf,
    f: Option<BufWriter<File>>,
}

impl Writer {
    fn create(final_path: PathBuf) -> io::Result<Self> {
        let tmp = final_path.with_extension("bin.tmp");
        let f = BufWriter::new(File::create(&tmp)?);
        Ok(Writer {
            final_path,
            tmp,
            f: Some(f),
        })
    }

    pub fn write_all(&mut self, data: &[u8]) -> io::Result<()> {
        self.f.as_mut().unwrap().write_all(data)
    }

    pub fn commit(mut self) -> io::Result<()> {
        let mut f = self.f.take().unwrap();
        f.flush()?;
        drop(f);
        fs::rename(&self.tmp, &self.final_path)
    }

    pub fn abort(mut self) {
        self.f.take();
        let _ = fs::remove_file(&self.tmp);
    }
}

/// Disk cache rooted at a directory; one instance per miner.
pub struct Cache {
    pub dir: PathBuf,
}

impl Cache {
    pub fn new(directory: Option<PathBuf>) -> io::Result<Cache> {
        let dir = directory.unwrap_or_else(default_dir);
        fs::create_dir_all(&dir)?;
        Ok(Cache { dir })
    }

    fn path(&self, prefix: &str, variant: &str, epoch: u32, parents: u32) -> PathBuf {
        self.dir
            .join(format!("{prefix}-{variant}-e{epoch}-p{parents}.bin"))
    }

    fn prune(&self, prefix: &str, variant: &str, keep: usize) {
        let needle = format!("{prefix}-{variant}-e");
        let mut stamped: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
        let Ok(rd) = fs::read_dir(&self.dir) else {
            return;
        };
        for entry in rd.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(&needle) && name.ends_with(".bin") {
                if let Ok(meta) = entry.metadata() {
                    if let Ok(mtime) = meta.modified() {
                        stamped.push((mtime, entry.path()));
                    }
                }
            }
        }
        stamped.sort_by(|a, b| b.0.cmp(&a.0)); // newest first
        for (_, old) in stamped.into_iter().skip(keep) {
            let _ = fs::remove_file(old);
        }
    }

    // --- light cache (small; read/written whole) ---
    pub fn load_light(&self, variant: &str, epoch: u32, parents: u32, seed: &[u8]) -> Option<Vec<u8>> {
        let mut f = File::open(self.path("light", variant, epoch, parents)).ok()?;
        let mut header = [0u8; HEADER_SIZE];
        f.read_exact(&mut header).ok()?;
        let payload_len = check_header(&header, MAGIC_LIGHT, epoch, parents, seed)? as usize;
        let mut data = vec![0u8; payload_len];
        f.read_exact(&mut data).ok()?;
        Some(data)
    }

    pub fn save_light(&self, variant: &str, epoch: u32, parents: u32, seed: &[u8], data: &[u8]) -> io::Result<()> {
        let mut w = Writer::create(self.path("light", variant, epoch, parents))?;
        let res = (|| -> io::Result<()> {
            w.write_all(&pack_header(MAGIC_LIGHT, epoch, parents, data.len() as u64, seed))?;
            w.write_all(data)
        })();
        match res {
            Ok(()) => w.commit()?,
            Err(e) => {
                w.abort();
                return Err(e);
            }
        }
        self.prune("light", variant, KEEP_LIGHT);
        Ok(())
    }

    // --- full DAG (multi-GB; streamed in chunks by the caller) ---
    pub fn has_dag(&self, variant: &str, epoch: u32, parents: u32, seed: &[u8], full_size: u64) -> bool {
        let Ok(mut f) = File::open(self.path("dag", variant, epoch, parents)) else {
            return false;
        };
        let mut header = [0u8; HEADER_SIZE];
        if f.read_exact(&mut header).is_err() {
            return false;
        }
        check_header(&header, MAGIC_DAG, epoch, parents, seed) == Some(full_size)
    }

    /// Open the DAG file positioned at the payload start (call `has_dag` first).
    pub fn open_dag_read(&self, variant: &str, epoch: u32, parents: u32) -> io::Result<File> {
        let mut f = File::open(self.path("dag", variant, epoch, parents))?;
        f.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
        Ok(f)
    }

    /// A `Writer` with the DAG header already written; caller streams the payload.
    pub fn open_dag_write(&self, variant: &str, epoch: u32, parents: u32, seed: &[u8], full_size: u64) -> io::Result<Writer> {
        let mut w = Writer::create(self.path("dag", variant, epoch, parents))?;
        if let Err(e) = w.write_all(&pack_header(MAGIC_DAG, epoch, parents, full_size, seed)) {
            w.abort();
            return Err(e);
        }
        Ok(w)
    }

    pub fn prune_dag(&self, variant: &str) {
        self.prune("dag", variant, KEEP_DAG);
    }
}

/// Convenience: does `path` look like a cache file we manage?
pub fn is_cache_file(path: &Path) -> bool {
    path.extension().map(|e| e == "bin").unwrap_or(false)
}
