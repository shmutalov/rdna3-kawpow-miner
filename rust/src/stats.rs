//! Built-in stats endpoint for HiveOS integration.
//!
//! Exposes a tiny HTTP/JSON API (any GET returns a snapshot) that the HiveOS
//! `h-stats.sh` curls and maps into the dashboard format with `jq`. The miner only
//! reports what it alone knows -- hashrate + share counters + uptime + device;
//! temps/fans are filled by HiveOS itself.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use serde_json::json;

struct Inner {
    hashrate_bits: AtomicU64, // f64::to_bits (hashes/s)
    accepted: AtomicU64,
    rejected: AtomicU64,
    invalid: AtomicU64,
    start: Instant,
    algo: String,
    device: Mutex<String>,
}

#[derive(Clone)]
pub struct Stats {
    inner: Arc<Inner>,
}

impl Stats {
    pub fn new(algo: &str) -> Self {
        Stats {
            inner: Arc::new(Inner {
                hashrate_bits: AtomicU64::new(0),
                accepted: AtomicU64::new(0),
                rejected: AtomicU64::new(0),
                invalid: AtomicU64::new(0),
                start: Instant::now(),
                algo: algo.to_string(),
                device: Mutex::new(String::new()),
            }),
        }
    }

    pub fn set_device(&self, name: &str) {
        *self.inner.device.lock().unwrap() = name.to_string();
    }
    pub fn set_hashrate(&self, hs: f64) {
        self.inner.hashrate_bits.store(hs.to_bits(), Ordering::Relaxed);
    }
    pub fn set_shares(&self, accepted: u64, rejected: u64) {
        self.inner.accepted.store(accepted, Ordering::Relaxed);
        self.inner.rejected.store(rejected, Ordering::Relaxed);
    }
    pub fn add_invalid(&self) {
        self.inner.invalid.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot_json(&self) -> String {
        let hs = f64::from_bits(self.inner.hashrate_bits.load(Ordering::Relaxed));
        json!({
            "hashrate": hs,
            "hashrate_mhs": hs / 1e6,
            "accepted": self.inner.accepted.load(Ordering::Relaxed),
            "rejected": self.inner.rejected.load(Ordering::Relaxed),
            "invalid": self.inner.invalid.load(Ordering::Relaxed),
            "uptime": self.inner.start.elapsed().as_secs(),
            "algo": self.inner.algo,
            "device": *self.inner.device.lock().unwrap(),
        })
        .to_string()
    }

    /// Spawn a background HTTP server. Any GET returns the JSON snapshot. Returns
    /// the actually-bound address (useful when binding to port 0).
    pub fn serve(&self, addr: &str) -> std::io::Result<std::net::SocketAddr> {
        let listener = TcpListener::bind(addr)?;
        let local = listener.local_addr()?;
        let me = self.clone();
        thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                me.handle_conn(stream);
            }
        });
        Ok(local)
    }

    fn handle_conn(&self, mut s: TcpStream) {
        // Drain the request line/headers (we ignore them; any request -> snapshot).
        let mut buf = [0u8; 1024];
        let _ = s.read(&mut buf);
        let body = self.snapshot_json();
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = s.write_all(resp.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_has_expected_fields() {
        let s = Stats::new("kawpow");
        s.set_device("Test GPU");
        s.set_hashrate(49_470_000.0);
        s.set_shares(7, 1);
        s.add_invalid();
        let v: serde_json::Value = serde_json::from_str(&s.snapshot_json()).unwrap();
        assert_eq!(v["algo"], "kawpow");
        assert_eq!(v["device"], "Test GPU");
        assert_eq!(v["accepted"], 7);
        assert_eq!(v["rejected"], 1);
        assert_eq!(v["invalid"], 1);
        assert!((v["hashrate_mhs"].as_f64().unwrap() - 49.47).abs() < 0.01);
    }

    #[test]
    fn http_endpoint_serves_json() {
        let s = Stats::new("kawpow");
        s.set_hashrate(1234.0);
        let addr = s.serve("127.0.0.1:0").unwrap(); // ephemeral port
        let mut conn = TcpStream::connect(addr).unwrap();
        conn.write_all(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
        let mut resp = String::new();
        conn.read_to_string(&mut resp).unwrap();
        assert!(resp.starts_with("HTTP/1.1 200 OK"), "bad status line: {resp}");
        let body = resp.split("\r\n\r\n").nth(1).unwrap();
        let v: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(v["hashrate"], 1234.0);
    }
}
