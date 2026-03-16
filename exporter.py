#!/usr/bin/env python3
"""
Prometheus exporter for Ollama inference server with AMD GPU metrics.
Reads GPU stats from sysfs (no rocm-smi dependency) and Ollama API.
GPU cards are auto-discovered by scanning /sys/class/drm for cards that
expose gpu_busy_percent (i.e. AMD dGPUs).

Features:
- AMD GPU metrics via sysfs (utilisation, VRAM, temperature, power)
- Ollama model state via /api/ps (model loaded, VRAM, context length)
- Per-request metrics parsed from Ollama's GIN access logs via docker logs
- Active job elapsed time tracking (GPU utilisation threshold crossing)
- Hung-job detection (sustained high GPU utilisation duration)
"""

from __future__ import annotations

import glob
import logging
import os
import re
import subprocess
import threading
import time

import requests
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
SCRAPE_INTERVAL = int(os.environ.get("SCRAPE_INTERVAL", "10"))
OLLAMA_CONTAINER = os.environ.get("OLLAMA_CONTAINER", "ollama")
EXPORTER_PORT = int(os.environ.get("EXPORTER_PORT", "9101"))
HIGH_UTIL_THRESHOLD = 80   # percent – hung-job detector
ACTIVE_THRESHOLD = 20      # percent – active-job elapsed tracker


# ---------------------------------------------------------------------------
# GPU discovery
# ---------------------------------------------------------------------------

def discover_gpu_cards() -> list[tuple[int, str]]:
    """
    Return a sorted list of (logical_gpu_index, sysfs_device_path) tuples
    for all DRM cards that expose gpu_busy_percent (AMD dGPUs).
    """
    found = []
    for card_path in sorted(glob.glob("/sys/class/drm/card*/device")):
        busy = os.path.join(card_path, "gpu_busy_percent")
        if os.path.exists(busy):
            found.append(card_path)
    return [(idx, path) for idx, path in enumerate(found)]


# ---------------------------------------------------------------------------
# Prometheus metrics – GPU
# ---------------------------------------------------------------------------

gpu_util = Gauge("ollama_gpu_utilization_percent", "GPU utilisation 0-100", ["gpu"])
gpu_mem_used = Gauge("ollama_gpu_memory_used_bytes", "VRAM used in bytes", ["gpu"])
gpu_mem_total = Gauge("ollama_gpu_memory_total_bytes", "VRAM total in bytes", ["gpu"])
gpu_temp = Gauge("ollama_gpu_temperature_celsius", "GPU temperature in Celsius", ["gpu", "sensor"])
gpu_power = Gauge("ollama_gpu_power_watts", "GPU power draw in watts", ["gpu"])
gpu_high_util_duration = Gauge(
    "ollama_gpu_high_util_duration_seconds",
    f"Seconds GPU has been continuously above {HIGH_UTIL_THRESHOLD}%% utilisation",
    ["gpu"],
)
active_job_elapsed = Gauge(
    "ollama_active_job_elapsed_seconds",
    f"Seconds since GPU utilisation crossed {ACTIVE_THRESHOLD}%% (0 when idle)",
    ["gpu"],
)

# ---------------------------------------------------------------------------
# Prometheus metrics – Ollama model / API
# ---------------------------------------------------------------------------

model_loaded = Gauge("ollama_model_loaded", "1 if model is loaded in memory", ["model"])
model_vram = Gauge("ollama_model_vram_bytes", "VRAM occupied by loaded model", ["model"])
model_context = Gauge("ollama_model_context_length", "Context length of loaded model", ["model"])

api_up = Gauge("ollama_api_up", "1 if Ollama /api/ps responds successfully")
api_latency = Gauge("ollama_api_response_seconds", "Latency of Ollama /api/ps endpoint")

# ---------------------------------------------------------------------------
# Prometheus metrics – per-request (from GIN log parsing)
# ---------------------------------------------------------------------------

request_counter = Counter(
    "ollama_requests_total",
    "Total Ollama inference requests parsed from GIN logs",
    ["endpoint", "status", "method"],
)
request_duration = Histogram(
    "ollama_request_duration_seconds",
    "Ollama inference request duration in seconds (from GIN logs, >1s only)",
    ["endpoint"],
    buckets=[1, 5, 15, 30, 60, 120, 180, 300, 600],
)
last_completed_ts = Gauge(
    "ollama_last_request_completed_timestamp",
    "Unix epoch timestamp of last completed inference request",
)
last_duration = Gauge(
    "ollama_last_request_duration_seconds",
    "Duration in seconds of last completed inference request",
)
last_request_info = Gauge(
    "ollama_last_request_info",
    "Labels carry info about the last completed request; value is always 1.0",
    ["client", "endpoint", "status", "method"],
)

# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------

_high_util_since: dict[int, float | None] = {}
_gpu_active_since: dict[int, float | None] = {}
_prev_models: set[str] = set()

_info_lock = threading.Lock()
_last_info_labels: dict | None = None


# ---------------------------------------------------------------------------
# GIN log parsing
# ---------------------------------------------------------------------------

GIN_PATTERN = re.compile(
    r'\[GIN\].*\|\s*(\d+)\s*\|\s*([\w\d.µns]+)\s*\|\s*([\d.]+(?:\.\d+)?)\s*\|\s*(\w+)\s+"([^"]+)"'
)


def parse_go_duration(s: str) -> float:
    """Convert a Go duration string (e.g. '2m47s', '35.838µs') to seconds."""
    s = s.strip()
    total = 0.0
    for val, unit in re.findall(r'([\d.]+)(h|m(?!s)|s|ms|µs|ns)', s):
        val = float(val)
        if unit == 'h':
            total += val * 3600
        elif unit == 'm':
            total += val * 60
        elif unit == 's':
            total += val
        elif unit == 'ms':
            total += val / 1e3
        elif unit == 'µs':
            total += val / 1e6
        elif unit == 'ns':
            total += val / 1e9
    return total


def _handle_gin_line(line: str) -> None:
    """Parse one GIN log line and update metrics if it is a slow request (>1s)."""
    m = GIN_PATTERN.search(line)
    if not m:
        return

    status_code, duration_str, client_ip, method, path = m.groups()
    duration = parse_go_duration(duration_str)

    # Filter out fast /api/ps polling and other sub-second requests
    if duration <= 1.0:
        return

    endpoint = path.split('?')[0]  # strip query string

    request_counter.labels(endpoint=endpoint, status=status_code, method=method).inc()
    request_duration.labels(endpoint=endpoint).observe(duration)

    now = time.time()
    last_completed_ts.set(now)
    last_duration.set(duration)

    with _info_lock:
        global _last_info_labels
        if _last_info_labels is not None:
            try:
                last_request_info.labels(**_last_info_labels).set(0)
            except Exception:
                pass
        new_labels = {
            "client": client_ip,
            "endpoint": endpoint,
            "status": status_code,
            "method": method,
        }
        last_request_info.labels(**new_labels).set(1.0)
        _last_info_labels = new_labels

    log.info(
        "GIN request recorded: %s %s -> %s  duration=%.1fs",
        method, endpoint, status_code, duration,
    )


def tail_docker_logs() -> None:
    """Daemon thread: follow docker logs for the Ollama container and parse GIN lines."""
    log.info("Starting docker log tail thread for container '%s'", OLLAMA_CONTAINER)
    while True:
        try:
            proc = subprocess.Popen(
                ['docker', 'logs', '--follow', '--tail', '100', OLLAMA_CONTAINER],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            log.info("docker logs process started (pid=%d)", proc.pid)
            for raw_line in proc.stdout:
                try:
                    line = raw_line.decode('utf-8', errors='replace').rstrip()
                    _handle_gin_line(line)
                except Exception as exc:
                    log.debug("Error processing log line: %s", exc)
            proc.wait()
            log.warning(
                "docker logs process exited (rc=%d), restarting in 5s", proc.returncode
            )
        except Exception as exc:
            log.error("tail_docker_logs error: %s", exc)
        time.sleep(5)


# ---------------------------------------------------------------------------
# sysfs helpers
# ---------------------------------------------------------------------------

def _read_sysfs(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _read_int(path: str) -> int | None:
    val = _read_sysfs(path)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _hwmon_dir(card_device_path: str) -> str | None:
    dirs = sorted(glob.glob(os.path.join(card_device_path, "hwmon", "hwmon*")))
    return dirs[0] if dirs else None


# ---------------------------------------------------------------------------
# GPU metric collection
# ---------------------------------------------------------------------------

def collect_gpu_metrics(gpu_cards: list[tuple[int, str]]) -> None:
    now_mono = time.monotonic()
    now_wall = time.time()

    for gpu_idx, dev_path in gpu_cards:
        _high_util_since.setdefault(gpu_idx, None)
        _gpu_active_since.setdefault(gpu_idx, None)

        # Utilisation
        util_val = _read_int(os.path.join(dev_path, "gpu_busy_percent"))
        if util_val is not None:
            gpu_util.labels(gpu=str(gpu_idx)).set(util_val)

            # Hung-job tracker (>80%)
            if util_val > HIGH_UTIL_THRESHOLD:
                if _high_util_since[gpu_idx] is None:
                    _high_util_since[gpu_idx] = now_mono
                hung_duration = now_mono - _high_util_since[gpu_idx]
            else:
                _high_util_since[gpu_idx] = None
                hung_duration = 0.0
            gpu_high_util_duration.labels(gpu=str(gpu_idx)).set(hung_duration)

            # Active-job elapsed (>=20%)
            if util_val >= ACTIVE_THRESHOLD:
                if _gpu_active_since[gpu_idx] is None:
                    _gpu_active_since[gpu_idx] = now_wall
                elapsed = now_wall - _gpu_active_since[gpu_idx]
            else:
                _gpu_active_since[gpu_idx] = None
                elapsed = 0.0
            active_job_elapsed.labels(gpu=str(gpu_idx)).set(elapsed)

        # VRAM
        vram_used = _read_int(os.path.join(dev_path, "mem_info_vram_used"))
        vram_total = _read_int(os.path.join(dev_path, "mem_info_vram_total"))
        if vram_used is not None:
            gpu_mem_used.labels(gpu=str(gpu_idx)).set(vram_used)
        if vram_total is not None:
            gpu_mem_total.labels(gpu=str(gpu_idx)).set(vram_total)

        # Temperature and power via hwmon
        hwmon = _hwmon_dir(dev_path)
        if hwmon:
            for sensor_file, sensor_name in [
                ("temp1_input", "edge"),
                ("temp2_input", "junction"),
                ("temp3_input", "memory"),
            ]:
                milli = _read_int(os.path.join(hwmon, sensor_file))
                if milli is not None:
                    gpu_temp.labels(gpu=str(gpu_idx), sensor=sensor_name).set(milli / 1000.0)

            uw = _read_int(os.path.join(hwmon, "power1_average"))
            if uw is not None:
                gpu_power.labels(gpu=str(gpu_idx)).set(uw / 1_000_000.0)


# ---------------------------------------------------------------------------
# Ollama API metric collection
# ---------------------------------------------------------------------------

def collect_ollama_metrics() -> None:
    global _prev_models

    t0 = time.monotonic()
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
        latency = time.monotonic() - t0
        resp.raise_for_status()
        data = resp.json()
        api_up.set(1)
        api_latency.set(latency)
    except Exception as exc:
        api_up.set(0)
        api_latency.set(time.monotonic() - t0)
        log.warning("Ollama /api/ps failed: %s", exc)
        for m in _prev_models:
            model_loaded.labels(model=m).set(0)
        _prev_models = set()
        return

    current_models: set[str] = set()
    for entry in data.get("models", []):
        name = entry.get("name", "unknown")
        current_models.add(name)
        model_loaded.labels(model=name).set(1)

        size_vram = entry.get("size_vram", 0)
        ctx = entry.get("context_length") or entry.get("details", {}).get("context_length", 0)
        if size_vram:
            model_vram.labels(model=name).set(size_vram)
        if ctx:
            model_context.labels(model=name).set(ctx)

    for stale in _prev_models - current_models:
        model_loaded.labels(model=stale).set(0)

    _prev_models = current_models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    gpu_cards = discover_gpu_cards()
    log.info(
        "Discovered %d GPU(s): %s",
        len(gpu_cards),
        [(idx, path) for idx, path in gpu_cards],
    )
    log.info("Starting Ollama Prometheus exporter on port %d", EXPORTER_PORT)
    log.info("Ollama URL: %s  |  Scrape interval: %ds", OLLAMA_URL, SCRAPE_INTERVAL)

    start_http_server(EXPORTER_PORT)

    log_thread = threading.Thread(target=tail_docker_logs, daemon=True)
    log_thread.start()
    log.info("Docker log tail thread started")

    while True:
        try:
            collect_gpu_metrics(gpu_cards)
        except Exception as exc:
            log.error("GPU metric collection error: %s", exc)
        try:
            collect_ollama_metrics()
        except Exception as exc:
            log.error("Ollama metric collection error: %s", exc)
        time.sleep(SCRAPE_INTERVAL)


if __name__ == "__main__":
    main()
