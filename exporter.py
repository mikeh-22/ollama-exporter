#!/usr/bin/env python3
"""
Prometheus exporter for Ollama inference server.

Supports two GPU backends, auto-detected at startup:
  - AMD/sysfs   : reads /sys/class/drm/card*/device — no rocm-smi required
  - NVIDIA/pynvml: reads NVML via nvidia-ml-py — requires NVIDIA container runtime

Metrics common to both backends:
  ollama_gpu_utilization_percent, ollama_gpu_memory_{used,total}_bytes,
  ollama_gpu_temperature_celsius, ollama_gpu_power_watts,
  ollama_gpu_high_util_duration_seconds, ollama_active_job_elapsed_seconds

NVIDIA-only additional metrics:
  ollama_gpu_memory_bandwidth_utilization_percent, ollama_gpu_clock_mhz,
  ollama_gpu_fan_speed_percent, ollama_gpu_compute_process_count

Per-request metrics are parsed from Ollama's GIN access logs via docker logs.
"""

from __future__ import annotations

import abc
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
HIGH_UTIL_THRESHOLD = 80
ACTIVE_THRESHOLD = 20


# ---------------------------------------------------------------------------
# Prometheus metrics — common to all backends
# ---------------------------------------------------------------------------

gpu_util = Gauge("ollama_gpu_utilization_percent", "GPU compute utilisation 0-100", ["gpu"])
gpu_mem_used = Gauge("ollama_gpu_memory_used_bytes", "VRAM used in bytes", ["gpu"])
gpu_mem_total = Gauge("ollama_gpu_memory_total_bytes", "VRAM total in bytes", ["gpu"])
gpu_temp = Gauge(
    "ollama_gpu_temperature_celsius", "GPU temperature in Celsius", ["gpu", "sensor"]
)
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
backend_info = Gauge(
    "ollama_gpu_backend_info",
    "Always 1; labels identify the active GPU backend and device count",
    ["backend", "gpu_count"],
)

# ---------------------------------------------------------------------------
# Prometheus metrics — NVIDIA-only extras
# ---------------------------------------------------------------------------

gpu_mem_bandwidth_util = Gauge(
    "ollama_gpu_memory_bandwidth_utilization_percent",
    "GPU memory bandwidth utilisation 0-100 (NVIDIA only)",
    ["gpu"],
)
gpu_clock = Gauge(
    "ollama_gpu_clock_mhz",
    "Current GPU clock speed in MHz (NVIDIA only)",
    ["gpu", "type"],
)
gpu_fan = Gauge(
    "ollama_gpu_fan_speed_percent",
    "GPU fan speed 0-100 (NVIDIA only)",
    ["gpu"],
)
gpu_compute_processes = Gauge(
    "ollama_gpu_compute_process_count",
    "Number of active compute processes on GPU (NVIDIA only)",
    ["gpu"],
)

# ---------------------------------------------------------------------------
# Prometheus metrics — Ollama model / API
# ---------------------------------------------------------------------------

model_loaded = Gauge("ollama_model_loaded", "1 if model is loaded in memory", ["model"])
model_vram = Gauge("ollama_model_vram_bytes", "VRAM occupied by loaded model", ["model"])
model_context = Gauge("ollama_model_context_length", "Context length of loaded model", ["model"])
api_up = Gauge("ollama_api_up", "1 if Ollama /api/ps responds successfully")
api_latency = Gauge("ollama_api_response_seconds", "Latency of Ollama /api/ps endpoint")

# ---------------------------------------------------------------------------
# Prometheus metrics — per-request (GIN log parsing)
# ---------------------------------------------------------------------------

request_counter = Counter(
    "ollama_requests_total",
    "Total Ollama inference requests parsed from GIN logs",
    ["endpoint", "status", "method"],
)
request_duration = Histogram(
    "ollama_request_duration_seconds",
    "Ollama inference request duration in seconds (>1s requests only)",
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
    "Always 1.0; labels carry metadata about the last completed request",
    ["client", "endpoint", "status", "method"],
)

# ---------------------------------------------------------------------------
# Prometheus metrics — context window (requires OLLAMA_DEBUG=1 on Ollama)
# ---------------------------------------------------------------------------

ctx_prompt_tokens = Gauge(
    "ollama_last_request_prompt_tokens",
    "Prompt token count at start of last inference request (requires OLLAMA_DEBUG=1)",
)
ctx_kv_reuse_tokens = Gauge(
    "ollama_last_request_kv_cache_reuse_tokens",
    "KV cache tokens reused from prior turn in last request (requires OLLAMA_DEBUG=1)",
)
ctx_eval_tokens = Gauge(
    "ollama_last_request_eval_tokens",
    "Generated token count of last inference request (requires OLLAMA_DEBUG=1)",
)
ctx_fill_ratio = Gauge(
    "ollama_last_request_context_fill_ratio",
    "Context window fill ratio at last request — prompt_tokens / context_length (requires OLLAMA_DEBUG=1)",
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_high_util_since: dict[int, float | None] = {}
_gpu_active_since: dict[int, float | None] = {}
_prev_models: set[str] = set()
_info_lock = threading.Lock()
_last_info_labels: dict | None = None
_current_context_length: int = 0
_context_length_lock = threading.Lock()


# ---------------------------------------------------------------------------
# GPU state tracking helpers (shared between backends)
# ---------------------------------------------------------------------------

def _update_utilisation_tracking(
    gpu_idx: int, util: float, now_mono: float, now_wall: float
) -> None:
    """Update hung-job and active-job elapsed Gauges for one GPU."""
    _high_util_since.setdefault(gpu_idx, None)
    _gpu_active_since.setdefault(gpu_idx, None)

    if util > HIGH_UTIL_THRESHOLD:
        if _high_util_since[gpu_idx] is None:
            _high_util_since[gpu_idx] = now_mono
        hung = now_mono - _high_util_since[gpu_idx]
    else:
        _high_util_since[gpu_idx] = None
        hung = 0.0
    gpu_high_util_duration.labels(gpu=str(gpu_idx)).set(hung)

    if util >= ACTIVE_THRESHOLD:
        if _gpu_active_since[gpu_idx] is None:
            _gpu_active_since[gpu_idx] = now_wall
        elapsed = now_wall - _gpu_active_since[gpu_idx]
    else:
        _gpu_active_since[gpu_idx] = None
        elapsed = 0.0
    active_job_elapsed.labels(gpu=str(gpu_idx)).set(elapsed)


# ---------------------------------------------------------------------------
# GPU backend abstraction
# ---------------------------------------------------------------------------

class GPUBackend(abc.ABC):
    name: str = "unknown"

    @abc.abstractmethod
    def collect(self) -> None:
        """Read GPU hardware and update Prometheus metrics."""

    @abc.abstractmethod
    def gpu_count(self) -> int:
        """Return number of GPUs managed by this backend."""


# ---------------------------------------------------------------------------
# AMD sysfs backend
# ---------------------------------------------------------------------------

class AMDBackend(GPUBackend):
    name = "AMD/sysfs"

    def __init__(self, cards: list[tuple[int, str]]) -> None:
        self.cards = cards

    @classmethod
    def detect(cls) -> AMDBackend | None:
        """Return an AMDBackend if any AMD DRM cards are found via sysfs."""
        found = [
            path
            for path in sorted(glob.glob("/sys/class/drm/card*/device"))
            if os.path.exists(os.path.join(path, "gpu_busy_percent"))
        ]
        if not found:
            return None
        log.info("AMD/sysfs backend: discovered %d card(s)", len(found))
        return cls(list(enumerate(found)))

    def gpu_count(self) -> int:
        return len(self.cards)

    def collect(self) -> None:
        now_mono = time.monotonic()
        now_wall = time.time()

        for gpu_idx, dev_path in self.cards:
            util_val = _read_int(os.path.join(dev_path, "gpu_busy_percent"))
            if util_val is not None:
                gpu_util.labels(gpu=str(gpu_idx)).set(util_val)
                _update_utilisation_tracking(gpu_idx, util_val, now_mono, now_wall)

            vram_used = _read_int(os.path.join(dev_path, "mem_info_vram_used"))
            vram_total = _read_int(os.path.join(dev_path, "mem_info_vram_total"))
            if vram_used is not None:
                gpu_mem_used.labels(gpu=str(gpu_idx)).set(vram_used)
            if vram_total is not None:
                gpu_mem_total.labels(gpu=str(gpu_idx)).set(vram_total)

            hwmon = _hwmon_dir(dev_path)
            if hwmon:
                for sensor_file, sensor_name in [
                    ("temp1_input", "edge"),
                    ("temp2_input", "junction"),
                    ("temp3_input", "memory"),
                ]:
                    milli = _read_int(os.path.join(hwmon, sensor_file))
                    if milli is not None:
                        gpu_temp.labels(gpu=str(gpu_idx), sensor=sensor_name).set(
                            milli / 1000.0
                        )

                uw = _read_int(os.path.join(hwmon, "power1_average"))
                if uw is not None:
                    gpu_power.labels(gpu=str(gpu_idx)).set(uw / 1_000_000.0)


# ---------------------------------------------------------------------------
# NVIDIA pynvml backend
# ---------------------------------------------------------------------------

class NvidiaBackend(GPUBackend):
    name = "NVIDIA/pynvml"

    def __init__(self, pynvml, handles: list) -> None:
        self._pynvml = pynvml
        self.handles = handles

    @classmethod
    def detect(cls) -> NvidiaBackend | None:
        """Return a NvidiaBackend if NVML initialises successfully."""
        try:
            import pynvml  # noqa: PLC0415
        except ImportError:
            log.debug("pynvml not installed; NVIDIA backend unavailable")
            return None
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            names = []
            for h in handles:
                try:
                    names.append(pynvml.nvmlDeviceGetName(h))
                except Exception:
                    names.append("unknown")
            log.info("NVIDIA/pynvml backend: %d device(s): %s", count, names)
            return cls(pynvml, handles)
        except Exception as exc:
            log.debug("NVML init failed: %s", exc)
            return None

    def gpu_count(self) -> int:
        return len(self.handles)

    def collect(self) -> None:
        pynvml = self._pynvml
        now_mono = time.monotonic()
        now_wall = time.time()

        for gpu_idx, handle in enumerate(self.handles):
            # Compute + memory bandwidth utilisation
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util.labels(gpu=str(gpu_idx)).set(util.gpu)
                gpu_mem_bandwidth_util.labels(gpu=str(gpu_idx)).set(util.memory)
                _update_utilisation_tracking(gpu_idx, util.gpu, now_mono, now_wall)
            except Exception as exc:
                log.debug("GPU %d utilisation error: %s", gpu_idx, exc)

            # VRAM
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem_used.labels(gpu=str(gpu_idx)).set(mem.used)
                gpu_mem_total.labels(gpu=str(gpu_idx)).set(mem.total)
            except Exception as exc:
                log.debug("GPU %d memory info error: %s", gpu_idx, exc)

            # Temperature (NVML exposes GPU core; no separate junction/memory)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                gpu_temp.labels(gpu=str(gpu_idx), sensor="core").set(temp)
            except Exception as exc:
                log.debug("GPU %d temperature error: %s", gpu_idx, exc)

            # Power (NVML returns milliwatts)
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                gpu_power.labels(gpu=str(gpu_idx)).set(mw / 1000.0)
            except Exception as exc:
                log.debug("GPU %d power error: %s", gpu_idx, exc)

            # Clock speeds (NVIDIA-specific)
            try:
                for clock_type, clock_name in [
                    (pynvml.NVML_CLOCK_SM, "sm"),
                    (pynvml.NVML_CLOCK_MEM, "memory"),
                    (pynvml.NVML_CLOCK_GRAPHICS, "graphics"),
                ]:
                    mhz = pynvml.nvmlDeviceGetClockInfo(handle, clock_type)
                    gpu_clock.labels(gpu=str(gpu_idx), type=clock_name).set(mhz)
            except Exception as exc:
                log.debug("GPU %d clock error: %s", gpu_idx, exc)

            # Fan speed (NVIDIA-specific; may be N/A on some cards)
            try:
                fan_pct = pynvml.nvmlDeviceGetFanSpeed(handle)
                gpu_fan.labels(gpu=str(gpu_idx)).set(fan_pct)
            except Exception as exc:
                log.debug("GPU %d fan speed error: %s", gpu_idx, exc)

            # Active compute processes (NVIDIA-specific)
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                gpu_compute_processes.labels(gpu=str(gpu_idx)).set(len(procs))
            except Exception as exc:
                log.debug("GPU %d process count error: %s", gpu_idx, exc)


# ---------------------------------------------------------------------------
# No-op backend (no GPU found — Ollama API metrics still work)
# ---------------------------------------------------------------------------

class NoOpBackend(GPUBackend):
    name = "none"

    def gpu_count(self) -> int:
        return 0

    def collect(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def detect_backend() -> GPUBackend:
    """Auto-detect GPU backend: AMD first, then NVIDIA, then no-op."""
    backend: GPUBackend = AMDBackend.detect() or NvidiaBackend.detect() or NoOpBackend()
    if isinstance(backend, NoOpBackend):
        log.warning(
            "No GPU backend detected (AMD sysfs or NVIDIA pynvml). "
            "GPU metrics will be unavailable; Ollama API metrics still active."
        )
    else:
        log.info("Using %s backend with %d GPU(s)", backend.name, backend.gpu_count())
    backend_info.labels(backend=backend.name, gpu_count=str(backend.gpu_count())).set(1)
    return backend


# ---------------------------------------------------------------------------
# sysfs helpers (AMD backend)
# ---------------------------------------------------------------------------

def _read_int(path: str) -> int | None:
    try:
        with open(path) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def _hwmon_dir(card_device_path: str) -> str | None:
    dirs = sorted(glob.glob(os.path.join(card_device_path, "hwmon", "hwmon*")))
    return dirs[0] if dirs else None


# ---------------------------------------------------------------------------
# GIN log parsing
# ---------------------------------------------------------------------------

GIN_PATTERN = re.compile(
    r'\[GIN\].*\|\s*(\d+)\s*\|\s*([\w\d.µns]+)\s*\|\s*([\d.]+(?:\.\d+)?)\s*\|\s*(\w+)\s+"([^"]+)"'
)

# OLLAMA_DEBUG=1: fires at the start of every inference request
# Example: msg="loading cache slot" id=0 cache=0 prompt=58657 used=0 remaining=58657
CACHE_SLOT_PATTERN = re.compile(
    r'msg="loading cache slot"\s+\S+\s+cache=\d+\s+prompt=(\d+)\s+used=(\d+)\s+remaining=\d+'
)

# OLLAMA_DEBUG=1: fires after each inference request completes (llama_print_timings)
# Example: llama:        eval time =   8765.43 ms /   298 runs
EVAL_TIMING_PATTERN = re.compile(
    r'(?<!prompt )\beval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s+runs'
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
    """Parse one GIN log line and update metrics if the request took >1s."""
    m = GIN_PATTERN.search(line)
    if not m:
        return

    status_code, duration_str, client_ip, method, path = m.groups()
    duration = parse_go_duration(duration_str)
    if duration <= 1.0:
        return

    endpoint = path.split('?')[0]
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


def _handle_debug_line(line: str) -> None:
    """Parse OLLAMA_DEBUG=1 log lines for context window metrics."""
    m = CACHE_SLOT_PATTERN.search(line)
    if m:
        prompt_tokens = int(m.group(1))
        reuse_tokens = int(m.group(2))
        ctx_prompt_tokens.set(prompt_tokens)
        ctx_kv_reuse_tokens.set(reuse_tokens)
        with _context_length_lock:
            ctx_len = _current_context_length
        if ctx_len > 0:
            ratio = prompt_tokens / ctx_len
            ctx_fill_ratio.set(ratio)
            log.info(
                "Context slot: prompt=%d tokens  reuse=%d  fill=%.1f%%  (ctx_len=%d)",
                prompt_tokens, reuse_tokens, ratio * 100, ctx_len,
            )
        else:
            log.info(
                "Context slot: prompt=%d tokens  reuse=%d  (ctx_len unknown)",
                prompt_tokens, reuse_tokens,
            )
        return

    m = EVAL_TIMING_PATTERN.search(line)
    if m:
        eval_tokens = int(m.group(1))
        ctx_eval_tokens.set(eval_tokens)
        log.info("Eval tokens: %d", eval_tokens)


def tail_docker_logs() -> None:
    """Daemon thread: follow docker logs for the Ollama container."""
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
                    _handle_debug_line(line)
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
            with _context_length_lock:
                global _current_context_length
                _current_context_length = ctx

    for stale in _prev_models - current_models:
        model_loaded.labels(model=stale).set(0)
    _prev_models = current_models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    backend = detect_backend()

    log.info("Starting Ollama Prometheus exporter on port %d", EXPORTER_PORT)
    log.info("Ollama URL: %s  |  Scrape interval: %ds", OLLAMA_URL, SCRAPE_INTERVAL)
    start_http_server(EXPORTER_PORT)

    log_thread = threading.Thread(target=tail_docker_logs, daemon=True)
    log_thread.start()
    log.info("Docker log tail thread started")

    while True:
        try:
            backend.collect()
        except Exception as exc:
            log.error("GPU metric collection error: %s", exc)
        try:
            collect_ollama_metrics()
        except Exception as exc:
            log.error("Ollama metric collection error: %s", exc)
        time.sleep(SCRAPE_INTERVAL)


if __name__ == "__main__":
    main()
