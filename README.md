# ollama-exporter

A Prometheus exporter for [Ollama](https://ollama.com) inference servers. Supports both **AMD ROCm** and **NVIDIA CUDA** GPU hosts. The GPU backend is auto-detected at startup — no configuration required.

Per-request inference metrics are parsed from Ollama's GIN access logs via `docker logs`, giving you real request durations, endpoints, and client IPs without a proxy or any changes to Ollama.

## GPU backend auto-detection

| Backend | Trigger | Metrics source |
|---|---|---|
| **AMD/sysfs** | `/sys/class/drm/card*/device/gpu_busy_percent` found | Linux `amdgpu` sysfs — no `rocm-smi` required |
| **NVIDIA/pynvml** | NVML initialises via `pynvml` | NVIDIA Management Library via container runtime |
| **None** | No GPU found | Ollama API metrics still exposed |

AMD is tried first, then NVIDIA. The active backend is identified by the `ollama_gpu_backend_info` metric.

## Metrics

### Common GPU metrics (AMD and NVIDIA)

| Metric | Labels | Description |
|---|---|---|
| `ollama_gpu_utilization_percent` | `gpu` | Compute utilisation 0–100 |
| `ollama_gpu_memory_used_bytes` | `gpu` | VRAM used (bytes) |
| `ollama_gpu_memory_total_bytes` | `gpu` | VRAM total (bytes) |
| `ollama_gpu_temperature_celsius` | `gpu`, `sensor` | Temperature — AMD: edge/junction/memory; NVIDIA: core |
| `ollama_gpu_power_watts` | `gpu` | Average power draw (watts) |
| `ollama_gpu_high_util_duration_seconds` | `gpu` | Seconds GPU has been continuously >80% — used for hung-job detection |
| `ollama_active_job_elapsed_seconds` | `gpu` | Seconds since GPU crossed 20% utilisation (0 when idle) |
| `ollama_gpu_backend_info` | `backend`, `gpu_count` | Always 1; labels identify the active backend |

### NVIDIA-only metrics

| Metric | Labels | Description |
|---|---|---|
| `ollama_gpu_memory_bandwidth_utilization_percent` | `gpu` | Memory bandwidth utilisation (distinct from compute) |
| `ollama_gpu_clock_mhz` | `gpu`, `type` | Current clock speed — types: `sm`, `memory`, `graphics` |
| `ollama_gpu_fan_speed_percent` | `gpu` | Fan speed 0–100 |
| `ollama_gpu_compute_process_count` | `gpu` | Number of active compute processes on GPU |

### Ollama model / API

| Metric | Labels | Description |
|---|---|---|
| `ollama_model_loaded` | `model` | 1 if model is resident in VRAM |
| `ollama_model_vram_bytes` | `model` | VRAM consumed by model |
| `ollama_model_context_length` | `model` | Configured context length |
| `ollama_api_up` | — | 1 if `/api/ps` responds |
| `ollama_api_response_seconds` | — | `/api/ps` response latency |

### Per-request (parsed from GIN access logs)

| Metric | Labels | Description |
|---|---|---|
| `ollama_requests_total` | `endpoint`, `status`, `method` | Total completed requests (Counter) |
| `ollama_request_duration_seconds` | `endpoint` | Request duration histogram (>1s requests only) |
| `ollama_last_request_completed_timestamp` | — | Unix epoch of last completed request |
| `ollama_last_request_duration_seconds` | — | Duration of last completed request |
| `ollama_last_request_info` | `client`, `endpoint`, `status`, `method` | Always 1.0; labels carry last-request metadata |

## Requirements

- Ollama running in a Docker container (named `ollama` by default, or set `OLLAMA_CONTAINER`)
- Docker socket accessible at `/var/run/docker.sock`
- **AMD:** Linux `amdgpu` driver with sysfs exposed at `/sys/class/drm/`
- **NVIDIA:** NVIDIA container runtime with `NVIDIA_VISIBLE_DEVICES=all` (provides NVML access)

## Quick start

### AMD (ROCm)

```yaml
services:
  ollama-exporter:
    image: ghcr.io/mikeh-22/ollama-exporter:latest
    container_name: ollama-exporter
    restart: unless-stopped
    environment:
      - OLLAMA_URL=http://ollama:11434
    ports:
      - "9101:9101"
    volumes:
      - /sys:/sys:ro
      - /var/run/docker.sock:/var/run/docker.sock
    depends_on:
      - ollama
```

### NVIDIA (CUDA)

```yaml
services:
  ollama-exporter:
    image: ghcr.io/mikeh-22/ollama-exporter:latest
    container_name: ollama-exporter
    restart: unless-stopped
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - OLLAMA_URL=http://ollama:11434
    ports:
      - "9101:9101"
    volumes:
      - /sys:/sys:ro
      - /var/run/docker.sock:/var/run/docker.sock
    depends_on:
      - ollama
```

If Ollama is not on the same Docker network, replace `http://ollama:11434` with `http://172.17.0.1:11434` (Docker bridge gateway) or the host's LAN IP.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://ollama:11434` | Ollama API base URL |
| `OLLAMA_CONTAINER` | `ollama` | Docker container name to tail logs from |
| `SCRAPE_INTERVAL` | `10` | Seconds between GPU + API scrapes |
| `EXPORTER_PORT` | `9101` | Port to expose `/metrics` on |

## How it works

**GPU backend detection** runs once at startup. AMD detection checks for `/sys/class/drm/card*/device/gpu_busy_percent`; NVIDIA detection calls `pynvml.nvmlInit()`. The first successful backend wins.

**AMD GPU metrics** are read from the Linux `amdgpu` sysfs interface. Cards are auto-discovered — any DRM card exposing `gpu_busy_percent` is included. No `rocm-smi` binary required.

**NVIDIA GPU metrics** are read via [nvidia-ml-py](https://pypi.org/project/nvidia-ml-py/) (Python bindings to NVML, the same library powering `nvidia-smi`). Requires the NVIDIA container runtime to mount NVML into the container.

**Per-request metrics** are extracted by tailing `docker logs --follow ollama` in a background thread and parsing Ollama's GIN HTTP server access log lines. Only requests longer than 1 second are recorded to filter out polling noise.

**Active job detection** tracks when GPU utilisation crosses the 20% threshold, exposing elapsed time of the current job in real time. Sustained >80% utilisation increments `ollama_gpu_high_util_duration_seconds` for hung-job alerting.

## Grafana dashboard

A full Grafana provisioning bundle is available at **[mikeh-22/ollama-grafana-dashboard](https://github.com/mikeh-22/ollama-grafana-dashboard)**, including:

- **Dashboard** — GPU utilisation, VRAM, temperatures, power, clock speeds, model info, per-request stats, and hung-job detection
- **Alert rules** — OllamaAPIDown, InferenceJobHung, GPUOverheat, GPUMemoryNearCapacity, OllamaContainerHighCPU
- **Datasource config** — Prometheus datasource pre-configured

Supports multi-host setups (AMD + NVIDIA) via a `gpu_host` Prometheus label.

## Development

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
ruff check exporter.py tests/
```

## CI/CD

- **CI** (`ci.yml`): runs on every push and PR — lints with `ruff`, runs `pytest` (33 tests), and does a Docker build smoke test.
- **Release** (`release.yml`): triggered by a `v*` tag — builds and pushes a versioned image to GHCR.

```bash
git tag v1.2.0
git push origin v1.2.0
```
