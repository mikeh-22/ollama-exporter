# ollama-exporter

A Prometheus exporter for [Ollama](https://ollama.com) inference servers running on AMD GPUs. Exposes GPU hardware metrics via sysfs and per-request inference metrics parsed from Ollama's access logs — no `rocm-smi` dependency required.

## Metrics

### AMD GPU (via sysfs)

| Metric | Labels | Description |
|---|---|---|
| `ollama_gpu_utilization_percent` | `gpu` | Compute utilisation 0–100 |
| `ollama_gpu_memory_used_bytes` | `gpu` | VRAM used (bytes) |
| `ollama_gpu_memory_total_bytes` | `gpu` | VRAM total (bytes) |
| `ollama_gpu_temperature_celsius` | `gpu`, `sensor` | Temperature (edge / junction / memory) |
| `ollama_gpu_power_watts` | `gpu` | Average power draw (watts) |
| `ollama_gpu_high_util_duration_seconds` | `gpu` | Seconds GPU has been continuously >80% — used for hung-job detection |
| `ollama_active_job_elapsed_seconds` | `gpu` | Seconds since GPU crossed 20% utilisation (0 when idle) |

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

- AMD GPU with `amdgpu` driver (exposes `/sys/class/drm/card*/device/gpu_busy_percent`)
- Ollama running in a Docker container named `ollama` (or set `OLLAMA_CONTAINER`)
- Docker socket accessible at `/var/run/docker.sock`
- Docker CLI available (included in the image)

## Quick start

```yaml
services:
  ollama-exporter:
    image: ghcr.io/mikeh-22/ollama-exporter:latest
    container_name: ollama-exporter
    restart: unless-stopped
    environment:
      - OLLAMA_URL=http://ollama:11434
      - OLLAMA_CONTAINER=ollama
      - SCRAPE_INTERVAL=10
    ports:
      - "9101:9101"
    volumes:
      - /sys:/sys:ro
      - /var/run/docker.sock:/var/run/docker.sock
    depends_on:
      - ollama
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://ollama:11434` | Ollama API base URL |
| `OLLAMA_CONTAINER` | `ollama` | Docker container name to tail logs from |
| `SCRAPE_INTERVAL` | `10` | Seconds between sysfs + API scrapes |
| `EXPORTER_PORT` | `9101` | Port to expose `/metrics` on |

## How it works

**GPU metrics** are read directly from the Linux `amdgpu` sysfs interface under `/sys/class/drm/card*/device/`. Cards are auto-discovered — any DRM card exposing `gpu_busy_percent` is included.

**Per-request metrics** are extracted by tailing `docker logs --follow ollama` in a background thread and parsing Ollama's GIN HTTP server access log lines. Only requests taking longer than 1 second are recorded (to filter out polling noise). This gives you real request durations, endpoints, and client IPs without requiring a proxy or any changes to Ollama.

**Active job detection** uses the GPU utilisation threshold (≥20%) to track when a job starts and compute elapsed time in real time.

## Grafana dashboard

A full Grafana dashboard with GPU utilisation, VRAM, temperatures, power, model info, per-request stats, and hung-job detection panels is included in the [homelab-configs](https://github.com/mikeh-22/homelab-configs) repository.

## Development

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
ruff check exporter.py tests/
```

## CI/CD

- **CI** (`ci.yml`): runs on every push and PR — lints with `ruff`, runs `pytest`, and does a Docker build smoke test.
- **Release** (`release.yml`): triggered by a `v*` tag — builds and pushes a versioned image to GHCR.

```bash
git tag v1.0.0
git push origin v1.0.0
```
