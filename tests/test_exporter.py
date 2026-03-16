"""Unit tests for ollama-exporter."""

import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# Allow importing exporter without running main()
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from exporter import (
    GIN_PATTERN,
    AMDBackend,
    NvidiaBackend,
    _handle_gin_line,
    _update_utilisation_tracking,
    parse_go_duration,
)

# ---------------------------------------------------------------------------
# parse_go_duration
# ---------------------------------------------------------------------------

class TestParseGoDuration:
    def test_minutes_and_seconds(self):
        assert parse_go_duration("2m47s") == pytest.approx(167.0)

    def test_minutes_only(self):
        assert parse_go_duration("5m") == pytest.approx(300.0)

    def test_seconds_only(self):
        assert parse_go_duration("45s") == pytest.approx(45.0)

    def test_fractional_seconds(self):
        assert parse_go_duration("1m2.345s") == pytest.approx(62.345)

    def test_microseconds(self):
        assert parse_go_duration("35.838µs") == pytest.approx(35.838e-6)

    def test_plain_microseconds(self):
        assert parse_go_duration("64µs") == pytest.approx(64e-6)

    def test_milliseconds(self):
        assert parse_go_duration("150ms") == pytest.approx(0.15)

    def test_hours_minutes_seconds(self):
        assert parse_go_duration("2h3m4s") == pytest.approx(7384.0)

    def test_nanoseconds(self):
        assert parse_go_duration("500ns") == pytest.approx(500e-9)

    def test_whitespace_stripped(self):
        assert parse_go_duration("  2m47s  ") == pytest.approx(167.0)

    def test_empty_string(self):
        assert parse_go_duration("") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# GIN log pattern matching
# ---------------------------------------------------------------------------

class TestGinPattern:
    SAMPLE_SLOW = (
        '[GIN] 2026/03/16 - 03:25:10 | 200 |         2m47s |   192.168.3.109 | POST     "/v1/messages?beta=true"'
    )
    SAMPLE_FAST = (
        '[GIN] 2026/03/16 - 03:25:19 | 200 |      40.566µs |      172.19.0.3 | GET      "/api/ps"'
    )
    SAMPLE_404 = (
        '[GIN] 2026/03/16 - 04:00:00 | 404 |         3m12s |   192.168.3.109 | POST     "/api/chat"'
    )

    def test_matches_slow_request(self):
        m = GIN_PATTERN.search(self.SAMPLE_SLOW)
        assert m is not None
        status, dur, client, method, path = m.groups()
        assert status == "200"
        assert dur == "2m47s"
        assert client == "192.168.3.109"
        assert method == "POST"
        assert path == "/v1/messages?beta=true"

    def test_matches_fast_request(self):
        m = GIN_PATTERN.search(self.SAMPLE_FAST)
        assert m is not None
        _, dur, _, _, _ = m.groups()
        assert dur == "40.566µs"

    def test_matches_non_200(self):
        m = GIN_PATTERN.search(self.SAMPLE_404)
        assert m is not None
        status, _, _, _, _ = m.groups()
        assert status == "404"

    def test_no_match_on_random_line(self):
        assert GIN_PATTERN.search("time=2026-03-16 level=info msg=starting") is None


# ---------------------------------------------------------------------------
# GIN line handler
# ---------------------------------------------------------------------------

class TestHandleGinLine:
    SLOW_LINE = (
        '[GIN] 2026/03/16 - 03:25:10 | 200 |         2m47s |   192.168.3.109 | POST     "/v1/messages?beta=true"'
    )
    FAST_LINE = (
        '[GIN] 2026/03/16 - 03:25:19 | 200 |      40.566µs |      172.19.0.3 | GET      "/api/ps"'
    )

    def test_slow_line_updates_counter(self):
        import exporter
        before = exporter.request_counter.labels(
            endpoint="/v1/messages", status="200", method="POST"
        )._value.get()
        _handle_gin_line(self.SLOW_LINE)
        after = exporter.request_counter.labels(
            endpoint="/v1/messages", status="200", method="POST"
        )._value.get()
        assert after == before + 1

    def test_slow_line_updates_last_duration(self):
        import exporter
        _handle_gin_line(self.SLOW_LINE)
        assert exporter.last_duration._value.get() == pytest.approx(167.0)

    def test_slow_line_strips_query_string(self):
        import exporter
        _handle_gin_line(self.SLOW_LINE)
        assert exporter._last_info_labels is not None
        assert exporter._last_info_labels["endpoint"] == "/v1/messages"

    def test_fast_line_is_ignored(self):
        import exporter
        before = exporter.request_counter.labels(
            endpoint="/api/ps", status="200", method="GET"
        )._value.get()
        _handle_gin_line(self.FAST_LINE)
        after = exporter.request_counter.labels(
            endpoint="/api/ps", status="200", method="GET"
        )._value.get()
        assert after == before  # no change

    def test_non_gin_line_is_ignored(self):
        # should not raise
        _handle_gin_line("completely unrelated log line")

    def test_last_info_labels_set(self):
        import exporter
        _handle_gin_line(self.SLOW_LINE)
        assert exporter._last_info_labels["client"] == "192.168.3.109"
        assert exporter._last_info_labels["method"] == "POST"
        assert exporter._last_info_labels["status"] == "200"


# ---------------------------------------------------------------------------
# Utilisation tracking
# ---------------------------------------------------------------------------

class TestUtilisationTracking:
    def setup_method(self):
        import exporter
        exporter._high_util_since.clear()
        exporter._gpu_active_since.clear()

    def test_active_threshold_starts_elapsed_timer(self):
        import exporter
        now = time.time()
        _update_utilisation_tracking(0, 50.0, time.monotonic(), now)
        assert exporter._gpu_active_since[0] is not None

    def test_below_active_threshold_clears_timer(self):
        import exporter
        _update_utilisation_tracking(0, 50.0, time.monotonic(), time.time())
        _update_utilisation_tracking(0, 5.0, time.monotonic(), time.time())
        assert exporter._gpu_active_since[0] is None

    def test_high_util_starts_hung_timer(self):
        import exporter
        _update_utilisation_tracking(0, 90.0, time.monotonic(), time.time())
        assert exporter._high_util_since[0] is not None

    def test_dropping_below_high_threshold_clears_hung_timer(self):
        import exporter
        _update_utilisation_tracking(0, 90.0, time.monotonic(), time.time())
        _update_utilisation_tracking(0, 50.0, time.monotonic(), time.time())
        assert exporter._high_util_since[0] is None


# ---------------------------------------------------------------------------
# AMD backend detection
# ---------------------------------------------------------------------------

class TestAMDBackend:
    def test_detect_returns_none_when_no_drm_cards(self, tmp_path):
        with patch("exporter.glob.glob", return_value=[]):
            result = AMDBackend.detect()
        assert result is None

    def test_detect_finds_card_with_gpu_busy_percent(self, tmp_path):
        # Create a fake sysfs card path
        card_dev = tmp_path / "card1" / "device"
        card_dev.mkdir(parents=True)
        (card_dev / "gpu_busy_percent").write_text("75\n")

        with patch("exporter.glob.glob", return_value=[str(card_dev)]):
            result = AMDBackend.detect()

        assert result is not None
        assert result.gpu_count() == 1
        assert result.name == "AMD/sysfs"

    def test_detect_ignores_card_without_gpu_busy_percent(self, tmp_path):
        card_dev = tmp_path / "card0" / "device"
        card_dev.mkdir(parents=True)
        # no gpu_busy_percent file

        with patch("exporter.glob.glob", return_value=[str(card_dev)]):
            result = AMDBackend.detect()

        assert result is None


# ---------------------------------------------------------------------------
# NVIDIA backend detection
# ---------------------------------------------------------------------------

class TestNvidiaBackend:
    def test_detect_returns_none_when_pynvml_not_installed(self):
        with patch.dict("sys.modules", {"pynvml": None}):
            result = NvidiaBackend.detect()
        assert result is None

    def test_detect_returns_none_when_nvml_init_fails(self):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = Exception("NVML not found")
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = NvidiaBackend.detect()
        assert result is None

    def test_detect_succeeds_with_mock_nvml(self):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA GeForce RTX 3090"

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = NvidiaBackend.detect()

        assert result is not None
        assert result.gpu_count() == 1
        assert result.name == "NVIDIA/pynvml"

    def test_collect_updates_gpu_utilisation(self):
        import exporter

        mock_pynvml = MagicMock()
        mock_handle = MagicMock()

        util_rates = MagicMock()
        util_rates.gpu = 85
        util_rates.memory = 60
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = util_rates

        mem_info = MagicMock()
        mem_info.used = 8 * 1024**3
        mem_info.total = 24 * 1024**3
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_info

        mock_pynvml.nvmlDeviceGetTemperature.return_value = 72
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 300_000  # 300W in mW
        mock_pynvml.nvmlDeviceGetClockInfo.return_value = 1800
        mock_pynvml.nvmlDeviceGetFanSpeed.return_value = 55
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = [MagicMock()]

        backend = NvidiaBackend(mock_pynvml, [mock_handle])
        backend.collect()

        assert exporter.gpu_util.labels(gpu="0")._value.get() == pytest.approx(85.0)
        assert exporter.gpu_mem_bandwidth_util.labels(gpu="0")._value.get() == pytest.approx(60.0)
        assert exporter.gpu_mem_used.labels(gpu="0")._value.get() == pytest.approx(8 * 1024**3)
        assert exporter.gpu_temp.labels(gpu="0", sensor="core")._value.get() == pytest.approx(72.0)
        assert exporter.gpu_power.labels(gpu="0")._value.get() == pytest.approx(300.0)
        assert exporter.gpu_fan.labels(gpu="0")._value.get() == pytest.approx(55.0)
        assert exporter.gpu_compute_processes.labels(gpu="0")._value.get() == pytest.approx(1.0)

    def test_collect_tolerates_individual_metric_errors(self):
        """Backend should not crash if one metric call raises (e.g. fan N/A)."""

        mock_pynvml = MagicMock()
        mock_handle = MagicMock()

        util_rates = MagicMock()
        util_rates.gpu = 0
        util_rates.memory = 0
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = util_rates
        mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = Exception("not supported")
        mock_pynvml.nvmlDeviceGetTemperature.side_effect = Exception("not supported")
        mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = Exception("not supported")
        mock_pynvml.nvmlDeviceGetClockInfo.side_effect = Exception("not supported")
        mock_pynvml.nvmlDeviceGetFanSpeed.side_effect = Exception("not supported")
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.side_effect = Exception("not supported")

        backend = NvidiaBackend(mock_pynvml, [mock_handle])
        backend.collect()  # must not raise
