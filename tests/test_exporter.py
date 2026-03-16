"""Unit tests for ollama-exporter."""

import os
import sys

import pytest

# Allow importing exporter without running main()
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from exporter import GIN_PATTERN, _handle_gin_line, parse_go_duration


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


class TestHandleGinLine:
    """Tests for _handle_gin_line metric updates."""

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
        labels = exporter._last_info_labels
        assert labels is not None
        assert labels["endpoint"] == "/v1/messages"

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
