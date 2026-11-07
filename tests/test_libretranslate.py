from __future__ import annotations

import fnmatch
import http.server
import json
import socket
import threading

import pytest

from sparehour.agent import DRAINING_EXECUTION_ERRORS, ExecutionError
from sparehour.registry import AdmissionError
from sparehour_workloads import libretranslate


_DEFAULT_LANGUAGES = [
    {"code": "tr", "name": "Turkish", "targets": ["en", "de", "es"]},
    {"code": "en", "name": "English", "targets": ["tr", "de", "es"]},
    {"code": "de", "name": "German", "targets": ["en", "tr"]},
    {"code": "es", "name": "Spanish", "targets": ["en", "tr"]},
]

# The degenerate response reproduced from the CPU int8 bug report: a short
# source decoded into endless repeats of one token.
_DEGENERATE_SAMPLE = (
    "Sie Sie für Sie für Sie für Sie für Sie für Sie für Sie sehr für Sie für "
    "Sie für Sie für Sie für Sie " + "für " * 300
)


class _StubService:
    """Minimal loopback stand-in for a pinned local LibreTranslate service."""

    def __init__(self, *, status=200, body=b'{"translatedText": "Hello world"}',
                 bodies=None, content_type="application/json", languages=None,
                 languages_status=200):
        self._languages = languages if languages is not None else _DEFAULT_LANGUAGES
        self._languages_status = languages_status
        self._bodies = list(bodies) if bodies is not None else [body]
        self._body_lock = threading.Lock()
        handler = _make_handler(self, status, content_type)
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/translate"
        self.requests = handler.requests
        self.language_probes = handler.language_probes
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def _next_body(self) -> bytes:
        """Response body for the next POST; the last one repeats forever."""
        with self._body_lock:
            if len(self._bodies) > 1:
                return self._bodies.pop(0)
            return self._bodies[0]

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _make_handler(service, status, content_type):
    class Handler(http.server.BaseHTTPRequestHandler):
        requests = []
        language_probes = []

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            Handler.requests.append(json.loads(self.rfile.read(length) or b"null"))
            body = service._next_body()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            Handler.language_probes.append(self.path)
            payload = json.dumps(service._languages).encode()
            self.send_response(service._languages_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    return Handler

@pytest.fixture(autouse=True)
def _clear_ready_pairs():
    libretranslate._READY_PAIRS.clear()
    yield
    libretranslate._READY_PAIRS.clear()



def _job(endpoint, payload):
    return {
        "id": "job-translate-1",
        "payload": payload,
        "executor": {
            "type": "plugin",
            "timeout_seconds": 30,
            "environment": {"LIBRETRANSLATE_ENDPOINT": endpoint},
        },
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "Merhaba dünya", "source": "tr", "target": "en"},
        {"text": "Merhaba dünya", "source": "auto", "target": "de"},
        {"text": "Hello world", "source": "en", "target": "es"},
    ],
)
def test_translates_segment_for_any_bounded_pair(tmp_path, payload):
    service = _StubService()
    try:
        result = libretranslate.LibreTranslateExecutor().execute(
            job=_job(service.url, payload), workdir=tmp_path, gpu_index=0
        )
    finally:
        service.close()

    # The wire request is assembled by the trusted route; only the bounded
    # source/target pair and text come from the job.
    assert service.requests == [
        {
            "q": payload["text"],
            "source": payload["source"],
            "target": payload["target"],
            "format": "text",
        }
    ]
    assert result == [
        {
            "name": "translation",
            "path": tmp_path / "translation.txt",
            "media_type": "text/plain",
        }
    ]
    assert (tmp_path / "translation.txt").read_text(encoding="utf-8") == "Hello world"


def test_translates_ordered_batch_and_reports_metrics(tmp_path):
    body = json.dumps({"translatedText": ["Hello", "World"]}).encode()
    service = _StubService(body=body)
    payload = {"segments": ["Merhaba", "Dünya"], "source": "tr", "target": "en"}
    try:
        result = libretranslate.LibreTranslateBatchExecutor().execute(
            job=_job(service.url, payload), workdir=tmp_path, gpu_index=0
        )
    finally:
        service.close()

    assert service.requests[0]["q"] == payload["segments"]
    assert result[0]["name"] == "translations"
    document = json.loads((tmp_path / "translations.json").read_text())
    assert document["translations"] == ["Hello", "World"]
    assert document["segment_count"] == 2
    assert document["source_character_count"] == 12
    assert document["source_language"] == "tr"
    assert document["target_language"] == "en"


def test_batch_rejects_wrong_length_response(tmp_path):
    service = _StubService(body=b'{"translatedText": ["only one"]}')
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateBatchExecutor().execute(
                job=_job(service.url, {"segments": ["a", "b"], "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "output_validation_failed"
    finally:
        service.close()


def test_admission_binds_batch_units_to_segment_count(registry):
    admitted = registry.admit(
        "translate-text-batch",
        {"segments": ["bir", "iki", "üç"], "source": "tr", "target": "en"},
    )
    assert admitted.route == "libretranslate-batch-v1"
    assert admitted.cost_unit == "segment"
    assert admitted.estimated_units == 3
    assert admitted.requirements["capability"] == "libretranslate-batch"


def test_client_error_is_not_retryable(tmp_path):
    service = _StubService(status=400, body=b'{"error": "invalid language"}')
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateExecutor().execute(
                job=_job(service.url, {"text": "hi", "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "translation_rejected"
    finally:
        service.close()


def test_server_error_is_retryable(tmp_path):
    service = _StubService(status=503, body=b"overloaded")
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateExecutor().execute(
                job=_job(service.url, {"text": "hi", "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "model_server_error"
    finally:
        service.close()


def test_malformed_success_response_is_a_service_error(tmp_path):
    service = _StubService(body=b"<html>maintenance</html>", content_type="text/html")
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateExecutor().execute(
                job=_job(service.url, {"text": "hi", "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "model_server_error"
    finally:
        service.close()


def test_oversized_response_is_rejected(tmp_path):
    body = b'{"translatedText": "' + b"a" * (libretranslate.MAX_RESPONSE_BYTES + 1) + b'"}'
    service = _StubService(body=body)
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateExecutor().execute(
                job=_job(service.url, {"text": "hi", "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "output_validation_failed"
    finally:
        service.close()


def test_unreachable_service_is_retryable_unavailability(tmp_path):
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    with pytest.raises(ExecutionError) as excinfo:
        libretranslate.LibreTranslateExecutor().execute(
            job=_job(f"http://127.0.0.1:{port}/translate",
                     {"text": "hi", "source": "tr", "target": "en"}),
            workdir=tmp_path,
            gpu_index=0,
        )
    assert excinfo.value.code == "model_server_unavailable"


def test_admission_binds_the_segment_contract(registry):
    admitted = registry.admit(
        "translate-segment",
        {"text": "Merhaba dünya", "source": "tr", "target": "en"},
    )
    assert admitted.route == "libretranslate-v1"
    assert admitted.cost_unit == "segment"
    assert admitted.estimated_units == 1
    assert admitted.requirements["capability"] == "libretranslate"
    assert admitted.requirements["model_packs"] == ["libretranslate-1.9.6-bounded-sbd"]
    patterns = [value.casefold() for value in admitted.requirements["gpu_models"]]
    assert any(fnmatch.fnmatch("nvidia rtx a5000 laptop gpu", value) for value in patterns)
    assert any(fnmatch.fnmatch("nvidia geforce gtx 1080 ti", value) for value in patterns)
    assert admitted.requirements["min_compute_capability"] == "6.1"
    # The endpoint is fixed by the trusted route; only the bounded pair rides
    # in the payload.
    assert admitted.executor["environment"]["LIBRETRANSLATE_ENDPOINT"].startswith(
        "http://127.0.0.1:"
    )
    assert "LIBRETRANSLATE_TARGET" not in admitted.executor["environment"]


@pytest.mark.parametrize(
    "payload, marker",
    [
        ({"source": "tr", "target": "en"}, "text"),
        ({"text": "hi", "target": "en"}, "source"),
        ({"text": "hi", "source": "tr"}, "target"),
        ({"text": "", "source": "tr", "target": "en"}, "text"),
        ({"text": "hi", "source": "klingon", "target": "en"}, "source"),
        ({"text": "hi", "source": "tr", "target": "klingon"}, "target"),
        ({"text": "hi", "source": "tr", "target": "auto"}, "target"),
        ({"text": "hi", "source": "tr", "target": "en", "extra": 1}, "extra"),
    ],
)
def test_admission_rejects_out_of_contract_payloads(registry, payload, marker):
    with pytest.raises(AdmissionError, match=marker):
        registry.admit("translate-segment", payload)


def test_probes_languages_once_then_translates(tmp_path):
    service = _StubService()
    try:
        executor = libretranslate.LibreTranslateExecutor()
        job = _job(service.url, {"text": "hi", "source": "tr", "target": "en"})
        executor.execute(job=job, workdir=tmp_path, gpu_index=0)
        executor.execute(job=job, workdir=tmp_path, gpu_index=0)
    finally:
        service.close()
    assert service.language_probes == ["/languages"]
    assert len(service.requests) == 2


def test_unloaded_language_pair_is_a_job_level_failure(tmp_path):
    service = _StubService(
        languages=[{"code": "tr", "name": "Turkish", "targets": ["de"]}]
    )
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateExecutor().execute(
                job=_job(service.url,
                         {"text": "hi", "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "unsupported_language_pair"
        assert service.requests == []
        # An unloaded pair is one job asking for the wrong thing; the service
        # itself is healthy, so this must never drain the worker.
        assert "unsupported_language_pair" not in DRAINING_EXECUTION_ERRORS
    finally:
        service.close()


def test_languages_probe_failure_is_unavailable(tmp_path):
    service = _StubService(languages_status=503)
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateExecutor().execute(
                job=_job(service.url,
                         {"text": "hi", "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "model_server_unavailable"
        assert service.requests == []
    finally:
        service.close()


def test_malformed_languages_response_is_service_error(tmp_path):
    service = _StubService(languages={"not": "a list"})
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateExecutor().execute(
                job=_job(service.url,
                         {"text": "hi", "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "model_server_error"
    finally:
        service.close()


def test_error_body_is_sanitized_and_bounded(tmp_path):
    detail = "x" * (libretranslate.MAX_ERROR_CHARS + 200) + "\x00SECRET\xff"
    service = _StubService(status=400, body=json.dumps({"error": detail}).encode())
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateExecutor().execute(
                job=_job(service.url,
                         {"text": "hi", "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "translation_rejected"
        assert "\x00" not in str(excinfo.value)
        assert len(str(excinfo.value)) < libretranslate.MAX_ERROR_CHARS + 200
    finally:
        service.close()


def test_oversized_error_body_is_bounded(tmp_path):
    body = b'{"error": "' + b"a" * (libretranslate.MAX_ERROR_BYTES + 1) + b'"}'
    service = _StubService(status=500, body=body)
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateExecutor().execute(
                job=_job(service.url,
                         {"text": "hi", "source": "tr", "target": "en"}),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "output_validation_failed"
    finally:
        service.close()


def _long_normal_pair() -> tuple[str, str]:
    """A ~2,000-character English paragraph and a ~1.2x German translation."""
    english, german = [], []
    for index in range(26):
        english.append(
            f"Record {index} shows the nightly warehouse count for aisle "
            f"{index} differed from the manifest by exactly {index} items."
        )
        german.append(
            f"Der Datensatz {index} zeigt, dass die nächtliche Lagerzählung "
            f"für Gang {index} sich vom Manifest um genau {index} Artikel "
            f"unterschied."
        )
    return " ".join(english), " ".join(german)


def test_degenerate_reason_flags_the_real_bad_sample():
    reason = libretranslate._degenerate_reason(
        "Same call twice, one job.", _DEGENERATE_SAMPLE
    )
    assert reason is not None


def test_degenerate_reason_flags_repetitive_cjk_output():
    reason = libretranslate._degenerate_reason("这是一个测试。", "好" * 200)
    assert reason is not None


def test_degenerate_reason_flags_repetition_within_length_bound():
    # Short enough to pass the length screen; caught by the n-gram run rule.
    source = "The worker restarted in the middle of this job and kept going."
    translated = "Der Arbeiter für für für für für für für startete neu."
    assert len(translated) <= libretranslate.LENGTH_RATIO * len(source)
    reason = libretranslate._degenerate_reason(source, translated)
    assert reason is not None and "in a row" in reason


def test_degenerate_reason_flags_unspaced_repetition_within_length_bound():
    # Unspaced (CJK) output is tokenized per character.
    source = "Please translate this sentence."
    translated = "好" * 48
    assert len(translated) <= libretranslate.LENGTH_RATIO * len(source)
    reason = libretranslate._degenerate_reason(source, translated)
    assert reason is not None and "in a row" in reason


@pytest.mark.parametrize(
    "source,translated",
    [
        ("Same call twice, one job.", "Der gleiche Anruf zweimal, ein Job."),
        ("Hello.", "Hallo."),
        ("One job.", "Ein Job."),
        ("你好，世界。", "Hello, world."),
        # A repetitive source may legitimately translate to a repetitive
        # output; only output far more cyclic than its source is degenerate.
        ("ha ha ha ha ha ha ha ha", "ha ha ha ha ha ha ha ha"),
        _long_normal_pair(),
    ],
)
def test_degenerate_reason_accepts_normal_translations(source, translated):
    assert libretranslate._degenerate_reason(source, translated) is None


def test_executor_retries_degenerate_output_once(tmp_path):
    sane = "Zweimal derselbe Anruf, ein Job."
    service = _StubService(
        bodies=[
            json.dumps({"translatedText": _DEGENERATE_SAMPLE}).encode(),
            json.dumps({"translatedText": sane}).encode(),
        ]
    )
    payload = {"text": "Same call twice, one job.", "source": "en", "target": "de"}
    try:
        libretranslate.LibreTranslateExecutor().execute(
            job=_job(service.url, payload), workdir=tmp_path, gpu_index=0
        )
    finally:
        service.close()

    assert len(service.requests) == 2
    # The retry asks for one alternative so Argos's last-request cache is
    # missed and the segment is decoded afresh.
    assert "alternatives" not in service.requests[0]
    assert service.requests[1]["alternatives"] == 1
    assert (tmp_path / "translation.txt").read_text(encoding="utf-8") == sane


def test_executor_retry_rejected_reports_degenerate_output(tmp_path, monkeypatch):
    calls = []

    def fake_translate(**kwargs):
        calls.append(kwargs.get("alternatives", 0))
        if len(calls) == 1:
            return _DEGENERATE_SAMPLE
        raise ExecutionError("translation_rejected", "alternatives not allowed")

    monkeypatch.setattr(libretranslate, "_translate", fake_translate)
    with pytest.raises(ExecutionError) as excinfo:
        libretranslate.LibreTranslateExecutor().execute(
            job=_job(
                "http://127.0.0.1:9/translate",
                {"text": "Same call twice, one job.", "source": "en", "target": "de"},
            ),
            workdir=tmp_path,
            gpu_index=0,
        )
    assert excinfo.value.code == "degenerate_output"
    assert calls == [0, 1]
    assert not (tmp_path / "translation.txt").exists()


def test_executor_fails_persistent_degenerate_output_without_artifact(tmp_path):
    service = _StubService(
        body=json.dumps({"translatedText": _DEGENERATE_SAMPLE}).encode()
    )
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateExecutor().execute(
                job=_job(
                    service.url,
                    {"text": "Same call twice, one job.", "source": "en", "target": "de"},
                ),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "degenerate_output"
        assert "sanity check" in str(excinfo.value)
        assert len(service.requests) == 2
        assert not (tmp_path / "translation.txt").exists()
    finally:
        service.close()


def test_batch_executor_retries_degenerate_segment(tmp_path):
    sane = "Zweimal derselbe Anruf, ein Job."
    service = _StubService(
        bodies=[
            json.dumps({"translatedText": ["Hallo", _DEGENERATE_SAMPLE]}).encode(),
            json.dumps({"translatedText": ["Hallo", sane]}).encode(),
        ]
    )
    payload = {
        "segments": ["Hello there.", "Same call twice, one job."],
        "source": "en",
        "target": "de",
    }
    try:
        libretranslate.LibreTranslateBatchExecutor().execute(
            job=_job(service.url, payload), workdir=tmp_path, gpu_index=0
        )
    finally:
        service.close()

    assert len(service.requests) == 2
    assert "alternatives" not in service.requests[0]
    assert service.requests[1]["alternatives"] == 1
    document = json.loads((tmp_path / "translations.json").read_text())
    assert document["translations"] == ["Hallo", sane]


def test_batch_executor_fails_persistent_degenerate_segment(tmp_path):
    service = _StubService(
        body=json.dumps({"translatedText": ["Hallo", _DEGENERATE_SAMPLE]}).encode()
    )
    try:
        with pytest.raises(ExecutionError) as excinfo:
            libretranslate.LibreTranslateBatchExecutor().execute(
                job=_job(
                    service.url,
                    {
                        "segments": ["Hello there.", "Same call twice, one job."],
                        "source": "en",
                        "target": "de",
                    },
                ),
                workdir=tmp_path,
                gpu_index=0,
            )
        assert excinfo.value.code == "degenerate_output"
        assert "segment 1" in str(excinfo.value)
        assert len(service.requests) == 2
        assert not (tmp_path / "translations.json").exists()
    finally:
        service.close()


def test_cpu_route_admits_cpu_slots_with_degenerate_retry(registry):
    admitted = registry.admit(
        "translate-segment",
        {"text": "Merhaba dünya", "source": "tr", "target": "en"},
        "libretranslate-cpu",
    )
    route = registry.get("translate-segment").routes["libretranslate-cpu"]

    assert admitted.route == "libretranslate-cpu"
    assert admitted.requirements["vendor"] == "cpu"
    assert admitted.requirements["capability"] == "libretranslate-cpu"
    assert admitted.requirements["model_packs"] == []
    assert admitted.requirements["min_cpu_cores"] == 2
    assert admitted.requirements["min_ram_mb"] == 4096
    assert admitted.requirements["min_disk_mb"] == 8192
    assert admitted.executor["handler"] == (
        "sparehour_workloads.libretranslate:LibreTranslateExecutor"
    )
    assert route.scaling.enabled is False
    assert route.scaling.max_slots == 4
    assert "degenerate_output" in admitted.retryable_error_codes
