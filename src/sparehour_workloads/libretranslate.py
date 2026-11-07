"""Fixed adapter for a local CUDA or CPU LibreTranslate service.

Jobs provide one bounded text segment plus a schema-bounded source and target
language from the set the pinned service loads. The trusted route fixes the
local model-service endpoint, the wire format, and the response size bound. A
job never selects the model server, an API key, or any execution code.

Before the first translation on an endpoint the adapter probes the service's
``/languages`` listing and requires the requested pair to be loaded, so the
slot only serves jobs while the correct pinned models are actually resident.
A reachable, healthy service that simply has not loaded the requested pair
fails that one job with ``unsupported_language_pair``; only an unreachable or
broken service reports ``model_server_unavailable`` / ``model_server_error``.

The service exposes no generation parameters, so the adapter also screens
every response with a cheap sanity check: LibreTranslate's CPU int8 path
occasionally decodes a short source into thousands of repeats of one token.
Degenerate output is re-requested once and then fails the job with
``degenerate_output``; it is never written as an artifact. All response reads
are bounded: error bodies are capped and sanitized before they are embedded
in an ``ExecutionError``.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from sparehour.agent import ExecutionError


DEFAULT_ENDPOINT = "http://127.0.0.1:5001/translate"
DEFAULT_FORMAT = "text"
LANGUAGES_PATH = "/languages"

# Bounded streaming keeps a misbehaving service from exhausting worker disk.
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_ERROR_BYTES = 64 * 1024
MAX_ERROR_CHARS = 500
MAX_BATCH_SEGMENTS = 64
MAX_BATCH_CHARACTERS = 256_000
MAX_READY_ENDPOINTS = 8
PROBE_TIMEOUT_SECONDS = 15.0

# Degenerate-output screens. LibreTranslate's HTTP API exposes no max-length
# or repetition controls, so the executor validates responses itself.
LENGTH_RATIO = 5
LENGTH_SLACK = 64
REPEAT_RUN = 6
MIN_DISTINCT_TOKENS = 24
LOW_DISTINCT_RATIO = 0.2
SOURCE_DISTINCT_RATIO = 0.5
# Text with at least this many characters and fewer than one space per
# SPACE_DENSITY characters (CJK, Thai, ...) is tokenized per character.
CHAR_TOKEN_MIN_CHARS = 40
SPACE_DENSITY = 20
MAX_NGRAM = 4

_CONTROL_CHARS = re.compile(r"[^\x20-\x7e\r\n\t]")

# (languages_url, source, target) pairs confirmed loaded by /languages.
# Bounded: routes pin one endpoint, so this never grows under normal use.
_READY_PAIRS: set[tuple[str, str, str]] = set()
_READY_LOCK = threading.Lock()

def _languages_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    path, sep, _ = base.rpartition("/translate")
    if sep:
        return f"{path}{LANGUAGES_PATH}"
    return f"{base}{LANGUAGES_PATH}"



def _read_bounded_body(response: httpx.Response, limit: int) -> bytes:
    """Drain a streamed response body, rejecting anything over ``limit``."""
    size = 0
    chunks: list[bytes] = []
    for chunk in response.iter_bytes():
        size += len(chunk)
        if size > limit:
            raise ExecutionError(
                "output_validation_failed",
                f"LibreTranslate response exceeds {limit} bytes",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _error_detail(response: httpx.Response) -> str:
    """Bounded, sanitized excerpt of a failed response body."""
    body = _read_bounded_body(response, MAX_ERROR_BYTES)
    text = body.decode("utf-8", errors="replace")
    text = _CONTROL_CHARS.sub(" ", text)[:MAX_ERROR_CHARS]
    return text.strip()


def _ensure_ready(
    *,
    languages_url: str,
    source: str,
    target: str,
) -> None:
    """Confirm the pinned service has the pair's models actually loaded."""
    key = (languages_url, source, target)
    with _READY_LOCK:
        if key in _READY_PAIRS:
            return
    try:
        with httpx.stream(
            "GET", languages_url, timeout=PROBE_TIMEOUT_SECONDS, trust_env=False
        ) as response:
            if response.status_code != 200:
                _read_bounded_body(response, MAX_ERROR_BYTES)
                raise ExecutionError(
                    "model_server_unavailable",
                    f"LibreTranslate /languages returned HTTP "
                    f"{response.status_code}",
                )
            body = _read_bounded_body(response, MAX_RESPONSE_BYTES)
    except httpx.HTTPError as exc:
        raise ExecutionError("model_server_unavailable", str(exc)) from exc

    try:
        languages = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ExecutionError(
            "model_server_error", f"unexpected /languages response: {exc}"
        ) from exc
    if not isinstance(languages, list) or not all(
        isinstance(entry, dict)
        and isinstance(entry.get("code"), str)
        and isinstance(entry.get("targets"), list)
        for entry in languages
    ):
        raise ExecutionError(
            "model_server_error", "unexpected /languages response shape"
        )

    loaded_targets: set[str] = set()
    for entry in languages:
        if source != "auto" and entry["code"] != source:
            continue
        loaded_targets.update(str(item) for item in entry["targets"])
    if target not in loaded_targets:
        # The service is up and healthy; this particular job asked for a pair
        # it never loaded. That is a job-level failure: "model_load_failed"
        # drains the whole worker (sparehour.agent.DRAINING_EXECUTION_ERRORS),
        # and one bad job must not stop the slot from serving loaded pairs.
        raise ExecutionError(
            "unsupported_language_pair",
            f"LibreTranslate has no loaded model for {source!r} -> {target!r}",
        )
    with _READY_LOCK:
        if len(_READY_PAIRS) < MAX_READY_ENDPOINTS:
            _READY_PAIRS.add(key)




def _translate(
    *,
    endpoint: str,
    request_format: str,
    source: str,
    target: str,
    texts: str | list[str],
    timeout: float,
    alternatives: int = 0,
) -> str | list[str]:
    _ensure_ready(
        languages_url=_languages_url(endpoint), source=source, target=target
    )
    request_body = {
        "q": texts,
        "source": source,
        "target": target,
        "format": request_format,
    }
    if alternatives:
        request_body["alternatives"] = alternatives
    try:
        with httpx.stream(
            "POST", endpoint, json=request_body, timeout=timeout, trust_env=False
        ) as response:
            if response.status_code >= 400:
                detail = _error_detail(response)
                if response.status_code >= 500:
                    raise ExecutionError(
                        "model_server_error",
                        f"LibreTranslate returned HTTP {response.status_code}: {detail}",
                    )
                raise ExecutionError(
                    "translation_rejected",
                    f"LibreTranslate rejected the input (HTTP {response.status_code}): {detail}",
                )
            body = _read_bounded_body(response, MAX_RESPONSE_BYTES)
    except httpx.HTTPError as exc:
        raise ExecutionError("model_server_unavailable", str(exc)) from exc

    try:
        document = json.loads(body)
        return document["translatedText"]
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as exc:
        raise ExecutionError(
            "model_server_error", f"unexpected LibreTranslate response: {exc}"
        ) from exc


def _tokenize(text: str) -> list[str]:
    """Whitespace tokens, or per-character tokens for unspaced CJK/Thai."""
    if (
        len(text) >= CHAR_TOKEN_MIN_CHARS
        and text.count(" ") * SPACE_DENSITY < len(text)
    ):
        return list(text)
    return text.split()


def _longest_repeat_run(tokens: list[str]) -> int:
    """Longest run of one n-gram (n <= MAX_NGRAM) repeated back-to-back.

    Linear in the token count: for each width the run ending at each token
    either starts fresh or extends the run one period back.
    """
    best = 0
    for width in range(1, MAX_NGRAM + 1):
        if len(tokens) < width:
            break
        runs = [1] * len(tokens)
        for index in range(width, len(tokens)):
            if tokens[index : index + width] == tokens[index - width : index]:
                runs[index] = runs[index - width] + 1
                if runs[index] > best:
                    best = runs[index]
    return best


def _distinct_ratio(tokens: list[str]) -> float:
    """Distinct-token ratio; 1.0 for empty input so it never looks cyclic."""
    if not tokens:
        return 1.0
    return len(set(tokens)) / len(tokens)


def _degenerate_reason(source: str, translated: str) -> str | None:
    """Return a short human reason when a translation fails the sanity check.

    Screens the response of a service that exposes no generation controls.
    Both rules compare against the same metric on the source text, so prose
    that is legitimately repetitive (or legitimately longer) passes: normal
    translations, including zh/ja -> en character expansion, stay clean.
    """
    if len(translated) > LENGTH_RATIO * len(source) + LENGTH_SLACK:
        return (
            f"translated length {len(translated)} exceeds "
            f"{LENGTH_RATIO}x the {len(source)}-character source"
        )

    output_tokens = _tokenize(translated)
    output_run = _longest_repeat_run(output_tokens)
    if output_run >= REPEAT_RUN:
        source_run = _longest_repeat_run(_tokenize(source))
        if output_run > 2 * source_run:
            return f"an n-gram repeats {output_run} times in a row"

    if len(output_tokens) >= MIN_DISTINCT_TOKENS:
        output_ratio = _distinct_ratio(output_tokens)
        if output_ratio < LOW_DISTINCT_RATIO:
            source_tokens = _tokenize(source)
            if _distinct_ratio(source_tokens) >= SOURCE_DISTINCT_RATIO:
                return (
                    f"only {len(set(output_tokens))} distinct tokens "
                    f"in {len(output_tokens)}"
                )
    return None


def _first_degenerate_segment(
    sources: list[str],
    translations: list[str],
) -> tuple[int, str] | None:
    """Index and reason of the first degenerate segment, else None."""
    for index, (source, translated) in enumerate(zip(sources, translations)):
        reason = _degenerate_reason(source, translated)
        if reason is not None:
            return index, reason
    return None


def _retranslate(request: dict[str, Any]) -> str | list[str]:
    """Re-request a translation that failed the sanity check.

    Argos Translate keeps the previous request's paragraphs in a per-pair
    cache, so repeating the identical request returns the same degenerate
    text. Asking for one alternative changes the hypothesis count, which
    misses that cache and forces a fresh decode; ``translatedText`` is still
    the best hypothesis. A service configured to refuse alternatives makes
    the retry fail as ``degenerate_output`` rather than a misleading
    ``translation_rejected``.
    """
    try:
        return _translate(**request, alternatives=1)
    except ExecutionError as exc:
        if exc.code != "translation_rejected":
            raise
        raise ExecutionError(
            "degenerate_output",
            f"LibreTranslate output failed sanity check and the retry was "
            f"rejected: {exc}",
        ) from exc


class LibreTranslateExecutor:
    def execute(
        self,
        *,
        job: dict[str, Any],
        workdir: Path,
        gpu_index: int,
    ) -> list[dict[str, Any]]:
        # gpu_index is intentionally unused: the pinned model service owns the
        # GPU for this route; the slot merely gates concurrent claim capacity.
        payload = job["payload"]
        environment = job["executor"].get("environment") or {}
        request = {
            "endpoint": environment.get("LIBRETRANSLATE_ENDPOINT", DEFAULT_ENDPOINT),
            "request_format": environment.get("LIBRETRANSLATE_FORMAT", DEFAULT_FORMAT),
            "source": str(payload["source"]),
            "target": str(payload["target"]),
            "texts": str(payload["text"]),
            "timeout": float(job["executor"].get("timeout_seconds") or 180),
        }
        translated = _translate(**request)
        if not isinstance(translated, str):
            raise ExecutionError(
                "model_server_error", "translatedText is not a single text segment"
            )

        reason = _degenerate_reason(str(payload["text"]), translated)
        if reason is not None:
            # Degenerate decodes are nondeterministic, so one re-request
            # usually returns sane text; the readiness probe is cached and
            # this costs a single extra POST. The bad output is never written.
            translated = _retranslate(request)
            if not isinstance(translated, str):
                raise ExecutionError(
                    "model_server_error",
                    "translatedText is not a single text segment",
                )
            reason = _degenerate_reason(str(payload["text"]), translated)
            if reason is not None:
                raise ExecutionError(
                    "degenerate_output",
                    f"LibreTranslate output failed sanity check: {reason}",
                )

        output = workdir / "translation.txt"
        output.write_text(translated, encoding="utf-8")
        return [{"name": "translation", "path": output, "media_type": "text/plain"}]


class LibreTranslateBatchExecutor:
    """Translate one ordered, same-language batch through a warm local service.

    The explicit batch is generic: producers retain ownership of item identity,
    while SPAREHOUR validates only ordered text segments and an allowlisted language
    pair. This amortizes leases, HTTP setup, artifacts, and queue transitions.
    """

    def execute(
        self,
        *,
        job: dict[str, Any],
        workdir: Path,
        gpu_index: int,
    ) -> list[dict[str, Any]]:
        del gpu_index
        payload = job["payload"]
        segments = [str(value) for value in payload["segments"]]
        if not 1 <= len(segments) <= MAX_BATCH_SEGMENTS:
            raise ExecutionError(
                "executor_contract_invalid",
                f"batch must contain 1-{MAX_BATCH_SEGMENTS} segments",
            )
        total_characters = sum(len(value) for value in segments)
        if total_characters > MAX_BATCH_CHARACTERS:
            raise ExecutionError(
                "executor_contract_invalid",
                f"batch exceeds {MAX_BATCH_CHARACTERS} characters",
            )

        environment = job["executor"].get("environment") or {}
        started = time.monotonic()
        request = {
            "endpoint": environment.get("LIBRETRANSLATE_ENDPOINT", DEFAULT_ENDPOINT),
            "request_format": environment.get("LIBRETRANSLATE_FORMAT", DEFAULT_FORMAT),
            "source": str(payload["source"]),
            "target": str(payload["target"]),
            "texts": segments,
            "timeout": float(job["executor"].get("timeout_seconds") or 600),
        }

        def validated(value: str | list[str]) -> list[str]:
            if not isinstance(value, list) or len(value) != len(segments):
                raise ExecutionError(
                    "output_validation_failed",
                    "LibreTranslate returned a batch with the wrong length",
                )
            if not all(isinstance(item, str) for item in value):
                raise ExecutionError(
                    "output_validation_failed",
                    "LibreTranslate batch contains a non-string translation",
                )
            return value

        translated = validated(_translate(**request))
        bad = _first_degenerate_segment(segments, translated)
        if bad is not None:
            # One degenerate segment spoils the whole batch; re-request it
            # once, then fail naming the first offending segment index.
            translated = validated(_retranslate(request))
            bad = _first_degenerate_segment(segments, translated)
            if bad is not None:
                index, reason = bad
                raise ExecutionError(
                    "degenerate_output",
                    f"LibreTranslate segment {index} failed sanity check: {reason}",
                )

        document = {
            "job_id": job["id"],
            "source_language": payload["source"],
            "target_language": payload["target"],
            "segment_count": len(segments),
            "source_character_count": total_characters,
            "elapsed_seconds": time.monotonic() - started,
            "translations": translated,
        }
        output = workdir / "translations.json"
        output.write_text(
            json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return [{"name": "translations", "path": output, "media_type": "application/json"}]
