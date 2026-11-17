"""Download step: redirects must not lead the fetch off https.

requests.get is replaced with a fake, so nothing touches the network. A response whose final
url, or any hop in its history, is not https must be refused before anything is written.
"""

import importlib

import pytest

pytestmark = pytest.mark.integration

START_URL = "https://example.test/data/file.parquet"
HTTPS_URL = "https://cdn.example.test/data/file.parquet"
HTTP_URL = "http://cdn.example.test/data/file.parquet"
PAYLOAD = [b"abc", b"def"]


def _download_module():
    return importlib.import_module("taxi_duration.ingest.download")


class FakeResponse:
    """Context-manager response with the parts http_fetch uses."""

    def __init__(self, url, history=()):
        self.url = url
        self.history = list(history)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=None):
        yield from PAYLOAD


def _patch_get(monkeypatch, response):
    download = _download_module()
    monkeypatch.setattr(
        download.requests, "get", lambda url, **kwargs: response, raising=True
    )
    return download


def test_redirect_to_http_is_refused(monkeypatch, tmp_path):
    download = _patch_get(monkeypatch, FakeResponse(HTTP_URL))
    dest = tmp_path / "out.part"

    with pytest.raises(ValueError):
        download.http_fetch(START_URL, dest)

    assert not dest.exists(), "refused download must not write the file"


def test_redirect_through_http_hop_is_refused(monkeypatch, tmp_path):
    hop = FakeResponse(HTTP_URL)
    download = _patch_get(monkeypatch, FakeResponse(HTTPS_URL, history=[hop]))
    dest = tmp_path / "out.part"

    with pytest.raises(ValueError):
        download.http_fetch(START_URL, dest)

    assert not dest.exists(), "refused download must not write the file"


def test_https_redirect_chain_is_accepted(monkeypatch, tmp_path):
    hop = FakeResponse("https://hop.example.test/data/file.parquet")
    download = _patch_get(monkeypatch, FakeResponse(HTTPS_URL, history=[hop]))
    dest = tmp_path / "out.part"

    download.http_fetch(START_URL, dest)

    assert dest.read_bytes() == b"".join(PAYLOAD)
