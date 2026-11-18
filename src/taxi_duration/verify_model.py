"""Verify a registered model before loading it. Copy to src/taxi_duration/.

Order matters:
1. The model URI names an alias, never a version number or "latest".
2. The tracking server is on this computer (http on a loopback host) or a local store.
3. The model files are downloaded to a private folder. No model code runs yet.
4. Their hash must match the hash pinned in the deploy settings.
5. The model metadata must name the safe loader and format, ship no code,
   and trust only the types in the project's allow list.
6. The model loads from the same folder that was checked.

The pinned hash and the allow list come from the deploy settings, never from
the model store: whoever can write the store can also change what it says.
Print the hash to pin with:
    uv run python -m taxi_duration.verify_model models:/<name>@<alias>
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow
import yaml

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
LOCAL_SCHEMES = frozenset({"file", "sqlite"})
SAFE_LOADER_MODULES = frozenset({"mlflow.sklearn"})
SAFE_FORMATS = frozenset({"skops"})
ALIAS_URI = re.compile(r"^models:/[A-Za-z0-9_.\-]+@[A-Za-z0-9_\-]+$")


class ModelVerificationError(RuntimeError):
    """A check failed: the model must not be loaded."""


def check_alias_uri(model_uri: str) -> None:
    """Accept only models:/<name>@<alias>."""
    if not ALIAS_URI.match(model_uri):
        raise ModelVerificationError(
            f"{model_uri!r} is not models:/<name>@<alias>. Serve by alias, "
            "never by version number or latest."
        )


def check_tracking_uri(tracking_uri: str, allowed_hosts=LOOPBACK_HOSTS) -> None:
    """Accept a local store, or an http(s) server on an allowed host."""
    parsed = urlparse(tracking_uri)
    if parsed.scheme in LOCAL_SCHEMES:
        return
    if parsed.scheme in ("http", "https") and parsed.hostname in allowed_hosts:
        return
    raise ModelVerificationError(
        f"Tracking URI {tracking_uri!r} is not a local store or an allowed host."
    )


def check_strict_tracking_uri(uri: str) -> str:
    """Accept only http://127.0.0.1:5001, with an optional trailing slash."""
    try:
        parsed = urlparse(uri)
        port = parsed.port
    except ValueError:
        raise ValueError(f"MLFLOW_TRACKING_URI not allowed: {uri!r}") from None
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or port != 5001
        or "@" in parsed.netloc
        or parsed.path not in ("", "/")
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"MLFLOW_TRACKING_URI not allowed: {uri!r}")
    return uri


def model_sha256(model_dir: str | Path) -> str:
    """Hash every file in the model folder: relative path and content, sorted.

    Renaming, adding or removing a file changes the hash too.
    """
    root = Path(model_dir)
    digest = hashlib.sha256()
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def check_model_metadata(model_dir: str | Path, trusted_types=frozenset()) -> None:
    """Read MLmodel as plain YAML. Refuse unsafe loaders, formats, code and types."""
    path = Path(model_dir) / "MLmodel"
    if not path.is_file():
        raise ModelVerificationError("MLmodel file is missing.")
    flavors = (yaml.safe_load(path.read_text()) or {}).get("flavors", {})
    loader = flavors.get("python_function", {}).get("loader_module")
    if loader not in SAFE_LOADER_MODULES:
        raise ModelVerificationError(f"Loader {loader!r} is not allowed.")
    sk = flavors.get("sklearn", {})
    if sk.get("serialization_format") not in SAFE_FORMATS:
        raise ModelVerificationError(
            f"Format {sk.get('serialization_format')!r} is not allowed; use skops."
        )
    if sk.get("code") or (Path(model_dir) / "code").exists():
        raise ModelVerificationError("The model ships code; refusing.")
    extra = set(sk.get("skops_trusted_types") or []) - set(trusted_types)
    if extra:
        raise ModelVerificationError(
            f"Model trusts types not in the project allow list: {sorted(extra)}"
        )


def verify_local_dir(
    model_dir: str | Path, expected_sha256: str | None, trusted_types=frozenset()
) -> str:
    """Check a downloaded model folder. Return its hash."""
    if not expected_sha256 or not expected_sha256.strip():
        raise ModelVerificationError(
            "No pinned hash. Print it with `python -m taxi_duration.verify_model "
            "<model-uri>` and set it in the deploy settings."
        )
    actual = model_sha256(model_dir)
    if not hmac.compare_digest(actual, expected_sha256.strip().lower()):
        raise ModelVerificationError(
            f"Hash mismatch: pinned {expected_sha256.strip()}, found {actual}."
        )
    check_model_metadata(model_dir, trusted_types)
    return actual


def download(
    model_uri: str,
    tracking_uri: str,
    allowed_hosts=LOOPBACK_HOSTS,
    version=None,
) -> str:
    """Check the URIs, then download the model files to a private folder.

    With version, the download is pinned to that version of the alias's model
    instead of resolving the alias again.
    """
    check_alias_uri(model_uri)
    check_tracking_uri(tracking_uri, allowed_hosts)
    if version is not None:
        model_uri = f"{model_uri.split('@', 1)[0]}/{int(version)}"
    work_dir = tempfile.mkdtemp(prefix="verified-model-")
    return mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri, dst_path=work_dir, tracking_uri=tracking_uri
    )


def verify_then_load(
    model_uri: str,
    *,
    expected_sha256: str | None,
    tracking_uri: str,
    trusted_types=frozenset(),
    allowed_hosts=LOOPBACK_HOSTS,
    version=None,
) -> tuple[Any, str]:
    """Download, verify, then load from the verified folder. Return (model, hash)."""
    local_dir = download(model_uri, tracking_uri, allowed_hosts, version=version)
    sha = verify_local_dir(local_dir, expected_sha256, trusted_types)
    return mlflow.pyfunc.load_model(local_dir), sha


def main(argv: list[str]) -> int:
    """Print the hash to pin for a model URI. Checks the metadata, never loads."""
    if len(argv) != 1:
        print("usage: python -m taxi_duration.verify_model models:/<name>@<alias>")
        return 2
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is not set")
    check_strict_tracking_uri(tracking_uri)
    local_dir = download(argv[0], tracking_uri)
    check_model_metadata(local_dir, frozenset())
    print(f"MODEL_URI={argv[0]}")
    print(f"MODEL_SHA256={model_sha256(local_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
