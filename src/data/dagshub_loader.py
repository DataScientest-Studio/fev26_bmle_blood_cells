"""Download model and test images from DagsHub if missing or outdated (hash-based check)."""

import json
import os
from pathlib import Path

import requests
import yaml


def _auth() -> tuple[str, str]:
    user = os.getenv("DAGSHUB_USER", "")
    token = os.getenv("DAGSHUB_TOKEN", "")
    if not user or not token:
        raise EnvironmentError(
            "DAGSHUB_USER and DAGSHUB_TOKEN must be set in your .env file."
        )
    return (user, token)


def _base_url() -> str:
    user = os.getenv("DAGSHUB_USER", "Dumegan")
    repo = os.getenv("DAGSHUB_REPO", "Bloodcells-project")
    return f"https://dagshub.com/{user}/{repo}"


def _remote_dvc_hash(dvc_filename: str) -> str:
    """Fetch the md5 hash from a .dvc manifest (tiny Git-tracked file, < 1 KB)."""
    url = f"{_base_url()}/raw/main/{dvc_filename}"
    r = requests.get(url, auth=_auth(), timeout=10)
    r.raise_for_status()
    manifest = yaml.safe_load(r.text)
    return manifest["outs"][0]["md5"]


def _cached_hash(version_file: Path) -> str:
    return version_file.read_text().strip() if version_file.exists() else ""


def _download_stream(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, auth=_auth(), stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)


def ensure_model(model_path: Path) -> bool:
    """Ensure DenseNet model is local and up to date.

    Fetches Models.dvc hash (< 1 KB) at each startup.
    Re-downloads the model (~27 MB) only if the hash differs.
    Returns True if the model was (re)downloaded.
    """
    version_file = model_path.parent / ".model_version"
    remote_hash = _remote_dvc_hash("Models.dvc")

    if model_path.exists() and _cached_hash(version_file) == remote_hash:
        return False

    url = f"{_base_url()}/resolve/main/Models/best_DenseNet_121.pth"
    _download_stream(url, model_path)
    version_file.write_text(remote_hash)
    return True


def _fetch_dir_manifest(dir_hash: str) -> list[dict]:
    """Fetch the DVC .dir manifest listing individual file hashes and paths."""
    prefix = dir_hash[:2]
    rest = dir_hash[2:].replace(".dir", "")
    url = f"{_base_url()}.dvc/files/md5/{prefix}/{rest}"
    r = requests.get(url, auth=_auth(), timeout=10)
    r.raise_for_status()
    return json.loads(r.text)


def ensure_source_100(data_dir: Path) -> bool:
    """Ensure Source_100 test images are local and up to date.

    Fetches Source_100.dvc hash at each startup.
    Re-downloads the 100 images (~1.6 MB) only if the hash differs.
    Returns True if images were (re)downloaded.
    """
    version_file = data_dir / ".source100_version"
    remote_hash = _remote_dvc_hash("Source_100.dvc")

    if data_dir.exists() and _cached_hash(version_file) == remote_hash:
        return False

    entries = _fetch_dir_manifest(remote_hash)
    for entry in entries:
        rel_path = entry["relpath"]
        file_hash = entry["md5"]
        prefix, rest = file_hash[:2], file_hash[2:]
        url = f"{_base_url()}.dvc/files/md5/{prefix}/{rest}"
        dest = data_dir / rel_path
        _download_stream(url, dest)

    version_file.write_text(remote_hash)
    return True
