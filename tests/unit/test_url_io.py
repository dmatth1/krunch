"""file:// path coverage for krunch.url_io.

s3:// and http(s):// paths are exercised by the integration suite
(real S3 / live HTTP); here we just hit the file-scheme branch end-to-end."""
import pytest
from pathlib import Path

from krunch import url_io


@pytest.fixture
def tmp_url(tmp_path):
    return f"file://{tmp_path}/blob.bin"


def test_write_then_read_all_roundtrip(tmp_url):
    payload = b"hello krunch" * 1000
    url_io.write(tmp_url, payload)
    assert url_io.read_all(tmp_url) == payload


def test_size_matches_written(tmp_url):
    url_io.write(tmp_url, b"abcdef")
    assert url_io.size(tmp_url) == 6


def test_read_range_slices_correctly(tmp_url):
    url_io.write(tmp_url, b"0123456789")
    assert url_io.read_range(tmp_url, 0, 4) == b"0123"
    assert url_io.read_range(tmp_url, 4, 10) == b"456789"
    assert url_io.read_range(tmp_url, 7, 8) == b"7"


def test_delete_removes_file(tmp_path):
    url = f"file://{tmp_path}/gone.bin"
    url_io.write(url, b"x")
    url_io.delete(url)
    assert not (tmp_path / "gone.bin").exists()


def test_delete_missing_is_noop(tmp_path):
    # file:// missing-target must not raise — workers may delete already-
    # collected parts during finalize cleanup races.
    url_io.delete(f"file://{tmp_path}/never_existed.bin")


def test_write_creates_parent_dirs(tmp_path):
    url = f"file://{tmp_path}/nested/deep/blob.bin"
    url_io.write(url, b"ok")
    assert (tmp_path / "nested" / "deep" / "blob.bin").read_bytes() == b"ok"


def test_unsupported_scheme_raises():
    with pytest.raises(ValueError, match="unsupported scheme"):
        url_io.read_all("ftp://example.com/x")


def test_url_without_scheme_raises():
    with pytest.raises(ValueError, match="not a URL"):
        url_io.read_all("/just/a/path")
