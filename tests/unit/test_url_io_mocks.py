"""S3 + HTTP coverage for krunch.url_io via unittest.mock.

The integration suite hits real S3 + live HTTP; here we verify the
URL-parsing branches and the request shapes (right HTTP headers,
right boto3 calls, right S3 key extraction) without any network."""
import io
import pytest
from unittest.mock import MagicMock, patch

from krunch import url_io


# ---- S3 path ---------------------------------------------------------------

@pytest.fixture
def fake_s3():
    """Patch the lazy boto3 client factory and return a MagicMock the
    test can configure + assert against."""
    client = MagicMock()
    with patch.object(url_io, "_s3", return_value=client):
        yield client


def test_s3_size_calls_head_object(fake_s3):
    fake_s3.head_object.return_value = {"ContentLength": 4242}
    assert url_io.size("s3://my-bucket/path/to/blob.bin") == 4242
    fake_s3.head_object.assert_called_once_with(
        Bucket="my-bucket", Key="path/to/blob.bin"
    )


def test_s3_read_all_returns_body_bytes(fake_s3):
    fake_s3.get_object.return_value = {"Body": io.BytesIO(b"payload")}
    assert url_io.read_all("s3://b/k") == b"payload"
    fake_s3.get_object.assert_called_once_with(Bucket="b", Key="k")


def test_s3_read_range_uses_inclusive_byte_header(fake_s3):
    """Real S3 Range syntax is `bytes=<start>-<end>` inclusive on both
    ends; krunch passes `[start, end)` Pythonic semantics so we have to
    subtract 1. Pin that translation."""
    fake_s3.get_object.return_value = {"Body": io.BytesIO(b"abc")}
    url_io.read_range("s3://b/k", 0, 100)
    args = fake_s3.get_object.call_args.kwargs
    assert args["Range"] == "bytes=0-99"


def test_s3_write_calls_put_object(fake_s3):
    url_io.write("s3://b/k", b"data")
    fake_s3.put_object.assert_called_once_with(Bucket="b", Key="k", Body=b"data")


def test_s3_delete_swallows_errors(fake_s3):
    """Finalize cleanup races mean the part may already be gone — delete
    must not raise. boto3 raises ClientError; we catch broadly."""
    fake_s3.delete_object.side_effect = RuntimeError("nope")
    url_io.delete("s3://b/k")  # should not raise


def test_s3_url_missing_key_raises():
    with pytest.raises(ValueError, match="missing key"):
        url_io.read_all("s3://just-a-bucket")


# ---- HTTP path -------------------------------------------------------------

class _FakeHTTPResponse:
    def __init__(self, body: bytes, headers: dict | None = None):
        self._body = body
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


@patch("urllib.request.urlopen")
def test_http_size_reads_content_length_header(urlopen):
    urlopen.return_value = _FakeHTTPResponse(b"", {"Content-Length": "1024"})
    assert url_io.size("https://example.com/x") == 1024


@patch("urllib.request.urlopen")
def test_http_size_raises_when_content_length_missing(urlopen):
    urlopen.return_value = _FakeHTTPResponse(b"", {})
    with pytest.raises(ValueError, match="no Content-Length"):
        url_io.size("https://example.com/x")


@patch("urllib.request.urlopen")
def test_http_read_all(urlopen):
    urlopen.return_value = _FakeHTTPResponse(b"hello")
    assert url_io.read_all("http://example.com/x") == b"hello"


@patch("urllib.request.urlopen")
def test_http_read_range_sets_byte_header(urlopen):
    urlopen.return_value = _FakeHTTPResponse(b"abc")
    url_io.read_range("http://example.com/x", 5, 10)
    req = urlopen.call_args.args[0]
    assert req.get_header("Range") == "bytes=5-9"


# ---- Write rejection on unsupported schemes --------------------------------

def test_write_rejects_http():
    with pytest.raises(ValueError, match="write not supported"):
        url_io.write("https://example.com/x", b"data")
