"""Fast JSON helpers — uses orjson when available (2-3x faster than stdlib).

Drop-in replacement for json.loads/json.dumps with consistent behaviour:
- loads(): accepts str or bytes (orjson is bytes-native)
- dumps(): always returns str (orjson returns bytes natively)

Use these in WebSocket message hot paths where every microsecond counts.
"""

try:
    import orjson as _orjson
    _USE_ORJSON = True
except ImportError:  # pragma: no cover
    import json as _json
    _USE_ORJSON = False


def loads(data):
    """Parse JSON string or bytes. Returns Python object."""
    if _USE_ORJSON:
        return _orjson.loads(data)
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8")
    return _json.loads(data)


def dumps(obj) -> str:
    """Serialise to JSON string."""
    if _USE_ORJSON:
        return _orjson.dumps(obj).decode("utf-8")
    return _json.dumps(obj)


def dumps_bytes(obj) -> bytes:
    """Serialise to JSON bytes (faster — skips decode for orjson)."""
    if _USE_ORJSON:
        return _orjson.dumps(obj)
    return _json.dumps(obj).encode("utf-8")


def using_orjson() -> bool:
    """True if orjson is the active backend."""
    return _USE_ORJSON
