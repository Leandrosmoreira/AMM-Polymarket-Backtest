"""
Performance optimizations for maximum speed.

Includes:
- uvloop: Faster event loop (2-4x faster than default asyncio)
- orjson: Faster JSON parsing (10x faster than stdlib json)
- PyPy detection and optimization hints
- msgspec: Even faster serialization (if available)
- httpx with HTTP/2: Faster network requests

Usage:
    from .performance import setup_performance, fast_json_loads, fast_json_dumps

    # Call once at startup (BEFORE any asyncio imports)
    setup_performance()

    # Use for JSON operations
    data = fast_json_loads(response_text)
    text = fast_json_dumps(data)
"""
import sys
import os
import logging
from typing import Any, Union

logger = logging.getLogger(__name__)

# Track if optimizations are enabled
_uvloop_enabled = False
_orjson_enabled = False
_msgspec_enabled = False
_httpx_h2_enabled = False
_pypy_detected = False


def is_pypy() -> bool:
    """Check if running on PyPy."""
    return sys.implementation.name == 'pypy'


def setup_uvloop() -> bool:
    """
    Install uvloop as the default event loop policy.

    uvloop is 2-4x faster than the default asyncio event loop.
    Must be called before any asyncio operations.

    Note: uvloop doesn't work on Windows or PyPy.

    Returns:
        True if uvloop was successfully installed
    """
    global _uvloop_enabled

    if sys.platform == 'win32':
        logger.info("uvloop not available on Windows")
        return False

    if is_pypy():
        logger.info("uvloop not needed on PyPy (has own optimized event loop)")
        return False

    try:
        import uvloop
        import asyncio

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        _uvloop_enabled = True
        logger.info("uvloop enabled (2-4x faster async)")
        return True

    except ImportError:
        logger.warning("uvloop not installed. Install with: pip install uvloop")
        return False
    except Exception as e:
        logger.warning(f"Failed to enable uvloop: {e}")
        return False


def setup_orjson() -> bool:
    """
    Check if orjson is available.

    orjson is 10x faster than stdlib json.
    Works on both CPython and PyPy.

    Returns:
        True if orjson is available
    """
    global _orjson_enabled

    try:
        import orjson
        _orjson_enabled = True
        logger.info("orjson enabled (10x faster JSON)")
        return True

    except ImportError:
        logger.warning("orjson not installed. Install with: pip install orjson")
        return False


def setup_msgspec() -> bool:
    """
    Check if msgspec is available.

    msgspec is even faster than orjson for some operations.

    Returns:
        True if msgspec is available
    """
    global _msgspec_enabled

    try:
        import msgspec
        _msgspec_enabled = True
        logger.info("msgspec enabled (fastest serialization)")
        return True

    except ImportError:
        # msgspec is optional
        return False


def setup_httpx_h2() -> bool:
    """
    Check if httpx with HTTP/2 support is available.

    HTTP/2 enables connection multiplexing for faster requests.

    Returns:
        True if httpx[http2] is available
    """
    global _httpx_h2_enabled

    try:
        import httpx
        import h2
        _httpx_h2_enabled = True
        logger.info("httpx HTTP/2 enabled (faster network)")
        return True

    except ImportError:
        logger.info("httpx HTTP/2 not available. Install with: pip install httpx[http2]")
        return False


def setup_performance() -> dict:
    """
    Setup all performance optimizations.

    Call this once at the start of your application,
    BEFORE any asyncio operations.

    Returns:
        Dict with status of each optimization
    """
    global _pypy_detected

    logger.info("Setting up performance optimizations...")

    # Detect PyPy
    _pypy_detected = is_pypy()
    if _pypy_detected:
        logger.info("PyPy detected - JIT compiler active")

    # Enable optimizations
    uvloop_ok = setup_uvloop()
    orjson_ok = setup_orjson()
    msgspec_ok = setup_msgspec()
    httpx_h2_ok = setup_httpx_h2()

    status = {
        "pypy": _pypy_detected,
        "uvloop": uvloop_ok,
        "orjson": orjson_ok,
        "msgspec": msgspec_ok,
        "httpx_h2": httpx_h2_ok,
    }

    # Summary
    enabled = sum([uvloop_ok, orjson_ok, msgspec_ok, httpx_h2_ok])
    if _pypy_detected:
        enabled += 1

    if enabled >= 4:
        logger.info("All performance optimizations enabled!")
    elif enabled >= 2:
        logger.info(f"{enabled} performance optimizations enabled")
    else:
        logger.warning("Few performance optimizations available")

    return status


# Fast JSON functions - use best available library
def fast_json_loads(data: Union[bytes, str]) -> Any:
    """
    Fast JSON parsing using best available library.

    Priority: msgspec > orjson > stdlib json

    Args:
        data: JSON string or bytes

    Returns:
        Parsed JSON data
    """
    if _msgspec_enabled:
        import msgspec
        if isinstance(data, str):
            data = data.encode('utf-8')
        return msgspec.json.decode(data)

    if _orjson_enabled:
        import orjson
        if isinstance(data, str):
            data = data.encode('utf-8')
        return orjson.loads(data)

    import json
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    return json.loads(data)


def fast_json_dumps(data: Any, pretty: bool = False) -> str:
    """
    Fast JSON serialization using best available library.

    Priority: msgspec > orjson > stdlib json

    Args:
        data: Data to serialize
        pretty: If True, format with indentation

    Returns:
        JSON string
    """
    if _msgspec_enabled and not pretty:
        import msgspec
        return msgspec.json.encode(data).decode('utf-8')

    if _orjson_enabled:
        import orjson
        options = orjson.OPT_INDENT_2 if pretty else 0
        return orjson.dumps(data, option=options).decode('utf-8')

    import json
    if pretty:
        return json.dumps(data, indent=2)
    return json.dumps(data, separators=(',', ':'))


def fast_json_dumps_bytes(data: Any) -> bytes:
    """
    Fast JSON serialization to bytes (avoids encode step).

    Args:
        data: Data to serialize

    Returns:
        JSON bytes
    """
    if _msgspec_enabled:
        import msgspec
        return msgspec.json.encode(data)

    if _orjson_enabled:
        import orjson
        return orjson.dumps(data)

    import json
    return json.dumps(data, separators=(',', ':')).encode('utf-8')


# Status functions
def is_uvloop_enabled() -> bool:
    """Check if uvloop is enabled."""
    return _uvloop_enabled


def is_orjson_enabled() -> bool:
    """Check if orjson is enabled."""
    return _orjson_enabled


def is_msgspec_enabled() -> bool:
    """Check if msgspec is enabled."""
    return _msgspec_enabled


def is_httpx_h2_enabled() -> bool:
    """Check if httpx HTTP/2 is enabled."""
    return _httpx_h2_enabled


def get_performance_status() -> dict:
    """Get current performance optimization status."""
    return {
        "python_version": sys.version,
        "implementation": sys.implementation.name,
        "pypy": _pypy_detected,
        "uvloop": _uvloop_enabled,
        "orjson": _orjson_enabled,
        "msgspec": _msgspec_enabled,
        "httpx_h2": _httpx_h2_enabled,
    }


def print_performance_report():
    """Print a performance optimization report."""
    status = get_performance_status()

    print("=" * 60)
    print("PERFORMANCE OPTIMIZATION REPORT")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Implementation: {status['implementation'].upper()}")
    print()

    optimizations = [
        ("PyPy JIT", status['pypy'], "Faster execution via JIT compilation"),
        ("uvloop", status['uvloop'], "2-4x faster async event loop"),
        ("orjson", status['orjson'], "10x faster JSON parsing"),
        ("msgspec", status['msgspec'], "Fastest serialization"),
        ("HTTP/2", status['httpx_h2'], "Multiplexed network requests"),
    ]

    print("Optimizations:")
    for name, enabled, desc in optimizations:
        icon = "" if enabled else ""
        print(f"  {icon} {name}: {desc}")

    print()

    # Recommendations
    missing = [name for name, enabled, _ in optimizations if not enabled]
    if missing:
        print("Recommendations:")
        if "orjson" in missing:
            print("  pip install orjson")
        if "uvloop" in missing and sys.platform != 'win32':
            print("  pip install uvloop")
        if "msgspec" in missing:
            print("  pip install msgspec")
        if "HTTP/2" in missing:
            print("  pip install httpx[http2]")
        if "PyPy JIT" in missing:
            print("  Consider using PyPy: https://www.pypy.org/")

    print("=" * 60)


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("\nTesting performance module...\n")
    setup_performance()
    print()

    # Test JSON
    test_data = {"price": 0.4823, "size": 100, "side": "BUY", "nested": {"a": 1, "b": [1, 2, 3]}}

    print("JSON benchmark:")
    import time

    # Warm up
    for _ in range(100):
        encoded = fast_json_dumps(test_data)
        decoded = fast_json_loads(encoded)

    # Benchmark
    iterations = 10000
    start = time.perf_counter()
    for _ in range(iterations):
        encoded = fast_json_dumps(test_data)
        decoded = fast_json_loads(encoded)
    elapsed = time.perf_counter() - start

    print(f"  {iterations} encode/decode cycles in {elapsed:.3f}s")
    print(f"  {iterations/elapsed:.0f} ops/sec")

    print()
    print_performance_report()
