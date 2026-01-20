"""
Performance optimizations for the trading bots.

Includes:
- uvloop: Faster event loop (2-4x faster than default)
- orjson: Faster JSON parsing (10x faster than stdlib json)

Usage:
    from .performance import setup_performance, fast_json_loads, fast_json_dumps

    # Call once at startup
    setup_performance()

    # Use for JSON operations
    data = fast_json_loads(response_text)
    text = fast_json_dumps(data)
"""
import sys
import logging

logger = logging.getLogger(__name__)

# Track if optimizations are enabled
_uvloop_enabled = False
_orjson_enabled = False


def setup_uvloop() -> bool:
    """
    Install uvloop as the default event loop policy.

    uvloop is 2-4x faster than the default asyncio event loop.
    Must be called before any asyncio operations.

    Returns:
        True if uvloop was successfully installed
    """
    global _uvloop_enabled

    try:
        import uvloop
        import asyncio

        # Set uvloop as the default event loop policy
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        _uvloop_enabled = True
        logger.info("✅ uvloop enabled (2-4x faster async)")
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

    Returns:
        True if orjson is available
    """
    global _orjson_enabled

    try:
        import orjson
        _orjson_enabled = True
        logger.info("✅ orjson enabled (10x faster JSON)")
        return True

    except ImportError:
        logger.warning("orjson not installed. Install with: pip install orjson")
        return False


def setup_performance() -> dict:
    """
    Setup all performance optimizations.

    Call this once at the start of your application,
    before any asyncio operations.

    Returns:
        Dict with status of each optimization
    """
    logger.info("Setting up performance optimizations...")

    uvloop_ok = setup_uvloop()
    orjson_ok = setup_orjson()

    status = {
        "uvloop": uvloop_ok,
        "orjson": orjson_ok,
    }

    if uvloop_ok and orjson_ok:
        logger.info("🚀 All performance optimizations enabled!")
    elif uvloop_ok or orjson_ok:
        logger.info("⚠️ Some performance optimizations enabled")
    else:
        logger.warning("❌ No performance optimizations available")

    return status


def fast_json_loads(data: bytes | str) -> dict | list:
    """
    Fast JSON parsing using orjson if available.

    Args:
        data: JSON string or bytes

    Returns:
        Parsed JSON data
    """
    if _orjson_enabled:
        import orjson
        if isinstance(data, str):
            data = data.encode('utf-8')
        return orjson.loads(data)
    else:
        import json
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return json.loads(data)


def fast_json_dumps(data: dict | list, pretty: bool = False) -> str:
    """
    Fast JSON serialization using orjson if available.

    Args:
        data: Data to serialize
        pretty: If True, format with indentation

    Returns:
        JSON string
    """
    if _orjson_enabled:
        import orjson
        options = orjson.OPT_INDENT_2 if pretty else 0
        return orjson.dumps(data, option=options).decode('utf-8')
    else:
        import json
        if pretty:
            return json.dumps(data, indent=2)
        return json.dumps(data)


def is_uvloop_enabled() -> bool:
    """Check if uvloop is enabled."""
    return _uvloop_enabled


def is_orjson_enabled() -> bool:
    """Check if orjson is enabled."""
    return _orjson_enabled


def get_performance_status() -> dict:
    """Get current performance optimization status."""
    return {
        "uvloop": _uvloop_enabled,
        "orjson": _orjson_enabled,
        "python_version": sys.version,
        "implementation": sys.implementation.name,  # 'cpython' or 'pypy'
    }


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing performance module...")
    status = setup_performance()
    print(f"\nStatus: {status}")

    # Test JSON
    test_data = {"price": 0.4823, "size": 100, "side": "BUY"}

    encoded = fast_json_dumps(test_data)
    print(f"\nEncoded: {encoded}")

    decoded = fast_json_loads(encoded)
    print(f"Decoded: {decoded}")

    print(f"\nFull status: {get_performance_status()}")
