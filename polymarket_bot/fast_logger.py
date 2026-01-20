"""
Fast JSONL logger for backtest data.
Optimized for minimal performance impact - uses buffered writes.

Supports both CSV and JSONL formats.
JSONL is preferred for structured data analysis.
"""
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal
import threading
import queue

# Try to use orjson for faster JSON serialization
try:
    import orjson
    def _json_dumps(data: dict) -> str:
        return orjson.dumps(data).decode('utf-8')
    _FAST_JSON = True
except ImportError:
    import json
    def _json_dumps(data: dict) -> str:
        return json.dumps(data, separators=(',', ':'))
    _FAST_JSON = False


class FastTradeLogger:
    """
    High-performance JSONL/CSV logger for trade/opportunity data.

    Features:
    - JSONL format (default) for structured data
    - CSV format for spreadsheet compatibility
    - Buffered writes (flushes every N records or M seconds)
    - Non-blocking (uses background thread)
    - Uses orjson if available (10x faster)
    - Minimal memory footprint
    """

    def __init__(
        self,
        log_dir: str = "logs",
        buffer_size: int = 10,
        flush_interval: float = 5.0,
        format: Literal["jsonl", "csv"] = "jsonl",
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.format = format

        # Create log files with timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "jsonl" if format == "jsonl" else "csv"
        self.trades_file = self.log_dir / f"trades_{ts}.{ext}"
        self.scans_file = self.log_dir / f"scans_{ts}.{ext}"

        # Queues for non-blocking writes
        self._trade_queue: queue.Queue = queue.Queue()
        self._scan_queue: queue.Queue = queue.Queue()

        # Write headers (only for CSV)
        if format == "csv":
            self._write_csv_headers()

        # Start background writer thread
        self._running = True
        self._writer_thread = threading.Thread(target=self._background_writer, daemon=True)
        self._writer_thread.start()

        # Stats
        self.trades_logged = 0
        self.scans_logged = 0

    def _write_csv_headers(self):
        """Write CSV headers."""
        with open(self.trades_file, 'w') as f:
            f.write("timestamp,market,price_up,price_down,pair_cost,profit_pct,order_size,investment,expected_profit,balance_after,ltm_bucket\n")

        with open(self.scans_file, 'w') as f:
            f.write("timestamp,market,up_ask,down_ask,pair_cost,has_opportunity,time_remaining_sec\n")

    def log_trade(
        self,
        market: str,
        price_up: float,
        price_down: float,
        pair_cost: float,
        profit_pct: float,
        order_size: float,
        investment: float,
        expected_profit: float,
        balance_after: float,
        ltm_bucket: Optional[int] = None,
    ):
        """Log a trade execution (non-blocking)."""
        ts = time.time()

        if self.format == "jsonl":
            data = {
                "ts": ts,
                "time": datetime.fromtimestamp(ts).isoformat(),
                "market": market,
                "price_up": round(price_up, 6),
                "price_down": round(price_down, 6),
                "pair_cost": round(pair_cost, 6),
                "profit_pct": round(profit_pct, 4),
                "order_size": order_size,
                "investment": round(investment, 4),
                "expected_profit": round(expected_profit, 4),
                "balance_after": round(balance_after, 4),
            }
            if ltm_bucket is not None:
                data["ltm_bucket"] = ltm_bucket
            row = _json_dumps(data) + "\n"
        else:
            row = f"{ts},{market},{price_up:.4f},{price_down:.4f},{pair_cost:.4f},{profit_pct:.2f},{order_size:.0f},{investment:.2f},{expected_profit:.2f},{balance_after:.2f},{ltm_bucket if ltm_bucket is not None else ''}\n"

        self._trade_queue.put(row)
        self.trades_logged += 1

    def log_scan(
        self,
        market: str,
        up_ask: float,
        down_ask: float,
        pair_cost: float,
        has_opportunity: bool,
        time_remaining_sec: int,
    ):
        """Log a market scan (non-blocking)."""
        ts = time.time()

        if self.format == "jsonl":
            data = {
                "ts": ts,
                "time": datetime.fromtimestamp(ts).isoformat(),
                "market": market,
                "up_ask": round(up_ask, 6),
                "down_ask": round(down_ask, 6),
                "pair_cost": round(pair_cost, 6),
                "has_opportunity": has_opportunity,
                "time_remaining_sec": time_remaining_sec,
            }
            row = _json_dumps(data) + "\n"
        else:
            row = f"{ts},{market},{up_ask:.4f},{down_ask:.4f},{pair_cost:.4f},{1 if has_opportunity else 0},{time_remaining_sec}\n"

        self._scan_queue.put(row)
        self.scans_logged += 1

    def log_event(
        self,
        event_type: str,
        data: dict,
    ):
        """Log a generic event (non-blocking). Always JSONL."""
        ts = time.time()
        event = {
            "ts": ts,
            "time": datetime.fromtimestamp(ts).isoformat(),
            "event": event_type,
            **data
        }
        row = _json_dumps(event) + "\n"
        self._trade_queue.put(row)
        self.trades_logged += 1

    def _background_writer(self):
        """Background thread that flushes queues to disk."""
        trade_buffer = []
        scan_buffer = []
        last_flush = time.time()

        while self._running:
            try:
                # Drain trade queue
                while not self._trade_queue.empty():
                    try:
                        trade_buffer.append(self._trade_queue.get_nowait())
                    except queue.Empty:
                        break

                # Drain scan queue
                while not self._scan_queue.empty():
                    try:
                        scan_buffer.append(self._scan_queue.get_nowait())
                    except queue.Empty:
                        break

                # Check if we should flush
                now = time.time()
                should_flush = (
                    len(trade_buffer) >= self.buffer_size or
                    len(scan_buffer) >= self.buffer_size or
                    (now - last_flush) >= self.flush_interval
                )

                if should_flush and (trade_buffer or scan_buffer):
                    # Flush trades
                    if trade_buffer:
                        with open(self.trades_file, 'a') as f:
                            f.writelines(trade_buffer)
                        trade_buffer.clear()

                    # Flush scans
                    if scan_buffer:
                        with open(self.scans_file, 'a') as f:
                            f.writelines(scan_buffer)
                        scan_buffer.clear()

                    last_flush = now

                # Sleep briefly to avoid busy-waiting
                time.sleep(0.1)

            except Exception:
                # Don't crash the bot if logging fails
                pass

    def flush(self):
        """Force flush all pending writes."""
        timeout = 2.0
        start = time.time()
        while (not self._trade_queue.empty() or not self._scan_queue.empty()) and (time.time() - start) < timeout:
            time.sleep(0.05)

    def stop(self):
        """Stop the background writer."""
        self.flush()
        self._running = False
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=1.0)

    def get_stats(self) -> dict:
        """Get logging statistics."""
        return {
            "trades_logged": self.trades_logged,
            "scans_logged": self.scans_logged,
            "trades_file": str(self.trades_file),
            "scans_file": str(self.scans_file),
            "format": self.format,
            "fast_json": _FAST_JSON,
        }


# Standalone JSONL writer for simple use cases
class JSONLWriter:
    """
    Simple JSONL file writer with buffering.

    Usage:
        writer = JSONLWriter("data.jsonl")
        writer.write({"event": "trade", "price": 0.48})
        writer.close()
    """

    def __init__(self, filepath: str, buffer_size: int = 100):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self._buffer = []
        self._file = open(self.filepath, 'a')

    def write(self, data: dict):
        """Write a single record."""
        line = _json_dumps(data) + "\n"
        self._buffer.append(line)

        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Flush buffer to disk."""
        if self._buffer:
            self._file.writelines(self._buffer)
            self._file.flush()
            self._buffer.clear()

    def close(self):
        """Close the writer."""
        self.flush()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    # Test the logger
    print(f"Fast JSON enabled: {_FAST_JSON}")

    # Test JSONL format
    logger = FastTradeLogger(log_dir="test_logs", format="jsonl")
    logger.log_trade(
        market="btc-15m-test",
        price_up=0.48,
        price_down=0.50,
        pair_cost=0.98,
        profit_pct=2.04,
        order_size=5,
        investment=4.90,
        expected_profit=0.10,
        balance_after=95.10,
        ltm_bucket=2,
    )
    logger.log_scan(
        market="btc-15m-test",
        up_ask=0.49,
        down_ask=0.51,
        pair_cost=1.00,
        has_opportunity=False,
        time_remaining_sec=300,
    )
    logger.stop()

    print(f"Stats: {logger.get_stats()}")

    # Show sample output
    with open(logger.trades_file) as f:
        print(f"\nTrades JSONL:\n{f.read()}")

    with open(logger.scans_file) as f:
        print(f"Scans JSONL:\n{f.read()}")
