"""
Fast CSV logger for backtest data.
Optimized for minimal performance impact - uses buffered writes.
"""
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import threading
import queue


class FastTradeLogger:
    """
    High-performance CSV logger for trade/opportunity data.

    Features:
    - Buffered writes (flushes every N records or M seconds)
    - Non-blocking (uses background thread)
    - Minimal memory footprint
    - CSV format for easy analysis
    """

    def __init__(
        self,
        log_dir: str = "logs",
        buffer_size: int = 10,
        flush_interval: float = 5.0,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        # Create log files with timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trades_file = self.log_dir / f"trades_{ts}.csv"
        self.scans_file = self.log_dir / f"scans_{ts}.csv"

        # Queues for non-blocking writes
        self._trade_queue: queue.Queue = queue.Queue()
        self._scan_queue: queue.Queue = queue.Queue()

        # Write headers
        self._write_headers()

        # Start background writer thread
        self._running = True
        self._writer_thread = threading.Thread(target=self._background_writer, daemon=True)
        self._writer_thread.start()

        # Stats
        self.trades_logged = 0
        self.scans_logged = 0

    def _write_headers(self):
        """Write CSV headers."""
        # Trades file header
        with open(self.trades_file, 'w') as f:
            f.write("timestamp,market,price_up,price_down,pair_cost,profit_pct,order_size,investment,expected_profit,balance_after,ltm_bucket\n")

        # Scans file header (lightweight - just opportunities)
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
        row = f"{ts},{market},{up_ask:.4f},{down_ask:.4f},{pair_cost:.4f},{1 if has_opportunity else 0},{time_remaining_sec}\n"
        self._scan_queue.put(row)
        self.scans_logged += 1

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

            except Exception as e:
                # Don't crash the bot if logging fails
                pass

    def flush(self):
        """Force flush all pending writes."""
        # Wait for queues to drain
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
        }
