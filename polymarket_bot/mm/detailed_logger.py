"""
Detailed Logger for Market Maker Bot (Bot 2).

Provides rich, real-time logging when running in LIVE mode.
Shows every action the bot takes with timestamps and context.
"""
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Try orjson for fast JSON
try:
    import orjson
    def _json_dumps(data: dict) -> str:
        return orjson.dumps(data).decode('utf-8')
except ImportError:
    import json
    def _json_dumps(data: dict) -> str:
        return json.dumps(data, separators=(',', ':'))


@dataclass
class LogEvent:
    """Structured log event."""
    timestamp: float
    event_type: str
    market: str
    data: Dict[str, Any]
    level: str = "INFO"

    def to_dict(self) -> dict:
        return {
            "ts": self.timestamp,
            "time": datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S.%f")[:-3],
            "event": self.event_type,
            "market": self.market,
            "level": self.level,
            **self.data
        }

    def to_jsonl(self) -> str:
        return _json_dumps(self.to_dict())


class DetailedBotLogger:
    """
    Detailed logger for Market Maker Bot.

    Features:
    - Console output with colors and formatting
    - JSONL file for analysis
    - Event categorization
    - Performance metrics
    - Error tracking

    Event types:
    - STARTUP: Bot initialization
    - MARKET_DISCOVERED: New market found
    - ORDERBOOK_UPDATE: Orderbook data received
    - QUOTE_CALCULATED: New quotes calculated
    - ORDER_SENT: Order submitted to exchange
    - ORDER_FILLED: Order executed
    - ORDER_CANCELLED: Order cancelled
    - INVENTORY_UPDATE: Position changed
    - REBALANCE: Inventory rebalancing
    - ERROR: Error occurred
    - SHUTDOWN: Bot stopping
    """

    # ANSI colors for console
    COLORS = {
        "RESET": "\033[0m",
        "BOLD": "\033[1m",
        "RED": "\033[91m",
        "GREEN": "\033[92m",
        "YELLOW": "\033[93m",
        "BLUE": "\033[94m",
        "MAGENTA": "\033[95m",
        "CYAN": "\033[96m",
        "WHITE": "\033[97m",
        "GRAY": "\033[90m",
    }

    EVENT_COLORS = {
        "STARTUP": "CYAN",
        "MARKET_DISCOVERED": "BLUE",
        "ORDERBOOK_UPDATE": "GRAY",
        "QUOTE_CALCULATED": "WHITE",
        "ORDER_SENT": "YELLOW",
        "ORDER_FILLED": "GREEN",
        "ORDER_CANCELLED": "MAGENTA",
        "INVENTORY_UPDATE": "CYAN",
        "REBALANCE": "YELLOW",
        "ERROR": "RED",
        "SHUTDOWN": "CYAN",
    }

    def __init__(
        self,
        log_dir: str = "logs",
        console_output: bool = True,
        file_output: bool = True,
        verbose: bool = True,
        use_colors: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.console_output = console_output
        self.file_output = file_output
        self.verbose = verbose
        self.use_colors = use_colors

        # Create log file
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"mm_detailed_{ts}.jsonl"
        self._file = None
        if file_output:
            self._file = open(self.log_file, 'a')

        # Stats
        self.event_counts: Dict[str, int] = {}
        self.start_time = time.time()

    def _colorize(self, text: str, color: str) -> str:
        """Add color to text if colors enabled."""
        if not self.use_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['RESET']}"

    def _format_console(self, event: LogEvent) -> str:
        """Format event for console output."""
        time_str = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S.%f")[:-3]
        color = self.EVENT_COLORS.get(event.event_type, "WHITE")

        # Event type with fixed width
        event_str = self._colorize(f"[{event.event_type:<18}]", color)

        # Market with fixed width
        market_str = self._colorize(f"{event.market:<12}", "CYAN") if event.market else " " * 12

        # Build detail string based on event type
        detail = self._format_detail(event)

        return f"{self._colorize(time_str, 'GRAY')} {event_str} {market_str} {detail}"

    def _format_detail(self, event: LogEvent) -> str:
        """Format event details based on type."""
        d = event.data

        if event.event_type == "STARTUP":
            return self._colorize(f"Bot starting - Mode: {d.get('mode', '?')} - Assets: {d.get('assets', '?')}", "BOLD")

        elif event.event_type == "MARKET_DISCOVERED":
            return f"Found: {d.get('slug', '?')} - Ends in {d.get('time_remaining', '?')}"

        elif event.event_type == "ORDERBOOK_UPDATE":
            bid = d.get('best_bid', 0)
            ask = d.get('best_ask', 0)
            spread = (ask - bid) * 100 if bid and ask else 0
            return f"Bid: ${bid:.4f} | Ask: ${ask:.4f} | Spread: {spread:.2f}%"

        elif event.event_type == "QUOTE_CALCULATED":
            return (
                f"BID ${d.get('bid_price', 0):.4f} x {d.get('bid_size', 0):.0f} | "
                f"ASK ${d.get('ask_price', 0):.4f} x {d.get('ask_size', 0):.0f} | "
                f"Spread: {d.get('spread_pct', 0):.2f}%"
            )

        elif event.event_type == "ORDER_SENT":
            side = d.get('side', '?')
            side_color = "GREEN" if side == "BUY" else "RED"
            return (
                f"{self._colorize(side, side_color)} "
                f"${d.get('price', 0):.4f} x {d.get('size', 0):.0f} "
                f"({d.get('order_type', 'GTC')}) → ID: {d.get('order_id', '?')[:8]}..."
            )

        elif event.event_type == "ORDER_FILLED":
            side = d.get('side', '?')
            side_color = "GREEN" if side == "BUY" else "RED"
            return self._colorize(
                f"★ FILL {side} ${d.get('price', 0):.4f} x {d.get('filled_size', 0):.0f} "
                f"(Total: ${d.get('fill_value', 0):.2f})",
                "BOLD"
            )

        elif event.event_type == "ORDER_CANCELLED":
            return f"Cancelled: {d.get('order_id', '?')[:8]}... ({d.get('reason', 'user')})"

        elif event.event_type == "INVENTORY_UPDATE":
            yes_exp = d.get('yes_exposure', 0)
            no_exp = d.get('no_exposure', 0)
            imbalance = d.get('imbalance', 0)
            imb_color = "GREEN" if abs(imbalance) < 0.2 else "YELLOW" if abs(imbalance) < 0.4 else "RED"
            return (
                f"YES: ${yes_exp:.2f} | NO: ${no_exp:.2f} | "
                f"Net: ${d.get('net_exposure', 0):+.2f} | "
                f"Imbalance: {self._colorize(f'{imbalance*100:+.1f}%', imb_color)}"
            )

        elif event.event_type == "REBALANCE":
            return (
                f"Adjusting sizes - YES mult: {d.get('yes_mult', 1):.2f} | "
                f"NO mult: {d.get('no_mult', 1):.2f} ({d.get('reason', '')})"
            )

        elif event.event_type == "ERROR":
            return self._colorize(f"ERROR: {d.get('error', '?')} - {d.get('details', '')}", "RED")

        elif event.event_type == "SHUTDOWN":
            return self._colorize(
                f"Bot stopped - Quotes: {d.get('total_quotes', 0)} | "
                f"Fills: {d.get('total_fills', 0)} | "
                f"PnL: ${d.get('pnl', 0):+.2f}",
                "BOLD"
            )

        else:
            # Generic format
            return str(d)

    def log(
        self,
        event_type: str,
        market: str = "",
        level: str = "INFO",
        **data
    ):
        """
        Log an event.

        Args:
            event_type: Type of event (STARTUP, ORDER_SENT, etc.)
            market: Market identifier
            level: Log level (INFO, WARNING, ERROR)
            **data: Event-specific data
        """
        event = LogEvent(
            timestamp=time.time(),
            event_type=event_type,
            market=market,
            data=data,
            level=level,
        )

        # Track counts
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1

        # Console output
        if self.console_output:
            # Skip verbose events if not verbose mode
            if not self.verbose and event_type in ["ORDERBOOK_UPDATE"]:
                pass
            else:
                print(self._format_console(event))

        # File output
        if self.file_output and self._file:
            self._file.write(event.to_jsonl() + "\n")
            self._file.flush()

    # Convenience methods for common events
    def startup(self, mode: str, assets: list, settings: dict = None):
        """Log bot startup."""
        self.log("STARTUP", mode=mode, assets=", ".join(assets), settings=settings or {})

    def market_discovered(self, market: str, slug: str, time_remaining: str, yes_token: str, no_token: str):
        """Log market discovery."""
        self.log("MARKET_DISCOVERED", market, slug=slug, time_remaining=time_remaining,
                 yes_token=yes_token[:16] + "...", no_token=no_token[:16] + "...")

    def orderbook_update(self, market: str, best_bid: float, best_ask: float, bid_depth: int = 0, ask_depth: int = 0):
        """Log orderbook update."""
        self.log("ORDERBOOK_UPDATE", market, best_bid=best_bid, best_ask=best_ask,
                 bid_depth=bid_depth, ask_depth=ask_depth)

    def quote_calculated(self, market: str, bid_price: float, bid_size: float, ask_price: float, ask_size: float):
        """Log calculated quote."""
        spread_pct = (ask_price - bid_price) * 100
        self.log("QUOTE_CALCULATED", market, bid_price=bid_price, bid_size=bid_size,
                 ask_price=ask_price, ask_size=ask_size, spread_pct=spread_pct)

    def order_sent(self, market: str, side: str, price: float, size: float, order_type: str, order_id: str):
        """Log order submission."""
        self.log("ORDER_SENT", market, side=side, price=price, size=size,
                 order_type=order_type, order_id=order_id)

    def order_filled(self, market: str, side: str, price: float, filled_size: float, order_id: str):
        """Log order fill."""
        fill_value = price * filled_size
        self.log("ORDER_FILLED", market, side=side, price=price, filled_size=filled_size,
                 fill_value=fill_value, order_id=order_id, level="INFO")

    def order_cancelled(self, market: str, order_id: str, reason: str = "user"):
        """Log order cancellation."""
        self.log("ORDER_CANCELLED", market, order_id=order_id, reason=reason)

    def inventory_update(self, market: str, yes_exposure: float, no_exposure: float, yes_shares: float, no_shares: float):
        """Log inventory change."""
        net_exposure = yes_exposure - no_exposure
        total = yes_exposure + no_exposure
        imbalance = net_exposure / total if total > 0 else 0
        self.log("INVENTORY_UPDATE", market, yes_exposure=yes_exposure, no_exposure=no_exposure,
                 yes_shares=yes_shares, no_shares=no_shares, net_exposure=net_exposure, imbalance=imbalance)

    def rebalance(self, market: str, yes_mult: float, no_mult: float, reason: str):
        """Log rebalancing action."""
        self.log("REBALANCE", market, yes_mult=yes_mult, no_mult=no_mult, reason=reason)

    def error(self, market: str, error: str, details: str = ""):
        """Log error."""
        self.log("ERROR", market, error=error, details=details, level="ERROR")

    def shutdown(self, total_quotes: int, total_fills: int, pnl: float):
        """Log shutdown."""
        runtime = time.time() - self.start_time
        self.log("SHUTDOWN", total_quotes=total_quotes, total_fills=total_fills,
                 pnl=pnl, runtime_seconds=runtime, event_counts=self.event_counts)

    def print_separator(self, title: str = ""):
        """Print a visual separator."""
        if self.console_output:
            line = "=" * 70
            if title:
                print(f"\n{self._colorize(line, 'CYAN')}")
                print(self._colorize(f"  {title}", 'BOLD'))
                print(f"{self._colorize(line, 'CYAN')}")
            else:
                print(self._colorize(line, 'GRAY'))

    def print_status_table(self, markets: dict):
        """Print a status table for all markets."""
        if not self.console_output:
            return

        print()
        print(self._colorize("┌" + "─" * 68 + "┐", "CYAN"))
        print(self._colorize("│", "CYAN") + self._colorize("  MARKET STATUS", "BOLD").center(77) + self._colorize("│", "CYAN"))
        print(self._colorize("├" + "─" * 68 + "┤", "CYAN"))

        header = f"│ {'Market':<10} │ {'Bid':>8} │ {'Ask':>8} │ {'YES $':>8} │ {'NO $':>8} │ {'Imb':>7} │"
        print(self._colorize(header, "CYAN"))
        print(self._colorize("├" + "─" * 68 + "┤", "CYAN"))

        for market_id, data in markets.items():
            imb = data.get('imbalance', 0)
            imb_str = f"{imb*100:+.1f}%"
            if abs(imb) < 0.2:
                imb_colored = self._colorize(imb_str, "GREEN")
            elif abs(imb) < 0.4:
                imb_colored = self._colorize(imb_str, "YELLOW")
            else:
                imb_colored = self._colorize(imb_str, "RED")

            row = (
                f"│ {market_id[:10]:<10} │ "
                f"${data.get('best_bid', 0):>6.4f} │ "
                f"${data.get('best_ask', 0):>6.4f} │ "
                f"${data.get('yes_exposure', 0):>7.2f} │ "
                f"${data.get('no_exposure', 0):>7.2f} │ "
            )
            print(self._colorize(row, "WHITE") + f"{imb_colored:>16} " + self._colorize("│", "CYAN"))

        print(self._colorize("└" + "─" * 68 + "┘", "CYAN"))
        print()

    def close(self):
        """Close the logger."""
        if self._file:
            self._file.close()

    def get_log_file(self) -> str:
        """Return path to log file."""
        return str(self.log_file)


# Test
if __name__ == "__main__":
    logger = DetailedBotLogger(verbose=True)

    logger.print_separator("MARKET MAKER BOT - LIVE MODE")

    logger.startup("LIVE", ["btc", "eth", "sol"], {"spread": 0.02, "max_position": 100})

    logger.market_discovered("btc", "btc-updown-15m-1737340800", "12m 30s",
                            "0x123abc...", "0x456def...")

    logger.orderbook_update("btc", 0.4823, 0.5134, 15, 12)

    logger.quote_calculated("btc", 0.4850, 10, 0.5100, 10)

    logger.order_sent("btc", "BUY", 0.4850, 10, "GTC", "order-123-abc-456")
    logger.order_sent("btc", "SELL", 0.5100, 10, "GTC", "order-789-def-012")

    logger.order_filled("btc", "BUY", 0.4850, 10, "order-123-abc-456")

    logger.inventory_update("btc", 4.85, 0, 10, 0)

    logger.rebalance("btc", 0.5, 1.5, "too much YES (+48%)")

    logger.error("eth", "Connection timeout", "Retrying in 5s...")

    logger.print_status_table({
        "btc": {"best_bid": 0.48, "best_ask": 0.52, "yes_exposure": 48.50, "no_exposure": 25.00, "imbalance": 0.32},
        "eth": {"best_bid": 0.45, "best_ask": 0.55, "yes_exposure": 22.50, "no_exposure": 27.50, "imbalance": -0.10},
        "sol": {"best_bid": 0.50, "best_ask": 0.50, "yes_exposure": 0, "no_exposure": 0, "imbalance": 0},
    })

    logger.shutdown(total_quotes=24, total_fills=3, pnl=1.25)

    logger.close()
    print(f"\nLog file: {logger.get_log_file()}")
