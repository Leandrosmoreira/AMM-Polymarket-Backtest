"""
Enhanced logging and console output module.

Provides rich console output with colors, progress indicators, and better formatting.
NÍVEL 1: DEBUG global configurável via env.
"""

import os
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

# Try to import rich for better console output, fallback to basic logging
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    RichHandler = None
    Progress = None
    Table = None
    Panel = None
    Text = None
    box = None

# Configuração via environment variables
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "logs/bot.log")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "10485760"))  # 10MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))


def setup_logging(verbose: bool = False, use_rich: bool = True) -> logging.Logger:
    """
    Set up logging with optional rich formatting.
    NÍVEL 1: DEBUG global configurável via env.
    
    Args:
        verbose: Enable verbose (DEBUG) logging (deprecated, use LOG_LEVEL env)
        use_rich: Use rich console formatting if available
        
    Returns:
        Configured logger instance
    """
    # Determinar nível de log
    if verbose or LOG_LEVEL == "DEBUG":
        level = logging.DEBUG
    else:
        level = getattr(logging, LOG_LEVEL, logging.INFO)
    
    # Garantir que diretório de logs existe
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Formato detalhado para arquivo
    detailed_format = (
        '%(asctime)s | %(levelname)-8s | %(name)s | '
        '%(funcName)s:%(lineno)d | %(message)s'
    )
    
    # Handler para arquivo (sempre DEBUG para capturar tudo)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(detailed_format))
    
    # Handler para console
    console_level = level
    if use_rich and RICH_AVAILABLE:
        console = Console(stderr=True)
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
        console_handler.setLevel(console_level)
        
        # Configurar root logger com ambos handlers
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Root sempre DEBUG para capturar tudo
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    else:
        # Formato simples para console
        console_format = '%(asctime)s | %(levelname)-8s | %(message)s'
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(logging.Formatter(console_format))
        
        # Configurar root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    # Suppress noisy HTTP logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def get_console() -> Optional[object]:
    """Get rich Console instance if available, None otherwise."""
    if RICH_AVAILABLE:
        return Console()
    return None


def print_success(message: str):
    """Print a success message with green color."""
    console = get_console()
    if console:
        console.print(f"[bold green]✓[/bold green] {message}")
    else:
        print(f"✓ {message}")


def print_error(message: str):
    """Print an error message with red color."""
    console = get_console()
    if console:
        console.print(f"[bold red]✗[/bold red] {message}")
    else:
        print(f"✗ {message}")


def print_warning(message: str):
    """Print a warning message with yellow color."""
    console = get_console()
    if console:
        console.print(f"[bold yellow]⚠[/bold yellow] {message}")
    else:
        print(f"⚠ {message}")


def print_info(message: str):
    """Print an info message."""
    console = get_console()
    if console:
        console.print(f"[bold blue]ℹ[/bold blue] {message}")
    else:
        print(f"ℹ {message}")


def print_header(message: str):
    """Print a header message."""
    console = get_console()
    if console:
        console.print(Panel(message, border_style="blue", title="BTC Arbitrage Bot"))
    else:
        print("=" * 70)
        print(message)
        print("=" * 70)


def create_stats_table(stats_data: dict) -> Optional[Table]:
    """Create a rich table for displaying statistics."""
    if not RICH_AVAILABLE:
        return None
    
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green", justify="right")
    
    for key, value in stats_data.items():
        # Format the key nicely
        formatted_key = key.replace("_", " ").title()
        table.add_row(formatted_key, str(value))
    
    return table


def print_stats_table(stats_data: dict):
    """Print statistics in a formatted table."""
    console = get_console()
    if console:
        table = create_stats_table(stats_data)
        if table:
            console.print(table)
            return
    
    # Fallback to simple formatting
    print("\n" + "=" * 50)
    for key, value in stats_data.items():
        formatted_key = key.replace("_", " ").title()
        print(f"{formatted_key:30s}: {value}")
    print("=" * 50)

