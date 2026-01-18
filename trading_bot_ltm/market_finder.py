"""
Market Finder - Busca mercados ativos de BTC, ETH e SOL 15min.

Suporta múltiplos mercados para operação simultânea.
"""

import re
import logging
from datetime import datetime
from typing import List, Optional, Dict
import httpx

logger = logging.getLogger(__name__)


def find_current_btc_15min_market() -> Optional[str]:
    """
    Find the current active BTC 15min market on Polymarket.
    
    Returns:
        Market slug (e.g., 'btc-updown-15m-1234567890') or None
    """
    return _find_market_by_pattern("btc-updown-15m", "BTC")


def find_current_eth_15min_market() -> Optional[str]:
    """
    Find the current active ETH 15min market on Polymarket.
    
    Returns:
        Market slug (e.g., 'eth-updown-15m-1234567890') or None
    """
    return _find_market_by_pattern("eth-updown-15m", "ETH")


def find_current_sol_15min_market() -> Optional[str]:
    """
    Find the current active SOL 15min market on Polymarket.
    
    Returns:
        Market slug (e.g., 'sol-updown-15m-1234567890') or None
    """
    return _find_market_by_pattern("sol-updown-15m", "SOL")


def _find_market_by_pattern(pattern_prefix: str, asset_name: str) -> Optional[str]:
    """
    Generic function to find markets by pattern.
    
    Args:
        pattern_prefix: Prefix do padrão (ex: 'btc-updown-15m')
        asset_name: Nome do ativo para logging (ex: 'BTC')
    
    Returns:
        Market slug ou None se não encontrar
    """
    logger.info(f"Searching for current {asset_name} 15min market...")
    
    try:
        # Search on Polymarket's crypto 15min page
        page_url = "https://polymarket.com/crypto/15M"
        resp = httpx.get(page_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
        
        # Find the market slug in the HTML
        pattern = rf'{pattern_prefix}-(\d+)'
        matches = re.findall(pattern, resp.text)
        
        if not matches:
            logger.warning(f"No active {asset_name} 15min market found")
            return None
        
        # Prefer the most recent timestamp that is still OPEN.
        # 15min markets close 900s after the timestamp in the slug.
        now_ts = int(datetime.now().timestamp())
        all_ts = sorted((int(ts) for ts in matches), reverse=True)
        open_ts = [ts for ts in all_ts if now_ts < (ts + 900)]
        chosen_ts = open_ts[0] if open_ts else all_ts[0]
        slug = f"{pattern_prefix}-{chosen_ts}"
        
        logger.info(f"✅ {asset_name} market found: {slug}")
        return slug
        
    except Exception as e:
        logger.error(f"Error searching for {asset_name} 15min market: {e}")
        return None


def find_all_active_markets() -> Dict[str, Optional[str]]:
    """
    Busca todos os mercados ativos (BTC, ETH, SOL).
    
    Returns:
        Dict com slugs dos mercados encontrados:
        {
            'btc': 'btc-updown-15m-1234567890' ou None,
            'eth': 'eth-updown-15m-1234567890' ou None,
            'sol': 'sol-updown-15m-1234567890' ou None
        }
    """
    return {
        'btc': find_current_btc_15min_market(),
        'eth': find_current_eth_15min_market(),
        'sol': find_current_sol_15min_market(),
    }


def get_market_info(slug: str) -> Dict[str, str]:
    """
    Obtém informações de um mercado (market_id, token_ids, etc).
    
    Args:
        slug: Market slug (ex: 'btc-updown-15m-1234567890')
    
    Returns:
        Dict com informações do mercado
    """
    from .lookup import fetch_market_from_slug
    return fetch_market_from_slug(slug)

