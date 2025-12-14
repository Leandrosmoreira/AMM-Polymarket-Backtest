#!/usr/bin/env python3
"""
Polymarket YES/NO Backtest - Main Entry Point

Script principal para executar o pipeline de backtesting de mercados
binários da Polymarket.

Uso:
    python main.py --phase 1        # Executa Fase 1 (seleção de mercados)
    python main.py --phase 2        # Executa Fase 2 (coleta de preços)
    python main.py --phase 3        # Executa Fase 3 (séries de arbitragem)
    python main.py --phase 4        # Executa Fase 4 (estatísticas de spread)
    python main.py --phase 5        # Executa Fase 5 (comparação de mercados)
    python main.py --phase 6        # Executa Fase 6 (análise temporal)
    python main.py --phase 7        # Executa Fase 7 (modelo de custos)
    python main.py --phase 8        # Executa Fase 8 (validação de edge)
    python main.py --phase 9        # Executa Fase 9 (framework de risco)
    python main.py --phase all      # Executa todas as fases
"""

import argparse
import logging
import sys
from typing import Optional

from config.settings import (
    LOG_LEVEL,
    LOG_FORMAT,
    DEFAULT_TIMEFRAME,
    TIMEFRAMES,
    ensure_data_dirs_exist,
)


def setup_logging(level: str = LOG_LEVEL) -> None:
    """Configura o sistema de logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def run_phase1() -> dict:
    """Executa Fase 1: Seleção de Mercados."""
    from pipeline.phase1_market_selection import run_phase1_pipeline

    groups, summary = run_phase1_pipeline()
    return {"groups": groups, "summary": summary}


def run_phase2(timeframes: Optional[list] = None) -> dict:
    """Executa Fase 2: Coleta de Histórico de Preços."""
    from pipeline.phase2_price_history import run_phase2_pipeline

    if timeframes is None:
        timeframes = [DEFAULT_TIMEFRAME]

    results = run_phase2_pipeline(timeframes=timeframes)
    return {"results": results}


def run_phase3(timeframes: Optional[list] = None) -> dict:
    """Executa Fase 3: Reconstrução de Séries de Arbitragem."""
    from pipeline.phase3_arbitrage_series import run_phase3_pipeline

    if timeframes is None:
        timeframes = [DEFAULT_TIMEFRAME]

    results = run_phase3_pipeline(timeframes=timeframes)
    return {"results": results}


def run_phase4(timeframe: str = DEFAULT_TIMEFRAME) -> dict:
    """Executa Fase 4: Estatísticas Descritivas de Spread."""
    from pipeline.phase4_stats_spread import run_phase4_pipeline

    stats_list, df_summary = run_phase4_pipeline(timeframe=timeframe)
    return {
        "stats_list": stats_list,
        "summary": df_summary,
        "count": len(stats_list),
    }


def run_phase5(timeframe: str = DEFAULT_TIMEFRAME) -> dict:
    """Executa Fase 5: Comparação entre Mercados."""
    from pipeline.phase5_market_comparison import run_phase5_pipeline

    rankings, category_stats, duration_stats = run_phase5_pipeline(timeframe=timeframe)
    return {
        "rankings": rankings,
        "category_stats": category_stats,
        "duration_stats": duration_stats,
        "count": len(rankings),
    }


def run_phase6(timeframe: str = DEFAULT_TIMEFRAME) -> dict:
    """Executa Fase 6: Análise Temporal (Sazonalidade)."""
    from pipeline.phase6_temporal_analysis import run_phase6_pipeline

    hourly_agg, daily_agg, insights = run_phase6_pipeline(timeframe=timeframe)
    return {
        "hourly": hourly_agg,
        "daily": daily_agg,
        "insights": insights,
    }


def run_phase7(timeframe: str = DEFAULT_TIMEFRAME, trade_size: float = 100.0) -> dict:
    """Executa Fase 7: Modelo de Custos."""
    from pipeline.phase7_cost_model import run_phase7_pipeline

    analyses, summaries = run_phase7_pipeline(
        timeframe=timeframe,
        trade_size_usd=trade_size,
    )
    return {
        "analyses": analyses,
        "summaries": summaries,
    }


def run_phase8(timeframe: str = DEFAULT_TIMEFRAME, scenario: str = "median") -> dict:
    """Executa Fase 8: Validação de Edge."""
    from pipeline.phase8_edge_validation import run_phase8_pipeline

    result = run_phase8_pipeline(
        timeframe=timeframe,
        scenario_name=scenario,
    )
    return {
        "existence": result.existence,
        "capturability": result.capturability,
        "scalability": result.scalability,
        "overall_viable": result.overall_viable,
        "recommendation": result.recommendation,
    }


def run_phase9(
    timeframe: str = DEFAULT_TIMEFRAME,
    scenario: str = "median",
    capital: float = 10000.0,
) -> dict:
    """Executa Fase 9: Framework de Risco."""
    from pipeline.phase9_risk_framework import run_phase9_pipeline

    framework = run_phase9_pipeline(
        timeframe=timeframe,
        scenario_name=scenario,
        total_capital=capital,
    )
    return {
        "buffer": framework.buffer,
        "capital": framework.capital,
        "markets": framework.markets,
        "time": framework.time,
        "limits": framework.limits,
        "rules": framework.operational_rules,
        "summary": framework.executive_summary,
    }


def run_all_phases(timeframes: Optional[list] = None) -> dict:
    """Executa todas as fases do pipeline."""
    results = {}

    if timeframes is None:
        timeframes = [DEFAULT_TIMEFRAME]

    # Fase 1
    print("\n" + "=" * 60)
    print("FASE 1: SELEÇÃO DE MERCADOS")
    print("=" * 60)
    phase1_result = run_phase1()
    results["phase1"] = phase1_result

    # Fase 2
    print("\n" + "=" * 60)
    print("FASE 2: COLETA DE HISTÓRICO DE PREÇOS")
    print("=" * 60)
    phase2_result = run_phase2(timeframes)
    results["phase2"] = phase2_result

    # Fase 3
    print("\n" + "=" * 60)
    print("FASE 3: RECONSTRUÇÃO DE SÉRIES DE ARBITRAGEM")
    print("=" * 60)
    phase3_result = run_phase3(timeframes)
    results["phase3"] = phase3_result

    # Fase 4
    print("\n" + "=" * 60)
    print("FASE 4: ESTATÍSTICAS DESCRITIVAS DE SPREAD")
    print("=" * 60)
    phase4_result = run_phase4(timeframes[0])
    results["phase4"] = phase4_result

    # Fase 5
    print("\n" + "=" * 60)
    print("FASE 5: COMPARAÇÃO ENTRE MERCADOS")
    print("=" * 60)
    phase5_result = run_phase5(timeframes[0])
    results["phase5"] = phase5_result

    # Fase 6
    print("\n" + "=" * 60)
    print("FASE 6: ANÁLISE TEMPORAL (SAZONALIDADE)")
    print("=" * 60)
    phase6_result = run_phase6(timeframes[0])
    results["phase6"] = phase6_result

    # Fase 7
    print("\n" + "=" * 60)
    print("FASE 7: MODELO DE CUSTOS")
    print("=" * 60)
    phase7_result = run_phase7(timeframes[0])
    results["phase7"] = phase7_result

    # Fase 8
    print("\n" + "=" * 60)
    print("FASE 8: VALIDAÇÃO DE EDGE")
    print("=" * 60)
    phase8_result = run_phase8(timeframes[0])
    results["phase8"] = phase8_result

    # Fase 9
    print("\n" + "=" * 60)
    print("FASE 9: FRAMEWORK DE RISCO")
    print("=" * 60)
    phase9_result = run_phase9(timeframes[0])
    results["phase9"] = phase9_result

    return results


def print_summary(results: dict) -> None:
    """Imprime resumo dos resultados."""
    print("\n" + "=" * 60)
    print("RESUMO FINAL")
    print("=" * 60)

    if "phase1" in results:
        summary = results["phase1"].get("summary", {})
        print(f"\nFase 1 - Seleção de Mercados:")
        print(f"  Total bruto: {summary.get('total_raw', 'N/A')}")
        print(f"  Total filtrado: {summary.get('total_filtered', 'N/A')}")
        print(f"  Total selecionado: {summary.get('total_selected', 'N/A')}")

        groups = summary.get("groups", {})
        for group_name, group_info in groups.items():
            count = group_info.get("count", 0)
            print(f"  Grupo {group_name}: {count} mercados")

    if "phase2" in results:
        phase2_results = results["phase2"].get("results", {})
        print(f"\nFase 2 - Coleta de Preços:")
        for timeframe, market_results in phase2_results.items():
            print(f"  Timeframe {timeframe}: {len(market_results)} mercados")

    if "phase3" in results:
        phase3_results = results["phase3"].get("results", {})
        print(f"\nFase 3 - Séries de Arbitragem:")
        for timeframe, market_results in phase3_results.items():
            print(f"  Timeframe {timeframe}: {len(market_results)} mercados processados")

    if "phase4" in results:
        count = results["phase4"].get("count", 0)
        print(f"\nFase 4 - Estatísticas de Spread:")
        print(f"  Mercados analisados: {count}")

        stats_list = results["phase4"].get("stats_list", [])
        if stats_list:
            import numpy as np
            avg_positive = np.mean([s.frequency.positive_pct for s in stats_list])
            max_positive = max(s.frequency.positive_pct for s in stats_list)
            print(f"  Arbitragem média (>0): {avg_positive:.2f}%")
            print(f"  Arbitragem máxima (>0): {max_positive:.2f}%")

    if "phase5" in results:
        count = results["phase5"].get("count", 0)
        category_stats = results["phase5"].get("category_stats", {})
        print(f"\nFase 5 - Comparação entre Mercados:")
        print(f"  Mercados rankeados: {count}")
        print(f"  Categorias identificadas: {len(category_stats)}")

        if category_stats:
            best_cat = max(category_stats.items(),
                          key=lambda x: x[1].avg_composite_score)
            print(f"  Melhor categoria: {best_cat[0]} (score={best_cat[1].avg_composite_score:.4f})")

    if "phase6" in results:
        insights = results["phase6"].get("insights", {})
        print(f"\nFase 6 - Análise Temporal:")

        if "best_period" in insights:
            print(f"  Melhor período: {insights['best_period']} ({insights['best_period_pct']:.2f}%)")

        if "best_day" in insights:
            d = insights["best_day"]
            print(f"  Melhor dia: {d['day_name']} ({d['positive_pct']:.2f}%)")

        if "weekend_has_more_arb" in insights:
            if insights["weekend_has_more_arb"]:
                print("  Fim de semana tem MAIS arbitragem que dias úteis")
            else:
                print("  Dias úteis têm MAIS arbitragem que fim de semana")

    if "phase7" in results:
        summaries = results["phase7"].get("summaries", {})
        print(f"\nFase 7 - Modelo de Custos:")
        for scenario_name, summary in summaries.items():
            print(f"  Cenário {scenario_name}:")
            print(f"    Viabilidade: {summary.overall_viability_rate:.1f}%")
            print(f"    Lucro médio/trade: {summary.avg_net_profit_per_trade_pct:.4f}%")

    if "phase8" in results:
        print(f"\nFase 8 - Validação de Edge:")
        print(f"  Edge existe: {results['phase8'].get('overall_viable', False)}")
        print(f"  Recomendação: {results['phase8'].get('recommendation', 'N/A')[:80]}...")

    if "phase9" in results:
        print(f"\nFase 9 - Framework de Risco:")
        buffer = results["phase9"].get("buffer")
        capital = results["phase9"].get("capital")
        if buffer:
            print(f"  Buffer recomendado: {buffer.recommended_buffer*100:.2f}%")
        if capital:
            print(f"  Capital por trade: ${capital.recommended_trade_size_usd:.2f}")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Polymarket YES/NO Backtest Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Fases disponíveis:
  1    Seleção de mercados (busca API, filtra, agrupa A/B/C)
  2    Coleta de histórico de preços YES/NO
  3    Reconstrução de séries de arbitragem (spread = 1 - YES - NO)
  4    Estatísticas descritivas de spread (frequência, magnitude, duração)
  5    Comparação entre mercados (ranking, categorias, duração)
  6    Análise temporal (hora do dia, dia da semana, proximidade resolução)
  7    Modelo de custos (fees, gas, slippage, falhas de execução)
  8    Validação de edge (existência, capturabilidade, escalabilidade)
  9    Framework de risco (regras operacionais, limites, recomendações)
  all  Executa todas as fases em sequência
        """
    )

    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "all"],
        default="all",
        help="Fase do pipeline a executar (1-9 ou all)",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        choices=TIMEFRAMES,
        default=DEFAULT_TIMEFRAME,
        help=f"Timeframe para análise (default: {DEFAULT_TIMEFRAME})",
    )

    parser.add_argument(
        "--all-timeframes",
        action="store_true",
        help="Processa todos os timeframes disponíveis",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=LOG_LEVEL,
        help=f"Nível de logging (default: {LOG_LEVEL})",
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)
    ensure_data_dirs_exist()

    # Determina timeframes
    if args.all_timeframes:
        timeframes = TIMEFRAMES
    else:
        timeframes = [args.timeframe]

    # Executa fase(s)
    try:
        if args.phase == "1":
            results = {"phase1": run_phase1()}
        elif args.phase == "2":
            results = {"phase2": run_phase2(timeframes)}
        elif args.phase == "3":
            results = {"phase3": run_phase3(timeframes)}
        elif args.phase == "4":
            results = {"phase4": run_phase4(timeframes[0])}
        elif args.phase == "5":
            results = {"phase5": run_phase5(timeframes[0])}
        elif args.phase == "6":
            results = {"phase6": run_phase6(timeframes[0])}
        elif args.phase == "7":
            results = {"phase7": run_phase7(timeframes[0])}
        elif args.phase == "8":
            results = {"phase8": run_phase8(timeframes[0])}
        elif args.phase == "9":
            results = {"phase9": run_phase9(timeframes[0])}
        else:
            results = run_all_phases(timeframes)

        print_summary(results)
        return 0

    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário.")
        return 130
    except Exception as e:
        logging.error(f"Erro fatal: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
