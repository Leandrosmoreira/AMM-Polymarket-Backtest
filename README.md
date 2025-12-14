# Polymarket YES/NO Backtest

Sistema de backtesting para análise de oportunidades de arbitragem em mercados binários (YES/NO) da Polymarket.

## 📋 Descrição

Este projeto implementa um pipeline completo de análise de dados históricos de mercados da Polymarket, focando em:
- Seleção e categorização de mercados
- Coleta de histórico de preços
- Identificação de oportunidades de arbitragem
- Análise estatística de spreads
- Modelagem de custos e riscos

## 🚀 Instalação

### Pré-requisitos
- Python 3.10 ou superior
- pip (gerenciador de pacotes Python)

### Passos

1. **Criar e ativar ambiente virtual** (se ainda não fez):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

2. **Instalar dependências**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Estrutura do Projeto

```
polymarket_yesno_backtest/
├── config/              # Configurações do projeto
│   └── settings.py     # Parâmetros globais
├── core/               # Módulos principais
│   ├── api_client.py   # Cliente da API Polymarket
│   ├── models.py       # Modelos de dados
│   └── utils_time.py   # Utilitários de tempo
├── pipeline/           # Pipeline de processamento
│   ├── phase1_market_selection.py
│   ├── phase2_price_history.py
│   ├── phase3_arbitrage_series.py
│   ├── phase4_stats_spread.py
│   ├── phase5_market_comparison.py
│   ├── phase6_temporal_analysis.py
│   ├── phase7_cost_model.py
│   ├── phase8_edge_validation.py
│   └── phase9_risk_framework.py
├── notebooks/          # Jupyter notebooks exploratórios
├── data/              # Dados (criado automaticamente)
│   ├── raw/           # Dados brutos
│   ├── processed/     # Dados processados
│   └── stats/         # Estatísticas
├── main.py            # Script principal
└── requirements.txt   # Dependências
```

## 🎯 Uso

### Executar uma fase específica:
```bash
python main.py --phase 1        # Fase 1: Seleção de mercados
python main.py --phase 2        # Fase 2: Coleta de preços
python main.py --phase 3        # Fase 3: Séries de arbitragem
python main.py --phase 4        # Fase 4: Estatísticas de spread
python main.py --phase 5        # Fase 5: Comparação de mercados
python main.py --phase 6        # Fase 6: Análise temporal
python main.py --phase 7        # Fase 7: Modelo de custos
python main.py --phase 8        # Fase 8: Validação de edge
python main.py --phase 9        # Fase 9: Framework de risco
```

### Executar todas as fases:
```bash
python main.py --phase all
```

### Ver ajuda:
```bash
python main.py --help
```

## 📈 Fases do Pipeline

1. **Fase 1 - Seleção de Mercados**: Filtra e categoriza mercados por volume
2. **Fase 2 - Coleta de Preços**: Obtém histórico de preços dos mercados selecionados
3. **Fase 3 - Séries de Arbitragem**: Identifica oportunidades de arbitragem
4. **Fase 4 - Estatísticas de Spread**: Calcula estatísticas descritivas
5. **Fase 5 - Comparação de Mercados**: Compara performance entre mercados
6. **Fase 6 - Análise Temporal**: Analisa padrões temporais
7. **Fase 7 - Modelo de Custos**: Calcula custos de trading
8. **Fase 8 - Validação de Edge**: Valida vantagens competitivas
9. **Fase 9 - Framework de Risco**: Análise de risco e gestão de capital

## ⚙️ Configuração

As configurações principais estão em `config/settings.py`:

- **Volume mínimo**: `MIN_VOLUME_USD = 50_000`
- **Duração mínima**: `MIN_LIFETIME_DAYS = 7`
- **Timeframes**: `["1m", "5m", "15m", "1h", "4h", "1d"]`
- **Threshold de arbitragem**: `ARBITRAGE_THRESHOLD = 0.98`

## 📝 Notebooks

Os notebooks em `notebooks/` fornecem análises exploratórias:
- `exploration_phase1.ipynb`: Exploração da Fase 1
- `exploration_phase2_3_4.ipynb`: Exploração das Fases 2, 3 e 4

## 🔧 Desenvolvimento

### Formatação de código:
```bash
black .
isort .
```

### Verificação de tipos:
```bash
mypy .
```

### Testes:
```bash
pytest
```

## 📄 Licença

Este projeto faz parte da formação blockchain da DIO.


