# Trading Alpha Visualizer

Interactive Streamlit app for discovering and ranking alpha candidates from trading simulation logs.

## What it supports

- CSV ingestion for the schema:
  `day,timestamp,product,bid/ask_price_1..3,bid/ask_volume_1..3,mid_price,profit_and_loss`
- Feature families:
  - Trend (SMA/EMA crossovers, momentum, z-score)
  - Volatility (rolling return volatility)
  - Order book and liquidity (L1/L1-L3 imbalance, spread, depth, microprice edge)
  - Microstructure proxies (book slope, quote update intensity)
- Predictive scoring:
  - Configurable multi-horizon forward returns
  - Information coefficient (correlation)
  - Hit rate
  - Quantile/bucket forward-return diagnostics

## Quickstart

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
streamlit run app.py
```

Upload `data/sample_log.csv` to test quickly.

## Project structure

- `app.py`: Streamlit app entrypoint and dashboard tabs.
- `src/data_loader.py`: schema validation and parsing.
- `src/features.py`: alpha feature engineering.
- `src/scoring.py`: forward targets and signal ranking metrics.
- `src/plots.py`: reusable Plotly chart wrappers.
- `tests/test_pipeline.py`: parser-feature-scoring pipeline test.
