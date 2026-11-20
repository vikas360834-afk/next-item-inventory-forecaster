# Next-Item Inventory Forecaster

An interactive Streamlit application for forecasting the next items customers are likely to buy, using sequential transaction patterns. Includes a lightweight n-gram baseline and a genuine small PyTorch causal Transformer, inventory planning, sales analytics, association insights, CSV import, reports, and model evaluation.

## Features
- Import transaction histories from CSV and explore sample retail transactions.
- Forecast likely next SKUs with the n-gram baseline or trained Transformer.
- Train the Transformer from SKU sequences, inspect training loss, and save/load checkpoints.
- Compare model performance on a holdout split.
- Inventory replenishment planning, sales analytics, item associations, and reports.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

## Tests
```bash
pytest -q
```

## Transformer details and limitations
The Transformer is a small causal sequence model implemented with PyTorch. It learns token and positional embeddings, applies causal self-attention, and predicts the next SKU token. It is intended as an educational/local retail prototype; quality depends on the size and representativeness of transaction history. It is not a production replenishment system and should be evaluated against real store data before operational use.

## Project structure
- `app/main.py` – Streamlit interface
- `app/core.py` – n-gram forecasting and core logic
- `app/features.py` – analytics and inventory features
- `app/transformer.py` – PyTorch causal Transformer
- `app/storage.py` – persistence helpers
- `data/` – sample transactions and evaluation metadata
- `tests/` – automated tests

## License
See repository license settings before reuse.
