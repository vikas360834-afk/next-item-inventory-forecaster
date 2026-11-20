# Next-Item Inventory Forecaster

A Streamlit application that recommends likely next purchases from an incomplete shopping basket. It includes a SKU sequence n-gram model, a trainable PyTorch causal Transformer, inventory planning, sales analytics, basket associations, CSV import, and holdout evaluation.

## Run locally (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
streamlit run app/main.py
```

Open http://localhost:8501.

## Milestones
1. Generate reproducible mock shopping transactions represented as ordered SKU sequences.
2. Encode SKU identifiers as tokens and learn next-token patterns.
3. Continue domain training with a small causal Transformer using next-token cross-entropy.
4. Predict likely next SKUs from a partial basket.
5. Display ranked recommendations and relative scores in Streamlit.
6. Compare a sequence model with a popularity baseline using HitRate@K and MRR.

## Features
- Partial basket input with product search.
- Top 3, 5, 7, or 10 next-item recommendations.
- Option to exclude products already in the basket.
- Causal Transformer training and local checkpoint persistence.
- N-gram baseline for lightweight comparison.
- Inventory reorder planner, sales charts, and frequently-bought-together analysis.
- CSV transaction import using transaction_id and sku columns.

## Model and limitations
The Transformer is a compact model trained from scratch on SKU token sequences; it is not a fine-tuned pretrained GPT model. The bundled data is synthetic. Recommendation scores are ranking scores rather than calibrated purchase probabilities. Evaluate on anonymized, time-ordered point-of-sale data before making real replenishment decisions.

## Tests
Run `pytest -q` to execute automated checks.
