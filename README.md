# Next-Item Inventory Forecaster

A Streamlit application that recommends likely next purchases from an incomplete shopping basket. It includes SKU sequence forecasting, a PyTorch causal Transformer, inventory planning, sales analytics, basket associations, CSV import, and model evaluation.

## Windows setup (PowerShell)

Run each command separately and wait for it to finish before entering the next one. Do not paste multiple commands onto one line.

```powershell
git clone https://github.com/vikas360834-afk/next-item-inventory-forecaster.git
cd next-item-inventory-forecaster
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip show streamlit
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

If the `py -3.12` command is unavailable, use `py -3` instead. The commands above deliberately use the virtual environment's Python executable directly, so activation is not required.

When Streamlit starts successfully, the terminal prints a Local URL. Open http://localhost:8501 in your browser and keep the terminal window open.

### Troubleshooting

- `No module named streamlit`: dependency installation did not complete. Rerun the pip install command above and wait for it to finish.
- `python.exe: No module named streamlit`: run Streamlit through `.venv\Scripts\python.exe -m streamlit`, not a different global Python installation.
- `Address already in use`: stop the old Streamlit process with Ctrl+C or start with `--server.port 8502`.
- If pip installation reports an error, copy the final 20 lines of the terminal output before trying another command.

## Features

- Partial basket input with searchable product selection.
- Top 3, 5, 7, or 10 next-item recommendations.
- Option to exclude products already in the basket.
- Causal Transformer training and local checkpoint persistence.
- N-gram baseline and popularity-baseline evaluation.
- Inventory reorder planner, sales charts, and frequently-bought-together analysis.
- CSV transaction import using `transaction_id` and `sku` columns.
- Downloadable recommendation and inventory reports.

## Milestones

1. Generate reproducible mock sales transactions as ordered SKU sequences.
2. Represent SKUs as tokens and encode transaction sequences.
3. Train a small causal Transformer with next-token cross-entropy.
4. Predict likely next SKUs from a partial basket.
5. Display ranked recommendations and relative scores in Streamlit.
6. Compare model performance with a simple popularity baseline using HitRate@K and MRR.

## Model and limitations

The Transformer is a compact model trained from scratch on SKU token sequences; it is not a fine-tuned pretrained GPT model. The bundled data is synthetic. Recommendation scores are relative ranking scores rather than calibrated purchase probabilities. Evaluate on anonymized, time-ordered point-of-sale data before making real replenishment decisions.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Reproducible evaluation results

A deterministic 1,200-basket synthetic dataset was split into 960 training and 240 holdout baskets. At K=5, the sequence model achieved **57.50% HitRate@5** and **0.4685 MRR**. The popularity-only baseline achieved **42.92% HitRate@5**. The measured result is also stored in `results/evaluation_results.json`; it is a synthetic demonstration and must not be interpreted as a real-store performance guarantee.

| Model | HitRate@5 |
|---|---:|
| SKU sequence n-gram | 57.50% |
| Popularity baseline | 42.92% |
