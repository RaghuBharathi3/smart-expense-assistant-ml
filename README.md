# Smart Expense Assistant

A natural-language expense assistant that classifies Indian expense descriptions, executes validated analytics tools, and renders interactive visualizations in Indian Rupees (`₹`).

## Architecture

`User → Interface (src/interface/app.py) → LLM Router (src/llm/router_llm.py) → Whitelisted Backend Router (src/backend/router.py) → Backend Tools (src/backend/tools.py) → LLM Response (src/llm/respond.py) → Interface`

## Project Structure

```text
smart-expense-assistant-ml/
├── data/
│   ├── expenses.csv                 # Seed dataset (read-only)
│   └── runtime/.gitkeep             # Runtime CSV writes (git-ignored)
├── models/classifier/
│   └── expense_classifier.pkl       # TF-IDF + Logistic Regression pipeline
├── notebooks/
│   ├── classifier_training.ipynb
│   ├── llm_experimentation.ipynb
│   └── backend_experimentation.ipynb
├── src/
│   ├── classifier/                  # train.py, predict.py, preprocessing.py
│   ├── llm/                         # load_model.py, prompts.py, router_llm.py, respond.py
│   ├── backend/                     # tools.py, router.py, validators.py
│   ├── interface/                   # app.py, charts.py
│   └── config.py
├── tests/                           # test_backend.py, test_queries.py
├── docs/function_specs.md
├── requirements.txt
└── README.md
```

## Setup & Run

```bash
pip install -r requirements.txt
python -m src.classifier.train
python -m pytest -v
python -m src.interface.app
```

## Data Privacy, Security & Licensing

- **Data Privacy (DPDP Act 2023 / GDPR):** `data/expenses.csv` contains synthetic expense descriptions only—no real names, emails, phone numbers, or account numbers. Runtime writes go strictly to `data/runtime/`.
- **Security:** No API keys or tokens are stored in the repo; configure optional `LLM_API_URL` and `LLM_API_KEY` via environment variables or Colab Secrets.
- **License & Attributions:** Licensed under the [MIT License](LICENSE). Built with `pandas`, `numpy`, `scikit-learn`, `joblib` (BSD-3-Clause), `streamlit`, `requests` (Apache-2.0), and `plotly`, `pytest` (MIT).
