# Smart Expense Assistant ML (Task 1)

A lightweight, production-ready Machine Learning pipeline that classifies natural-language Indian expense descriptions into four categories: **Food**, **Travel**, **Shopping**, and **Utilities**.

---

## Problem Statement

Personal finance apps and AI expense assistants receive messy, free-text expense descriptions (e.g., `"Swiggy food delivery"`, `"Chennai Metro ticket"`, `"BESCOM power bill online payment"`). Rule-based keyword matching breaks when merchants or phrasing vary. This module provides a fast, offline-capable TF-IDF + linear ML classifier with a clean Python API for downstream retrieval, LLM tool-calling, and chatbot integration.

## Architecture

```text
data/expenses.csv (1,200 Indian expense records)
        │
        ▼
src/preprocessing.py (clean_text, deduplication, 80:20 stratified split)
        │
        ▼
TfidfVectorizer (fit ONLY on training split, ngram_range=(1, 2))
        │
        ▼
src/train.py (MultinomialNB vs LogisticRegression vs LinearSVC)
        │
        ▼
models/expense_model.pkl + models/vectorizer.pkl
        │
        ▼
src/expense_classifier.py (categorize_expense, batch_categorize_expenses)
```

## Technology Stack

- **Language:** Python 3.10+
- **Data & EDA:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (TF-IDF, Multinomial Naive Bayes, Logistic Regression, Linear SVM)
- **Model Persistence:** Joblib
- **Testing:** Pytest
- **Notebook:** Google Colab

## Installation

```bash
git clone https://github.com/RaghuBharathi3/smart-expense-assistant-ml.git
cd smart-expense-assistant-ml
pip install -r requirements.txt
```

## Execution Instructions

1. **Generate dataset (optional, already committed at `data/expenses.csv`):**
   ```bash
   python -m src.generate_dataset
   ```
2. **Run Exploratory Data Analysis (EDA) and save plots:**
   ```bash
   python -m src.eda
   ```
3. **Train, compare models, and export best artifacts:**
   ```bash
   python -m src.train
   ```
4. **Run automated tests:**
   ```bash
   python -m pytest -v
   ```

## Dataset Description

- **File:** `data/expenses.csv`
- **Size:** 1,200 records (300 per category: `Food`, `Travel`, `Shopping`, `Utilities`)
- **Columns:** `expense_id`, `description`, `amount` (INR), `date`, `category`
- **Quality:** 0 missing values, 0 duplicate rows, 0 duplicate descriptions; includes Indian merchants (Swiggy, Zomato, Uber, Ola, Chennai Metro, Amazon, Flipkart, Myntra, TANGEDCO, BESCOM, Airtel, Jio, Indane) and cross-category phrasing (e.g., Amazon Fresh vs Amazon Pay vs Amazon shopping) to avoid trivial keyword shortcuts.

## Model Training & Actual Evaluation Results

All models were trained on 960 records and evaluated on a stratified held-out test set of 240 records (60 per category).

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 0.9792 | 0.9808 | 0.9792 | 0.9794 |
| Logistic Regression | 0.9875 | 0.9881 | 0.9875 | 0.9875 |
| **Linear SVM (`LinearSVC`)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

**Selected Model:** `LinearSVC` (Linear SVM), exported to `models/expense_model.pkl` alongside `models/vectorizer.pkl`.

## Prediction Examples

```python
from src.expense_classifier import categorize_expense, batch_categorize_expenses

print(categorize_expense("Uber ride to office"))
# {'description': 'Uber ride to office', 'category': 'Travel', 'confidence': 0.7258}

print(categorize_expense("Ordered pizza online"))
# {'description': 'Ordered pizza online', 'category': 'Food', 'confidence': 0.692}

print(categorize_expense("Purchased headphones"))
# {'description': 'Purchased headphones', 'category': 'Shopping', 'confidence': 0.6907}

print(categorize_expense("Monthly electricity bill"))
# {'description': 'Monthly electricity bill', 'category': 'Utilities', 'confidence': 0.723}
```

## Integration Instructions (For Tasks 2, 3 & 4)

Teammates can import and call the classifier directly without retraining:

```python
from src.expense_classifier import categorize_expense, batch_categorize_expenses

result = categorize_expense("Swiggy food delivery")
category = result["category"]      # "Food"
confidence = result["confidence"]  # float in (0, 1]
```

See [docs/integration_guide.md](docs/integration_guide.md) for full details.

## Limitations

1. **Synthetic Dataset:** Although realistic Indian merchants and overlapping phrases are included, real-world user text may contain typos, Hinglish/regional slang, or unseen merchant names.
2. **Confidence Scores:** `LinearSVC` is a margin-based classifier and does not natively output calibrated probabilities. `predict_with_confidence` applies softmax to `decision_function` margins, so confidence reflects relative margin strength rather than a true probability.
3. **Closed Category Set:** Every valid input is assigned to one of the four fixed classes (`Food`, `Travel`, `Shopping`, `Utilities`); there is no `Other` class yet.

---

## Beginner-Friendly Explanation (For Professor Demo)

> **How does our ML pipeline work?**
> 1. **Data Collection:** We created 1,200 realistic Indian expense descriptions (like `"Uber ride to office"` or `"Swiggy food delivery"`) across 4 balanced categories.
> 2. **Cleaning & Splitting:** We clean the text (lowercase, remove extra spaces) and split the data into **80% training (960 rows)** and **20% testing (240 rows)** so we can test the model on data it has never seen.
> 3. **Turning Words into Numbers (TF-IDF):** Computers cannot read words directly. TF-IDF converts each description into numbers based on how important each word or 2-word phrase is. We fit TF-IDF **only on the training data** to prevent cheating (data leakage).
> 4. **Training & Comparing 3 Models:** We train Naive Bayes, Logistic Regression, and Linear SVM, then compare their accuracy and F1-score on the 240 test examples. **Linear SVM** achieved 100% on the test set and was saved as `.pkl` files.
> 5. **Ready-to-Use Function:** Our teammates can now simply call `categorize_expense("...")` in Python and instantly get the category and confidence score without retraining anything.
