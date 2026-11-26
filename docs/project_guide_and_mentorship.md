# Smart Expense Assistant ML — Project Guide, Audit & Mentorship Notes

This document captures our complete Task 1 delivery, Git/GitHub & Google Colab walkthrough, project audit, architecture guide, professor demo prep, and team coordination plan.

---

## 1. Initial Repository Assessment & Task 1 Delivery

- **Repository:** `https://github.com/RaghuBharathi3/smart-expense-assistant-ml` (Private)
- **Collaborators:** `zephyrgrimm`, `praveenkumar-2006`
- **Initial State:** Repository existed on GitHub with 0 commits and 0 branches; cloned locally, initialized `main` with `.gitignore`, `LICENSE`, and `requirements.txt`, and built Task 1 across `feature/dataset`, `feature/model`, `feature/integration`, and `development`.

### Files Implemented for Task 1
- **Dataset & EDA:** `data/expenses.csv` (1,200 balanced records: 300 each for `Food`, `Travel`, `Shopping`, `Utilities`), `src/generate_dataset.py`, `src/eda.py`, and plots in `docs/figures/`.
- **Preprocessing & Training:** `src/preprocessing.py`, `src/train.py`, `models/expense_model.pkl`, `models/vectorizer.pkl`, `models/model_metadata.json`, `models/README.md`.
- **Prediction & Testing:** `src/expense_classifier.py` (`load_model`, `categorize_expense`, `predict_with_confidence`, `batch_categorize_expenses`) and `tests/test_classifier.py` (24/24 pytest tests passing).
- **Notebook & Docs:** `notebooks/expense_classification.ipynb`, `README.md`, `CONTRIBUTING.md`, `docs/dataset_documentation.md`, `docs/model_documentation.md`, `docs/integration_guide.md`, `.github/pull_request_template.md`, and `.github/ISSUE_TEMPLATE/`.

### Actual Model Evaluation Results (240-Record Stratified Test Set)

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 0.9792 | 0.9808 | 0.9792 | 0.9794 |
| Logistic Regression | 0.9875 | 0.9881 | 0.9875 | 0.9875 |
| **Linear SVM (`LinearSVC`)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

---

## 2. Git, GitHub & Google Colab Quick Guide

### Git Basics (Lesson 1)
- **Repository (Repo):** A smart project folder (`smart-expense-assistant-ml`) tracking all files and history.
- **Git:** Local version control installed on your computer that tracks every change.
- **GitHub:** Cloud hosting (`github.com`) where you and your teammates share branches and open Pull Requests.
- **Check status anytime:**
  ```powershell
  git status
  ```

### Using Google Colab Safely with a Private Repo
1. Open `https://colab.research.google.com` → **File → Open notebook → GitHub** tab.
2. Check **"Include private repos"**, select `RaghuBharathi3/smart-expense-assistant-ml`, and open `notebooks/expense_classification.ipynb`.
3. Run cells (`Runtime → Run all` or `Shift + Enter`). When Cell 2 asks for a token via `getpass`, enter a fine-grained GitHub Personal Access Token (never hardcode tokens in cells).
4. To save edits from Colab without overwriting teammates: **File → Save a copy in GitHub** → choose a `feature/*` branch (never `main`), then open a Pull Request into `development`.

### Syncing Local Branches
```powershell
git checkout main
git pull origin main
```

---

## 3. Task 1 Audit Summary (98% Complete)

| Area | Status | Notes |
|---|---|---|
| Dataset preparation & EDA | **COMPLETED** | 1,200 balanced records, 0 missing/duplicates, 3 EDA plots |
| Data preprocessing & TF-IDF | **COMPLETED** | Stratified 80:20 split; TF-IDF fit strictly on training data |
| Model training & comparison | **COMPLETED** | NB, Logistic Regression, and Linear SVM compared |
| Model evaluation | **COMPLETED** | Saved in `models/model_metadata.json` & `docs/figures/confusion_matrix.png` |
| Prediction module | **COMPLETED** | `src/expense_classifier.py` verified |
| Model export & loading | **COMPLETED** | Independent loading from `.pkl` without retraining |
| Automated testing | **COMPLETED** | `python -m pytest -v` → 24 passed |
| Google Colab execution | **PARTIAL** | Notebook JSON valid & updated; live Colab runtime execution pending user run |
| Integration compatibility | **COMPLETED** | Smoke test passed across all 4 categories |
| README & technical docs | **COMPLETED** | Full docs & collaboration templates present |
| GitHub version control & collab | **PARTIAL** | PR #1 & PR #2 merged into `main`; sync `development` with `main` |

---

## 4. Full System Architecture & Codebase Walkthrough

```text
[Task 4: Chatbot UI]
        │
        ▼
[Task 3: LLM + Tool Calling]
        │
        ├──► [Task 1: ML Expense Classifier] (DONE - src/expense_classifier.py)
        │
        └──► [Task 2: Expense Retrieval & Calculation Tools] (NEXT)
```

### Execution Flow (Task 1)
1. Input description (e.g., `"Uber ride to office"`) → validated & cleaned (`clean_text()`).
2. `vectorizer.pkl` converts unigrams/bigrams into TF-IDF features.
3. `expense_model.pkl` (`LinearSVC`) predicts `"Travel"`.
4. Softmax over `decision_function` margins computes confidence (`0.7258`).
5. Returns `{"description": "Uber ride to office", "category": "Travel", "confidence": 0.7258}`.

---

## 5. Professor Demo & Q&A Prep

- **Why TF-IDF + Linear Models?** Expense descriptions are short (3–12 words); TF-IDF bigrams + linear models are fast, accurate, and require no GPU.
- **How was data leakage prevented?** Stratified 80:20 split happens *before* fitting `TfidfVectorizer`.
- **Why 100% test accuracy?** Dataset is synthetic and structured, though overlapping keywords (`online`, `bill`, `Amazon`) were included to differentiate NB (`0.9794`), Logistic Regression (`0.9875`), and Linear SVM (`1.0000`).
- **Confidence Limitation:** `LinearSVC` outputs margin distances rather than calibrated probabilities; softmax over margins reflects relative margin strength.

---

## 6. Team Task Allocation & Meeting Agenda

| Member | Focus | Branch | Deliverables |
|---|---|---|---|
| **Raghu (`RaghuBharathi3`)** | Repo Lead & Task 1 | `main` / `development` | Sync `development`, verify Colab, review PRs |
| **Teammate 1 (`zephyrgrimm`)** | Task 2: Retrieval & Math Tools | `feature/expense-tools` | `src/expense_tools.py` + `tests/test_tools.py` |
| **Teammate 2 (`praveenkumar-2006`)** | Task 3 & 4: LLM + Chatbot UI | `feature/llm-chatbot` | `src/assistant.py` + chat UI (`app.py`) |

### 20-Minute Team Meeting Agenda
1. **Status Demo (5 min):** Show `src/expense_classifier.py` and `pytest` passing (24/24).
2. **Git Workflow Rules (3 min):** Pull `development` first, branch `feature/*`, never push to `main`, use PRs.
3. **Define Task 2 & 3 Signatures (7 min):** Agree on exact function names/parameters for Task 2.
4. **Assign Branches & Deadlines (5 min):** Confirm branch owners and Milestone 2 check-in date.
