# Contributing Guide

Welcome to **Smart Expense Assistant ML**! Follow this guide so we can collaborate cleanly without overwriting each other's work.

## Team Workflow & Roles

- **Developer 1:** Dataset & preprocessing (`data/`, `src/generate_dataset.py`, `src/preprocessing.py`)
- **Developer 2:** Model training & evaluation (`src/train.py`, `models/`, `notebooks/`)
- **Raghu:** Integration, repository management, testing & documentation (`src/expense_classifier.py`, `tests/`, `docs/`)

## Branching Guidelines

- `main`: Stable, tested, production-ready code only. **Never push directly to `main`.**
- `development`: Shared integration branch where completed features are merged and tested together.
- `feature/<topic>`: Personal/task branches created from `development` (e.g., `feature/dataset`, `feature/model`, `feature/integration`).

## Step-by-Step Contribution Flow

1. **Sync with `development` before starting:**
   ```bash
   git checkout development
   git pull origin development
   ```
2. **Create your feature branch:**
   ```bash
   git checkout -b feature/your-task-name
   ```
3. **Make small, focused changes and run tests:**
   ```bash
   python -m pytest -v
   ```
4. **Stage and commit using Conventional Commits:**
   ```bash
   git add <files>
   git commit -m "feat: short description of your change"
   ```
5. **Push your branch and open a Pull Request into `development`:**
   ```bash
   git push -u origin feature/your-task-name
   ```

## Commit Message Guidelines

Use clear prefixes:
- `feat:` New feature or module
- `fix:` Bug fix
- `docs:` Documentation updates
- `test:` Adding or updating tests
- `chore:` Config, `.gitignore`, or maintenance

## What Never to Commit

- API keys, tokens, `.env` files, or credentials
- Virtual environments (`venv/`, `.venv/`)
- `__pycache__/`, `.pytest_cache/`, or temporary files
