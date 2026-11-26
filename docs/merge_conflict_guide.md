# Merge Conflict Prevention & Resolution Guide

## 1. Prevention Rules
1. Never push directly to `main`.
2. Always pull `development` before creating a feature branch:
   ```bash
   git checkout development
   git pull origin development
   git checkout -b feature/your-task
   ```
3. Keep files isolated by task:
   - Task 1: `data/`, `models/`, `src/generate_dataset.py`, `src/preprocessing.py`, `src/train.py`, `src/expense_classifier.py`
   - Task 2: `src/expense_tools.py`, `tests/test_tools.py`
   - Task 3: `src/assistant.py`, `tests/test_assistant.py`
   - Task 4: `app.py`, `frontend/`, `tests/test_frontend.py`

## 2. Resolving a Merge Conflict Safely
If `git merge` or GitHub PR reports a conflict:
1. Pull `development` into your feature branch: `git merge origin/development`
2. Open the conflicted file and look for `<<<<<<<`, `=======`, `>>>>>>>`.
3. Choose the correct combined code (never blindly delete a teammate's changes).
4. Remove the conflict markers, run `python -m pytest -v`, then commit:
   ```bash
   git add <resolved-file>
   git commit -m "chore: resolve merge conflict with development"
   ```
