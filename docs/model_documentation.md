# Model Documentation

## Pipeline Overview

1. **Text Normalization (`src/preprocessing.py`):** Lowercasing, whitespace collapsing, punctuation removal (retaining hyphens and alphanumeric tokens), and duplicate removal.
2. **Data Split:** Stratified 80:20 split (`random_state=42`) -> 960 train, 240 test.
3. **Feature Extraction:** `TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)` fit **strictly on training data**.
4. **Model Comparison (`src/train.py`):**
   - Multinomial Naive Bayes: Accuracy `0.9792`, F1 (weighted) `0.9794`
   - Logistic Regression: Accuracy `0.9875`, F1 (weighted) `0.9875`
   - **Linear SVM (`LinearSVC`):** Accuracy `1.0000`, F1 (weighted) `1.0000` (Selected)

## Confidence Score Calculation & Limitations

`LinearSVC` outputs raw hyperplane margin distances via `decision_function`. `src/expense_classifier.py` applies softmax over those margins to produce a normalized score in `(0, 1]`. This score indicates relative margin strength rather than a calibrated probability.
