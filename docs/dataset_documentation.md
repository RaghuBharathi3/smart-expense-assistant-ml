# Dataset Documentation

## Overview

- **File:** `data/expenses.csv`
- **Generator:** `src/generate_dataset.py` (deterministic with `SEED = 42`)
- **Total Records:** 1,200
- **Categories (Balanced):** `Food` (300), `Travel` (300), `Shopping` (300), `Utilities` (300)

## Schema

| Column | Type | Description |
|---|---|---|
| `expense_id` | `str` | Unique identifier (`EXP-0001` to `EXP-1200`) |
| `description` | `str` | Natural-language Indian expense description |
| `amount` | `float` | Transaction amount in INR |
| `date` | `str` | ISO-8601 date (`YYYY-MM-DD`) |
| `category` | `str` | Ground-truth label (`Food`, `Travel`, `Shopping`, `Utilities`) |

## EDA Summary

- **Missing Values:** 0 across all columns
- **Duplicates:** 0 exact row duplicates, 0 duplicate descriptions
- **Amount (INR):** Min `25.07`, Median `876.25`, Mean `1420.43`, Max `11985.93`
- **Description Length:** 3 to 12 words (mean `7.01`), 19 to 75 characters (mean `43.21`)

EDA plots are stored in `docs/figures/`:
- `category_distribution.png`
- `amount_distribution.png`
- `description_length.png`
