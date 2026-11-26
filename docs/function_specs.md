# Backend Function Specifications (`src/backend/tools.py` & `src/backend/router.py`)

All tools read from `data/runtime/expenses.csv` (initialized from read-only `data/expenses.csv`).

## 1. Backend Tool Functions (`src/backend/tools.py`)

| Function | Arguments | Return Value |
|---|---|---|
| `get_expenses` | `category: str \| None = None, month: int \| None = None, year: int \| None = None` | `list[dict]` of matching expense records sorted newest first |
| `calculate_spending` | `category: str \| None = None, month: int \| None = None, year: int \| None = None` | `float` total spending in INR (`₹`) |
| `compare_months` | `month_a: int, month_b: int, year: int \| None = None` | `dict` with `total_a`, `total_b`, `difference`, `percent_change`, and category breakdowns |
| `get_highest_category` | `month: int \| None = None, year: int \| None = None` | `dict` with `category` and `amount` |
| `get_category_breakdown` | `month: int \| None = None, year: int \| None = None` | `dict[str, float]` mapping each category to total INR spend |
| `add_expense` | `description: str, amount: float, expense_date: str, category: str \| None = None, expense_id: str \| None = None` | `dict` of the validated and persisted expense record |

## 2. Qwen2.5-0.5B-Instruct JSON Contract (`src/backend/router.py`)

`execute_tool_call(payload)` and `process_query(user_message, llm_json=...)` validate Qwen's experimental JSON using Pydantic (`QwenIntentPayload`) and map intents + string months cleanly:

```json
{
  "intent": "calculate_spending | compare_months | highest_category | categorize_expense | show_expenses | add_expense",
  "category": "Food | Travel | Shopping | Utilities | null",
  "month": "November | null",
  "amount": 250.0,
  "description": "Uber ride to office"
}
```

### Connecting Qwen in Google Colab / Streamlit

```python
from src.backend.router import register_llm_handler

register_llm_handler(your_qwen_json_function)  # takes user_query: str -> JSON dict or str
```

Run the Streamlit UI locally:

```bash
streamlit run src/interface/app.py
```
