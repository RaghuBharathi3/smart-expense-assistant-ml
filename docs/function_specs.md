# Backend Function Specifications (`src/backend/tools.py`)

All tools read from `data/runtime/expenses.csv` (initialized from read-only `data/expenses.csv`).

| Function | Arguments | Return Value |
|---|---|---|
| `get_expenses` | `category: str \| None = None, month: int \| None = None, year: int \| None = None` | `list[dict]` of matching expense records sorted newest first |
| `calculate_spending` | `category: str \| None = None, month: int \| None = None, year: int \| None = None` | `float` total spending in INR (`₹`) |
| `compare_months` | `month_a: int, month_b: int, year: int \| None = None` | `dict` with `total_a`, `total_b`, `difference`, `percent_change`, and category breakdowns |
| `get_highest_category` | `month: int \| None = None, year: int \| None = None` | `dict` with `category` and `amount` |
| `get_category_breakdown` | `month: int \| None = None, year: int \| None = None` | `dict[str, float]` mapping each category to total INR spend |
| `add_expense` | `description: str, amount: float, expense_date: str, category: str \| None = None, expense_id: str \| None = None` | `dict` of the validated and persisted expense record |
