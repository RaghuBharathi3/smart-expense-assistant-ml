# API & Module Integration Contracts (Task 4 <-> Tasks 2 & 3)

This document defines the exact interfaces **Task 4 (Chatbot UI & Dashboard)** consumes so **Task 2** and **Task 3** can be developed independently without merge conflicts or frontend rewrites.

## 1. Task 2 Contract (`src/expense_tools.py`)

Expected functions:
- `get_expenses(category: str | None = None, month: int | None = None, year: int | None = None) -> list[dict]`
  - Each dict: `{"expense_id": str, "description": str, "amount": float, "date": "YYYY-MM-DD", "category": str}`
- `calculate_total(category: str | None = None, month: int | None = None, year: int | None = None) -> float`
- `compare_months(month_a: int, month_b: int, year: int | None = None) -> dict`

## 2. Task 3 Contract (`src/assistant.py` or HTTP `POST /chat`)

Task 4 automatically detects `src.assistant.handle_query(user_message, history=...)` or an HTTP server at `EXPENSE_API_URL` (`POST /chat`).

### Expected Response Schema

```json
{
  "response_type": "summary | category_breakdown | monthly_comparison | transaction_list | categorization | text | error",
  "message": "Human-readable markdown answer in INR (₹)",
  "structured_data": {},
  "visualization": {
    "chart_type": "donut | comparison | bar | line",
    "title": "Chart Title",
    "data": {}
  },
  "error": null
}
```

## 3. Running Task 4 Locally

```bash
streamlit run app.py
```

### Environment Variables (Optional)
- `EXPENSE_API_URL`: Base URL of Task 3 HTTP backend (e.g., `http://localhost:8000`)
- `EXPENSE_API_TIMEOUT`: Request timeout in seconds (default: `10.0`)
- `EXPENSE_MOCK_MODE`: Set `true` to force the Development Fallback adapter
