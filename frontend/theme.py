"""Coherent financial UI/UX design system tokens and formatting helpers."""

from __future__ import annotations

from datetime import datetime

PALETTE = {
    "primary": "#0F172A",       # Slate 900
    "secondary": "#1D4ED8",     # Blue 700
    "accent": "#059669",        # Emerald 600
    "warning": "#D97706",       # Amber 600
    "danger": "#DC2626",        # Red 600
    "bg": "#F8FAFC",            # Slate 50
    "surface": "#FFFFFF",       # White
    "border": "#E2E8F0",        # Slate 200
    "text_primary": "#0F172A",
    "text_secondary": "#475569",
    "chart_colors": ["#1D4ED8", "#059669", "#D97706", "#7C3AED"],
}


def format_inr(amount: float) -> str:
    """Format a numeric amount in Indian Rupees (e.g., ₹12,450.50)."""
    sign = "-" if amount < 0 else ""
    abs_val = abs(float(amount))
    integer_part, decimal_part = f"{abs_val:.2f}".split(".")
    if len(integer_part) <= 3:
        formatted_int = integer_part
    else:
        last_three = integer_part[-3:]
        rest = integer_part[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        formatted_int = ",".join(groups) + "," + last_three
    return f"{sign}₹{formatted_int}.{decimal_part}"


def format_indian_date(iso_date: str) -> str:
    """Convert YYYY-MM-DD to readable Indian display format (DD MMM YYYY)."""
    try:
        dt = datetime.strptime(str(iso_date).strip(), "%Y-%m-%d")
        return dt.strftime("%d %b %Y")
    except ValueError:
        return str(iso_date)


def get_custom_css() -> str:
    """Return minimal, clean CSS for the financial product design system."""
    return f"""
    <style>
        .stApp {{
            background-color: {PALETTE["bg"]};
            color: {PALETTE["text_primary"]};
        }}
        .kpi-card {{
            background-color: {PALETTE["surface"]};
            border: 1px solid {PALETTE["border"]};
            border-radius: 10px;
            padding: 16px 18px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }}
        .kpi-label {{
            font-size: 0.80rem;
            font-weight: 600;
            color: {PALETTE["text_secondary"]};
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 1.45rem;
            font-weight: 700;
            color: {PALETTE["text_primary"]};
            line-height: 1.2;
        }}
        .kpi-sub {{
            font-size: 0.78rem;
            color: {PALETTE["text_secondary"]};
            margin-top: 4px;
        }}
        .source-badge {{
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 999px;
            border: 1px solid {PALETTE["border"]};
            background-color: #F1F5F9;
            color: {PALETTE["text_secondary"]};
            margin-bottom: 6px;
        }}
    </style>
    """
