"""Enterprise financial UI/UX design system tokens, CSS, and formatting helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

PALETTE = {
    "primary": "#1E3A5F",       # Deep Steel Navy
    "secondary": "#0F766E",     # Muted Teal
    "ink": "#111827",           # Gray 900 (WCAG AAA on white)
    "ink_secondary": "#4B5563", # Gray 600 (WCAG AA on white)
    "ink_muted": "#6B7280",     # Gray 500
    "bg": "#F9FAFB",            # Gray 50 canvas
    "surface": "#FFFFFF",       # Card / table surface
    "surface_alt": "#F3F4F6",   # Subtle hover / tag surface
    "border": "#E5E7EB",        # Hairline border
    "border_strong": "#D1D5DB", # Interactive control border
    "danger": "#B91C1C",        # Muted Crimson
    "chart_colors": ["#1E3A5F", "#0F766E", "#64748B", "#B45309"],
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


def style_plotly_figure(fig: Any, height: int = 310) -> Any:
    """Apply consistent, restrained enterprise styling to all Plotly figures."""
    fig.update_layout(
        height=height,
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            size=12,
            color=PALETTE["ink_secondary"],
        ),
        title_font=dict(size=14, color=PALETTE["ink"]),
        margin=dict(t=42, b=16, l=12, r=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="left",
            x=0,
            font=dict(size=11, color=PALETTE["ink_secondary"]),
        ),
        hoverlabel=dict(
            bgcolor=PALETTE["ink"],
            font_size=12,
            font_color="#FFFFFF",
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=PALETTE["border"],
        tickfont=dict(size=11, color=PALETTE["ink_secondary"]),
        title_font=dict(size=12, color=PALETTE["ink_secondary"]),
    )
    fig.update_yaxes(
        gridcolor=PALETTE["surface_alt"],
        zerolinecolor=PALETTE["border"],
        tickfont=dict(size=11, color=PALETTE["ink_secondary"]),
        title_font=dict(size=12, color=PALETTE["ink_secondary"]),
    )
    return fig


def get_custom_css() -> str:
    """Return restrained, enterprise-grade CSS with WCAG focus & responsive rules."""
    return f"""
    <style>
        html, body, [class*="css"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: {PALETTE["ink"]};
        }}
        .stApp {{
            background-color: {PALETTE["bg"]};
        }}
        .block-container {{
            max-width: 1200px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}
        .app-header {{
            border-bottom: 1px solid {PALETTE["border"]};
            padding-bottom: 0.75rem;
            margin-bottom: 1rem;
        }}
        .app-header h1 {{
            font-size: 1.35rem;
            font-weight: 600;
            color: {PALETTE["ink"]};
            margin: 0;
            letter-spacing: -0.01em;
        }}
        .app-header p {{
            font-size: 0.875rem;
            color: {PALETTE["ink_secondary"]};
            margin: 0.25rem 0 0 0;
        }}
        .section-title {{
            font-size: 0.95rem;
            font-weight: 600;
            color: {PALETTE["ink"]};
            margin: 0.5rem 0 0.75rem 0;
        }}
        .kpi-card {{
            background-color: {PALETTE["surface"]};
            border: 1px solid {PALETTE["border"]};
            border-radius: 6px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(17, 24, 39, 0.03);
        }}
        .kpi-label {{
            font-size: 0.72rem;
            font-weight: 600;
            color: {PALETTE["ink_secondary"]};
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 4px;
        }}
        .kpi-value {{
            font-size: 1.3rem;
            font-weight: 600;
            color: {PALETTE["ink"]};
            font-variant-numeric: tabular-nums;
            line-height: 1.25;
        }}
        .kpi-sub {{
            font-size: 0.75rem;
            color: {PALETTE["ink_muted"]};
            margin-top: 4px;
        }}
        .source-badge {{
            display: inline-block;
            font-size: 0.68rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            padding: 2px 7px;
            border-radius: 4px;
            border: 1px solid {PALETTE["border"]};
            background-color: {PALETTE["surface_alt"]};
            color: {PALETTE["ink_secondary"]};
            margin-bottom: 6px;
        }}
        .stButton > button {{
            background-color: {PALETTE["surface"]};
            color: {PALETTE["ink"]};
            border: 1px solid {PALETTE["border_strong"]};
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 500;
            padding: 0.4rem 0.75rem;
            transition: background-color 0.15s ease, border-color 0.15s ease;
        }}
        .stButton > button:hover {{
            background-color: {PALETTE["surface_alt"]};
            border-color: {PALETTE["primary"]};
            color: {PALETTE["ink"]};
        }}
        .stButton > button:focus-visible {{
            outline: 2px solid {PALETTE["primary"]};
            outline-offset: 2px;
        }}
        @media (max-width: 768px) {{
            .block-container {{
                padding-left: 0.85rem;
                padding-right: 0.85rem;
                padding-top: 1rem;
            }}
            .kpi-value {{
                font-size: 1.12rem;
            }}
        }}
    </style>
    """
