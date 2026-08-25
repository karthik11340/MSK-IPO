import io
import os
import re
import numpy as np
import pandas as pd
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Flowable,
    Paragraph, Spacer, Table, TableStyle, Image as RLImage,
    KeepTogether
)

try:
    import groq_analyzer
except Exception:
    groq_analyzer = None


# ==========================================================
# BRAND CONFIG  ->  MS KAPITAL (BLUE THEME)
# ==========================================================
BRAND_NAME = "MS KAPITAL"
BRAND_TAG = "IPO Market Intelligence  •  Capital Markets Research  •  Investment Advisory"

INK_HEX     = "#12263F"
ACCENT_HEX  = "#2563EB"
GREY_HEX    = "#6B7280"
LIGHT_HEX   = "#EAF0F7"

INK    = colors.HexColor(INK_HEX)
ACCENT = colors.HexColor(ACCENT_HEX)
GREY   = colors.HexColor(GREY_HEX)
LIGHT  = colors.HexColor(LIGHT_HEX)

W, H = A4
M_L = 16 * mm
M_R = 16 * mm
M_T = 18 * mm
M_B = 18 * mm
AVAIL = W - M_L - M_R

TREND_FS  = (7.4, 2.9)
SECTOR_FS = (7.4, 3.2)
SIZE_FS   = (7.4, 2.6)
WEEKLY_FS = (7.4, 2.4)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#999999",
    "axes.labelcolor": "#555555",
    "figure.facecolor": "white"
})


# ==========================================================
# HELPERS
# ==========================================================
def fmt_month(period):
    try:
        return pd.Period(period, freq="M").strftime("%B %Y")
    except Exception:
        return str(period)


def _cr(value):
    try:
        if value is None or pd.isna(value):
            return "0"
        return f"{float(value):,.0f}"
    except Exception:
        return "0"


def _pdf_safe(text):
    """
    Converts Unicode characters that Helvetica/WinAnsi cannot print
    into safe equivalents, then drops anything still unprintable.
    """
    if text is None:
        return ""
    s = str(text)

    replacements = {
        "‑": "-", "‐": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
        "‘": "'", "’": "'",
        "“": '"', "”": '"',
        "…": "...",
        " ": " ", "​": "",
        "₹": "Rs. ",
        "→": "->", "←": "<-", "⇒": "->",
        "■": "", "▪": "", "●": "", "◆": "", "◇": "",
        "✓": "", "✔": "", "✗": "", "✖": "",
        "★": "", "☆": "", "▲": "", "▼": "",
        "•": "•",
        "»": "»",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    s = s.encode("cp1252", errors="ignore").decode("cp1252")
    return s


def _esc(value):
    """Plain text escape - no HTML allowed. Use for table cells and raw data."""
    if value is None:
        return ""
    return escape(_pdf_safe(str(value)))


def _esc_html(value):
    """
    Escape text but preserve allowed HTML formatting tags:
    <b>, </b>, <i>, </i>, <font ...>, </font>, <br/>, <br />
    """
    if value is None:
        return ""
    s = _pdf_safe(str(value))

    # Capture allowed tags
    allowed_tags = []
    tag_pattern = re.compile(r'<(/?)(b|i|br)(\s*/?>|\s*>)|<(/?)font(\s[^>]*)?>')

    def store_tag(match):
        allowed_tags.append(match.group(0))
        return f"##TAG_{len(allowed_tags)-1}##"

    s = tag_pattern.sub(store_tag, s)
    s = escape(s)

    # Restore tags
    for i, tag in enumerate(allowed_tags):
        s = s.replace(f"##TAG_{i}##", tag)

    return s


def _md_to_html(text):
    """Convert markdown bold **text** and italic *text* to HTML tags."""
    if not text:
        return text
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    return text


def bold(text):
    """
    EXPLICIT BOLD - Reliable across all reportlab environments.
    Use this instead of <b> tags for guaranteed bold rendering.
    """
    return f'<font name="Helvetica-Bold">{_esc(text)}</font>'


def b_html(text):
    """Explicit bold for text that may contain other HTML."""
    return f'<font name="Helvetica-Bold">{_esc_html(text)}</font>'


def _date(s):
    try:
        if s is None or str(s).lower() in ("nan", "none", "", "nat"):
            return "-"
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.isna(dt):
            return "-"
        return dt.strftime("%d %b %Y")
    except Exception:
        return "-"


def _img_height(figsize):
    return AVAIL * (figsize[1] / figsize[0])


def _growth(curr, prev):
    try:
        if prev in (0, None) or np.isnan(prev):
            return "-"
        g = (curr - prev) / prev * 100.0
        return f"{'+' if g >= 0 else ''}{g:.1f}%"
    except Exception:
        return "-"


def _growth_html(curr, prev):
    """Growth with explicit color coding."""
    try:
        if prev in (0, None) or np.isnan(prev):
            return "-"
        g = (curr - prev) / prev * 100.0
        color = ACCENT_HEX if g >= 0 else "#DC2626"
        arrow = "▲" if g >= 0 else "▼"
        return f'<font color="{color}">{arrow} {"+" if g >= 0 else ""}{g:.1f}%</font>'
    except Exception:
        return "-"


def _stats_block(sub):
    n = len(sub)
    mb = sub[sub["board"] == "Main Board"]
    sme = sub[sub["board"] == "SME"]
    return {
        "count": n,
        "raised": float(sub["size_cr"].sum()) if n else 0.0,
        "mb_count": len(mb),
        "mb_raised": float(mb["size_cr"].sum()) if len(mb) else 0.0,
        "sme_count": len(sme),
        "sme_raised": float(sme["size_cr"].sum()) if len(sme) else 0.0,
        "avg": float(sub["size_cr"].mean()) if n else 0.0,
        "median": float(sub["size_cr"].median()) if n else 0.0,
    }


# ==========================================================
# AI SECTION PARSING
# ==========================================================
def parse_ai_sections(text):
    sections = {}
    if not text:
        return sections
    current = "_intro"
    sections[current] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            current = line.lstrip("#").strip()
            sections.setdefault(current, [])
            continue
        if line.startswith("- ") or line.startswith("* ") or line.startswith("• "):
            block = {"type": "bullet", "text": line[2:].strip()}
        elif len(line) > 2 and line[0].isdigit() and line[1] in ".)":
            block = {"type": "bullet", "text": line[2:].strip()}
        elif line.startswith("**") and line.endswith("**") and len(line) < 80:
            block = {"type": "heading", "text": line.strip("*").strip()}
        else:
            block = {"type": "para", "text": line}
        sections.setdefault(current, []).append(block)
    return sections


def find_ai_section(ai_sections, *keywords):
    if not ai_sections:
        return None
    for heading, blocks in ai_sections.items():
        h = heading.lower()
        for kw in keywords:
            if kw.lower() in h:
                return blocks
    return None


# ==========================================================
# FLOWABLES
# ==========================================================
class CoverHead(Flowable):
    def __init__(self, month_label):
        Flowable.__init__(self)
        self.month_label = month_label
        self.height = 52 * mm

    def wrap(self, aw, ah):
        return aw, self.height

    def draw(self):
        c = self.canv
        top = self.height
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(ACCENT)
        c.drawString(0, top - 6 * mm, "MONTHLY CAPITAL MARKETS BRIEF")
        c.setFont("Helvetica-Bold", 30)
        c.setFillColor(INK)
        c.drawString(0, top - 18 * mm, "IPO MARKET UPDATE")
        c.setFont("Helvetica", 11)
        c.setFillColor(GREY)
        c.drawString(0, top - 26 * mm, f"Comprehensive Monthly IPO Review - {self.month_label}")
        c.setFillColor(ACCENT)
        c.rect(0, top - 31 * mm, 42 * mm, 1.6 * mm, stroke=0, fill=1)
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(INK)
        c.drawString(0, top - 37 * mm, BRAND_NAME)
        c.setFont("Helvetica", 7.5)
        c.setFillColor(GREY)
        c.drawString(24 * mm, top - 36.8 * mm, "Capital Markets Intelligence Desk")


class SectionHead(Flowable):
    def __init__(self, kicker, title):
        Flowable.__init__(self)
        self.kicker = kicker
        self.title = title
        self.height = 15 * mm

    def wrap(self, aw, ah):
        return aw, self.height

    def draw(self):
        c = self.canv
        top = self.height
        c.setFont("Helvetica-Bold", 7.5)
        c.setFillColor(ACCENT)
        c.drawString(0, top - 4 * mm, self.kicker.upper())
        c.setFont("Helvetica-Bold", 14.5)
        c.setFillColor(INK)
        c.drawString(0, top - 10 * mm, self.title)
        c.setFillColor(ACCENT)
        c.rect(0, top - 13 * mm, 24 * mm, 1.2 * mm, stroke=0, fill=1)


class KeyStatsBox(Flowable):
    """Visual highlight box for cover page key stats."""
    def __init__(self, stats):
        Flowable.__init__(self)
        self.stats = stats  # list of (label, value) tuples
        self.height = 28 * mm

    def wrap(self, aw, ah):
        self.width = aw
        return aw, self.height

    def draw(self):
        c = self.canv
        aw = self.width
        c.setFillColor(LIGHT)
        c.roundRect(0, 0, aw, self.height, 3 * mm, stroke=0, fill=1)

        n = len(self.stats)
        col_w = aw / n
        for i, (label, value) in enumerate(self.stats):
            x = i * col_w + 4 * mm
            c.setFont("Helvetica-Bold", 11)
            c.setFillColor(INK)
            c.drawString(x, self.height - 8 * mm, str(value))
            c.setFont("Helvetica", 6.5)
            c.setFillColor(GREY)
            c.drawString(x, self.height - 13 * mm, str(label).upper())


# ==========================================================
# TABLE CELL STYLES
# ==========================================================
cell_header = ParagraphStyle("cell_header", fontName="Helvetica-Bold", fontSize=7.6, textColor=colors.white, leading=9.5)
cell_left = ParagraphStyle("cell_left", fontName="Helvetica", fontSize=7.6, textColor=INK, leading=9.5)
cell_right = ParagraphStyle("cell_right", fontName="Helvetica", fontSize=7.6, textColor=INK, leading=9.5, alignment=TA_RIGHT)
cell_bold_left = ParagraphStyle("cell_bold_left", fontName="Helvetica-Bold", fontSize=7.6, textColor=INK, leading=9.5)
cell_bold_right = ParagraphStyle("cell_bold_right", fontName="Helvetica-Bold", fontSize=7.6, textColor=INK, leading=9.5, alignment=TA_RIGHT)


def kpi_table(kpis):
    if not kpis:
        return Spacer(1, 1)
    cells = []
    row = []
    styles_v = ParagraphStyle("kv", fontName="Helvetica-Bold", fontSize=14, textColor=INK, leading=16)
    styles_l = ParagraphStyle("kl", fontName="Helvetica", fontSize=6.6, textColor=GREY, leading=8.5)
    for i, (value, label) in enumerate(kpis):
        row.append([Paragraph(_esc(value), styles_v), Paragraph(_esc(str(label).upper()), styles_l)])
        if len(row) == 3 or i == len(kpis) - 1:
            while len(row) < 3:
                row.append(["", ""])
            cells.append(row)
            row = []
    data = [[[cells[i][j][0], cells[i][j][1]] for j in range(3)] for i in range(len(cells))]
    t = Table(data, colWidths=[AVAIL / 3.0] * 3)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("INNERGRID", (0, 0), (-1, -1), 2, colors.white),
        ("BOX", (0, 0), (-1, -1), 0, colors.white),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))
    return t


def nice_table(header, rows, widths, right=(), bold_cols=()):
    header_cells = [Paragraph(_esc(h), cell_header) for h in header]
    data = [header_cells]
    for row in rows:
        cells = []
        for j, val in enumerate(row):
            if j in bold_cols:
                style = cell_bold_right if j in right else cell_bold_left
            else:
                style = cell_right if j in right else cell_left
            cells.append(Paragraph(_esc(val), style))
        data.append(cells)
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8DEE7")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


# ==========================================================
# CHARTS (BLUE THEME)
# ==========================================================
def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _style_axes(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EEEEEE")


def chart_trend(monthly, selected_period):
    fig, ax1 = plt.subplots(figsize=TREND_FS, dpi=160)
    if monthly.empty:
        return _png(fig)
    monthly = monthly.copy()
    if "raised" not in monthly.columns: monthly["raised"] = 0
    if "count" not in monthly.columns: monthly["count"] = 0
    monthly["raised"] = pd.to_numeric(monthly["raised"], errors="coerce").fillna(0)
    monthly["count"] = pd.to_numeric(monthly["count"], errors="coerce").fillna(0)
    x = np.arange(len(monthly))
    ax1.bar(x, monthly["raised"], color=INK_HEX, width=0.62, zorder=3)
    sel = (monthly["period"].astype(str) == str(selected_period)).to_numpy()
    if sel.any():
        ax1.bar(x[sel], monthly.loc[sel, "raised"].values, color=ACCENT_HEX, width=0.62, zorder=4)
    for i, v in enumerate(monthly["raised"].values):
        ax1.annotate(_cr(v), (i, v), ha="center", va="bottom", fontsize=6, color="#555555")
    ax2 = ax1.twinx()
    ax2.plot(x, monthly["count"].values, color=ACCENT_HEX, marker="o", ms=4, lw=1.8, zorder=5)
    labels = pd.to_datetime(monthly["period"].astype(str) + "-01", errors="coerce").dt.strftime("%b %y")
    labels = labels.fillna(monthly["period"].astype(str))
    ax1.set_xticks(x); ax1.set_xticklabels(labels.tolist(), fontsize=7)
    ax1.set_ylabel("Rs. Cr", fontsize=7.5)
    ax2.set_ylabel("IPO Count", fontsize=7.5, color=ACCENT_HEX)
    ax2.tick_params(labelsize=7, colors=ACCENT_HEX)
    _style_axes(ax1)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    return _png(fig)


def chart_boards_bar(board_df):
    fig, ax = plt.subplots(figsize=SECTOR_FS, dpi=160)
    if board_df is None or board_df.empty:
        return _png(fig)

    boards = [str(b) for b in board_df.index.tolist()]
    counts = pd.to_numeric(board_df["count"], errors="coerce").fillna(0).tolist()
    raised = pd.to_numeric(board_df["raised"], errors="coerce").fillna(0).tolist()
    x = np.arange(len(boards))

    ax.bar(x, raised, color=[INK_HEX, ACCENT_HEX][:len(boards)], width=0.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(boards, fontsize=8)
    ax.set_ylabel("Rs. Cr", fontsize=7.5)
    ax.set_title("Main Board vs SME - Mobilisation (Rs. Cr)", fontsize=8.5, color=INK_HEX, weight="bold")

    for i, v in enumerate(raised):
        ax.annotate(
            f"Rs. {_cr(v)} Cr\n{int(counts[i])} issues",
            (i, v), ha="center", va="bottom", fontsize=7.5, color="#333333"
        )

    _style_axes(ax)
    return _png(fig)


def chart_sectors(sector_df):
    fig, ax = plt.subplots(figsize=SECTOR_FS, dpi=160)
    if sector_df.empty:
        return _png(fig)
    val_col = "raised" if "raised" in sector_df.columns else "r"
    label_col = "sector" if "sector" in sector_df.columns else sector_df.columns[0]
    sector_df = sector_df.copy()
    sector_df[val_col] = pd.to_numeric(sector_df[val_col], errors="coerce").fillna(0)
    sector_df = sector_df.sort_values(val_col, ascending=True).tail(9)
    values = sector_df[val_col].values
    labels = sector_df[label_col].astype(str).values
    y = np.arange(len(sector_df))
    max_pos = int(values.argmax()) if len(values) else -1
    bar_colors = [ACCENT_HEX if i == max_pos else INK_HEX for i in range(len(values))]
    ax.barh(y, values, color=bar_colors, height=0.62)
    ax.set_yticks(y); ax.set_yticklabels([_pdf_safe(str(x))[:26] for x in labels], fontsize=7.5)
    for i, v in enumerate(values):
        ax.annotate(_cr(v), (v, i), va="center", fontsize=6.5, color="#555555", xytext=(3, 0), textcoords="offset points")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.set_xlabel("Rs. Cr", fontsize=7.5)
    return _png(fig)


def chart_size_dist(size_df):
    fig, ax = plt.subplots(figsize=SIZE_FS, dpi=160)
    if size_df.empty:
        return _png(fig)
    labels = size_df["bucket"].tolist()
    counts = size_df["n"].tolist()
    y = np.arange(len(labels))
    ax.barh(y, counts, color=ACCENT_HEX, height=0.6)
    ax.set_yticks(y); ax.set_yticklabels([_pdf_safe(str(x)) for x in labels], fontsize=7.5)
    ax.invert_yaxis()
    for i, v in enumerate(counts):
        ax.annotate(str(int(v)), (v, i), va="center", fontsize=7, color="#555555", xytext=(3, 0), textcoords="offset points")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.set_xlabel("Number of IPOs", fontsize=7.5)
    ax.xaxis.get_major_locator().set_params(integer=True)
    return _png(fig)


def chart_weekly(weekly_df):
    """Weekly distribution chart for listings within the month."""
    fig, ax = plt.subplots(figsize=WEEKLY_FS, dpi=160)
    if weekly_df is None or weekly_df.empty:
        return _png(fig)
    weekly_df = weekly_df.copy()
    weekly_df["week_label"] = "W" + weekly_df["listing_week"].astype(str)
    x = np.arange(len(weekly_df))
    ax.bar(x, weekly_df["r"], color=INK_HEX, width=0.55, zorder=3)
    sel = weekly_df["is_max"].values if "is_max" in weekly_df.columns else np.zeros(len(weekly_df), dtype=bool)
    if sel.any():
        ax.bar(x[sel], weekly_df.loc[sel, "r"].values, color=ACCENT_HEX, width=0.55, zorder=4)
    for i, v in enumerate(weekly_df["r"].values):
        ax.annotate(_cr(v), (i, v), ha="center", va="bottom", fontsize=6.5, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels(weekly_df["week_label"].tolist(), fontsize=7.5)
    ax.set_ylabel("Rs. Cr", fontsize=7.5)
    ax.set_xlabel("Listing Week", fontsize=7.5)
    ax.set_title("Weekly Mobilisation Distribution", fontsize=8.5, color=INK_HEX, weight="bold")
    _style_axes(ax)
    return _png(fig)


# ==========================================================
# DATA AGGREGATION
# ==========================================================
def aggregate(df):
    if df is None:
        df = pd.DataFrame()
    d = df.copy()
    required = ["period", "size_cr", "board", "sector", "company_name", "merchant_banker", "city", "listing_date"]
    for col in required:
        if col not in d.columns:
            d[col] = ""
    d["size_cr"] = pd.to_numeric(d["size_cr"], errors="coerce").fillna(0)
    d["period"] = d["period"].astype(str).str.strip()
    invalid = ["", "nan", "nat", "none", "null"]
    d = d[~d["period"].str.lower().isin(invalid)].copy()
    if d.empty:
        monthly = pd.DataFrame(columns=["period", "raised", "count"])
        return d, monthly
    monthly = (d.groupby("period").agg(raised=("size_cr", "sum"), count=("period", "size"))
               .reset_index().sort_values("period"))
    return d, monthly


def _size_bucket(x):
    if x >= 1000: return "Mega  (Rs. 1,000+ Cr)"
    if x >= 500: return "Large  (Rs. 500-1,000 Cr)"
    if x >= 100: return "Medium  (Rs. 100-500 Cr)"
    if x >= 10: return "Small  (Rs. 10-100 Cr)"
    return "Micro  (Below Rs. 10 Cr)"


BUCKET_ORDER = [
    "Mega  (Rs. 1,000+ Cr)",
    "Large  (Rs. 500-1,000 Cr)",
    "Medium  (Rs. 100-500 Cr)",
    "Small  (Rs. 10-100 Cr)",
    "Micro  (Below Rs. 10 Cr)"
]


def board_listing_table(sub_df):
    rows = []
    for _, r in sub_df.iterrows():
        rows.append([
            _esc(r.get("company_name", "")),
            _esc(r.get("sector", "")),
            _cr(r.get("size_cr", 0)),
            _esc(r.get("city", "")),
            _esc(r.get("merchant_banker", "")),
            _date(r.get("listing_date", "")),
        ])
    widths = [AVAIL * x for x in (0.24, 0.15, 0.10, 0.12, 0.27, 0.12)]
    return nice_table(
        ["Company", "Sector", "Size (Rs. Cr)", "City", "Merchant Banker", "Listing Date"],
        rows, widths, right=(2,)
    )


# ==========================================================
# MAIN MONTHLY PDF BUILDER
# ==========================================================
def build_monthly_pdf(df, period):
    d, monthly = aggregate(df)
    period = str(period)
    month_label = fmt_month(period)

    mdf = d[d["period"] == period].copy()

    # Previous month
    try:
        prev_period = (pd.Period(period, freq="M") - 1).strftime("%Y-%m")
    except Exception:
        prev_period = ""
    prev_df = d[d["period"] == prev_period].copy() if prev_period else pd.DataFrame()

    # Year-over-year (same month last year)
    try:
        yoy_period = (pd.Period(period, freq="M") - 12).strftime("%Y-%m")
    except Exception:
        yoy_period = ""
    yoy_df = d[d["period"] == yoy_period].copy() if yoy_period else pd.DataFrame()

    raised = float(mdf["size_cr"].sum()) if not mdf.empty else 0.0
    count = int(len(mdf))
    prev_raised = float(prev_df["size_cr"].sum()) if not prev_df.empty else 0.0
    yoy_raised = float(yoy_df["size_cr"].sum()) if not yoy_df.empty else 0.0
    yoy_count = int(len(yoy_df))

    def change_percent(cur, prev):
        try:
            if prev in (0, None) or np.isnan(prev): return None
            return (cur - prev) / prev * 100.0
        except Exception:
            return None

    main_board_df = mdf[mdf["board"] == "Main Board"].copy()
    sme_df = mdf[mdf["board"] == "SME"].copy()
    mb_total = float(main_board_df["size_cr"].sum()) if not main_board_df.empty else 0.0
    sme_total = float(sme_df["size_cr"].sum()) if not sme_df.empty else 0.0

    # Sector analysis
    if not mdf.empty:
        sector_df = mdf.groupby("sector").agg(n=("size_cr", "size"), r=("size_cr", "sum")).reset_index()
        sector_df["n"] = pd.to_numeric(sector_df["n"], errors="coerce").fillna(0).astype(int)
        sector_df["r"] = pd.to_numeric(sector_df["r"], errors="coerce").fillna(0)
        sector_df = sector_df.sort_values("r", ascending=False)
        sector_df["share"] = (sector_df["r"] / raised * 100) if raised else 0
        sector_df["avg"] = sector_df["r"] / sector_df["n"].replace(0, np.nan)
    else:
        sector_df = pd.DataFrame(columns=["sector", "n", "r", "share", "avg"])

    # City analysis
    if not mdf.empty:
        city_df = mdf.groupby("city").agg(n=("size_cr", "size"), r=("size_cr", "sum")).reset_index()
        city_df["r"] = pd.to_numeric(city_df["r"], errors="coerce").fillna(0)
        city_df = city_df.sort_values("r", ascending=False).head(8)
        city_df["share"] = (city_df["r"] / raised * 100) if raised else 0
    else:
        city_df = pd.DataFrame(columns=["city", "n", "r", "share"])

    # Size distribution
    if not mdf.empty:
        mdf["_bucket"] = mdf["size_cr"].apply(_size_bucket)
        size_df = (mdf.groupby("_bucket").agg(n=("size_cr", "size"), r=("size_cr", "sum"))
                   .reindex(BUCKET_ORDER).fillna(0).reset_index().rename(columns={"_bucket": "bucket"}))
        size_df["n"] = size_df["n"].astype(int)
    else:
        size_df = pd.DataFrame({"bucket": BUCKET_ORDER, "n": [0] * 5, "r": [0.0] * 5})

    # Weekly distribution
    weekly_df = pd.DataFrame(columns=["listing_week", "n", "r"])
    if not mdf.empty and "listing_date" in mdf.columns:
        try:
            mdf_temp = mdf.copy()
            mdf_temp["ld_dt"] = pd.to_datetime(mdf_temp["listing_date"], errors="coerce")
            mdf_temp = mdf_temp[~mdf_temp["ld_dt"].isna()]
            if not mdf_temp.empty:
                mdf_temp["listing_week"] = mdf_temp["ld_dt"].dt.isocalendar().week.astype(int)
                weekly_df = (mdf_temp.groupby("listing_week")
                             .agg(n=("size_cr", "size"), r=("size_cr", "sum"))
                             .reset_index().sort_values("listing_week"))
                weekly_df["is_max"] = weekly_df["r"] == weekly_df["r"].max()
        except Exception:
            weekly_df = pd.DataFrame(columns=["listing_week", "n", "r"])

    # Sector momentum (current vs previous month)
    curr_sector_series = mdf.groupby("sector")["size_cr"].sum() if not mdf.empty else pd.Series(dtype=float)
    prev_sector_series = prev_df.groupby("sector")["size_cr"].sum() if not prev_df.empty else pd.Series(dtype=float)
    all_secs = sorted(set(curr_sector_series.index) | set(prev_sector_series.index),
                      key=lambda s: -float(curr_sector_series.get(s, 0)))
    sec_mom_rows = []
    for s in all_secs[:8]:
        c = float(curr_sector_series.get(s, 0)); p = float(prev_sector_series.get(s, 0))
        sec_mom_rows.append([_esc(s), _cr(p), _cr(c), _growth(c, p)])

    # Concentration metrics
    top3_raised = float(mdf.sort_values("size_cr", ascending=False).head(3)["size_cr"].sum()) if not mdf.empty else 0.0
    top5_raised = float(mdf.sort_values("size_cr", ascending=False).head(5)["size_cr"].sum()) if not mdf.empty else 0.0
    top3_share = (top3_raised / raised * 100) if raised else 0.0
    top5_share = (top5_raised / raised * 100) if raised else 0.0
    hhi = float((mdf["size_cr"] / raised * 100).pow(2).sum()) if raised and not mdf.empty else 0.0

    # Largest / smallest / records
    largest_issue = mdf.sort_values("size_cr", ascending=False).head(1) if not mdf.empty else pd.DataFrame()
    smallest_issue = mdf.sort_values("size_cr", ascending=True).head(1) if not mdf.empty else pd.DataFrame()
    largest_size = float(largest_issue["size_cr"].iloc[0]) if len(largest_issue) else 0.0
    smallest_size = float(smallest_issue["size_cr"].iloc[0]) if len(smallest_issue) else 0.0
    largest_name = _esc(largest_issue["company_name"].iloc[0]) if len(largest_issue) else "-"
    smallest_name = _esc(smallest_issue["company_name"].iloc[0]) if len(smallest_issue) else "-"

    # Trailing 12 months
    cumulative = monthly[monthly["period"] <= period].tail(12) if not monthly.empty else pd.DataFrame()
    t12_raised = float(cumulative["raised"].sum()) if not cumulative.empty else 0.0
    t12_count = int(cumulative["count"].sum()) if not cumulative.empty else 0

    # ---------- Generate AI once, then distribute ----------
    ai_full_text = None
    ai_sections = {}
    if groq_analyzer is not None:
        try:
            ai_full_text = groq_analyzer.generate_monthly_insights(d, period)
        except Exception:
            ai_full_text = None
        if ai_full_text:
            ai_sections = parse_ai_sections(ai_full_text)

    # ---------- Build document ----------
    buffer = io.BytesIO()
    doc = BaseDocTemplate(buffer, pagesize=A4, leftMargin=M_L, rightMargin=M_R, topMargin=M_T, bottomMargin=M_B,
                          title=f"IPO Market Update - {month_label}")

    def chrome(canvas, doc_obj):
        canvas.saveState()
        canvas.setFillColor(ACCENT)
        canvas.rect(M_L, H - 11 * mm, 3 * mm, 3 * mm, stroke=0, fill=1)
        canvas.setFillColor(INK)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(M_L + 5 * mm, H - 10.6 * mm, BRAND_NAME)
        canvas.setFillColor(GREY)
        canvas.setFont("Helvetica", 7.2)
        canvas.drawRightString(W - M_R, H - 10.4 * mm, "IPO MARKET UPDATE  |  MONTHLY CAPITAL MARKETS BRIEF")
        canvas.setStrokeColor(colors.HexColor("#D8DEE7")); canvas.setLineWidth(0.6)
        canvas.line(M_L, H - 13 * mm, W - M_R, H - 13 * mm)
        canvas.setFont("Helvetica", 6.8); canvas.setFillColor(GREY)
        canvas.drawString(M_L, 9 * mm, BRAND_TAG)
        canvas.drawRightString(W - M_R, 9 * mm, f"Page {doc_obj.page}")
        canvas.restoreState()

    frame = Frame(M_L, M_B, AVAIL, H - M_T - M_B)
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=chrome)])

    # Paragraph styles
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=8.6, textColor=INK, leading=12.5)
    bullet_style = ParagraphStyle("bullet", fontName="Helvetica", fontSize=8.4, textColor=INK, leading=12, spaceAfter=4)
    caption_style = ParagraphStyle("caption", fontName="Helvetica", fontSize=6.8, textColor=GREY, leading=9)
    note_style = ParagraphStyle("note", fontName="Helvetica-Oblique", fontSize=7.2, textColor=GREY, leading=10)

    ai_label_style = ParagraphStyle("ai_label", fontName="Helvetica-Bold", fontSize=7.5,
                                    textColor=ACCENT, leading=10, spaceBefore=2, spaceAfter=4)
    ai_heading_style = ParagraphStyle("ai_heading", fontName="Helvetica-Bold", fontSize=9.5,
                                      textColor=INK, leading=13, spaceBefore=8, spaceAfter=4)
    ai_para_style = ParagraphStyle("ai_para", fontName="Helvetica", fontSize=8.4,
                                   textColor=INK, leading=13.5, spaceAfter=7, alignment=TA_JUSTIFY)
    ai_bullet_style = ParagraphStyle("ai_bullet", fontName="Helvetica", fontSize=8.4,
                                     textColor=INK, leading=13, spaceAfter=4, leftIndent=6)

    story = []

    def insert_ai(*keywords):
        blocks = find_ai_section(ai_sections, *keywords)
        if not blocks:
            return
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph('<font color="#2563EB">&raquo;</font>&nbsp;&nbsp;AI INSIGHT', ai_label_style))
        for b in blocks:
            txt = _md_to_html(b["text"])
            if b["type"] == "heading":
                story.append(Paragraph(_esc_html(txt), ai_heading_style))
            elif b["type"] == "bullet":
                story.append(Paragraph(f'<font color="{ACCENT_HEX}">&bull;</font>&nbsp;&nbsp;{_esc_html(txt)}', ai_bullet_style))
            else:
                story.append(Paragraph(_esc_html(txt), ai_para_style))
        story.append(Spacer(1, 3 * mm))

    # ======================================================
    # COVER PAGE
    # ======================================================
    story.append(CoverHead(month_label))

    # Key stats box on cover
    cover_stats = [
        ("Total IPOs", count),
        ("Mobilised", f"Rs. {_cr(raised)} Cr"),
        ("Main Board", len(main_board_df)),
        ("SME Board", len(sme_df)),
    ]
    story.append(KeyStatsBox(cover_stats))
    story.append(Spacer(1, 4 * mm))

    # Executive summary with EXPLICIT BOLD (reliable rendering)
    exec_summary = (
        f"{month_label} saw {bold(str(count))} IPOs mobilise {bold(f'Rs. {_cr(raised)} Cr')} in aggregate. "
        f"The Main Board contributed {bold(str(len(main_board_df)))} issues raising {bold(f'Rs. {_cr(mb_total)} Cr')}, "
        f"while the SME platform listed {bold(str(len(sme_df)))} issues raising {bold(f'Rs. {_cr(sme_total)} Cr')}. "
        f"This brief covers market momentum, month-on-month & year-on-year changes, board and sector dynamics, "
        f"geography, issue-size distribution, weekly patterns, market concentration, detailed issue listings "
        f"and AI-driven strategic intelligence."
    )
    story.append(Paragraph(exec_summary, body))
    story.append(Spacer(1, 6 * mm))

    # ======================================================
    # 01 SNAPSHOT
    # ======================================================
    avg_size = float(mdf["size_cr"].mean()) if count else 0.0
    median_size = float(mdf["size_cr"].median()) if count else 0.0
    sme_share_count = (len(sme_df) / count * 100) if count else 0.0
    n_cities = int(mdf["city"].replace("", np.nan).nunique()) if not mdf.empty else 0

    kpis = [
        (f"Rs. {_cr(raised)} Cr", "Total Mobilised"),
        (f"{count}", "Total IPOs"),
        (f"Rs. {_cr(avg_size)} Cr", "Average Issue Size"),
        (f"Rs. {_cr(median_size)} Cr", "Median Issue Size"),
        (f"{len(main_board_df)}  /  Rs. {_cr(mb_total)} Cr", "Main Board (Issues / Mobilised)"),
        (f"{len(sme_df)}  /  Rs. {_cr(sme_total)} Cr", "SME (Issues / Mobilised)"),
        (f"Rs. {_cr(largest_size)} Cr", "Largest Issue"),
        (f"{sme_share_count:.0f}%", "SME Share of Count"),
        (f"{len(sector_df)}  /  {n_cities}", "Active Sectors / Cities"),
    ]

    story.append(KeepTogether([
        SectionHead("01 - Snapshot", f"Market Snapshot: {month_label}"),
        Spacer(1, 3 * mm),
        kpi_table(kpis),
        Spacer(1, 5 * mm),
    ]))

    # ======================================================
    # 02 HIGHLIGHTS
    # ======================================================
    bullets = []
    bullets.append(f"Total mobilisation of {bold(f'Rs. {_cr(raised)} Cr')} across {bold(str(count))} IPOs in {month_label}.")
    mom = change_percent(raised, prev_raised)
    if mom is not None:
        direction = "up" if mom >= 0 else "down"
        bullets.append(f"Mobilisation {direction} {bold(f'{abs(mom):.1f}%')} vs the previous month (Rs. {_cr(prev_raised)} Cr).")
    yoy = change_percent(raised, yoy_raised)
    if yoy is not None:
        direction = "up" if yoy >= 0 else "down"
        bullets.append(f"Year-on-year mobilisation {direction} {bold(f'{abs(yoy):.1f}%')} vs {fmt_month(yoy_period)} (Rs. {_cr(yoy_raised)} Cr).")
    if count:
        bullets.append(f"Main Board: {bold(str(len(main_board_df)))} issues raising Rs. {_cr(mb_total)} Cr; "
                       f"SME: {bold(str(len(sme_df)))} issues raising Rs. {_cr(sme_total)} Cr.")
    if not sector_df.empty:
        top_r = sector_df.sort_values("r", ascending=False).iloc[0]
        top_n = sector_df.sort_values("n", ascending=False).iloc[0]
        bullets.append(f"Top sector by mobilisation: {bold(_esc(top_r['sector']))} (Rs. {_cr(top_r['r'])} Cr); "
                       f"most active by count: {bold(_esc(top_n['sector']))} ({int(top_n['n'])} issues).")
    if not city_df.empty:
        top_city = city_df.iloc[0]
        bullets.append(f"Leading IPO city: {bold(_esc(top_city['city']))} "
                       f"({int(top_city['n'])} issues, Rs. {_cr(top_city['r'])} Cr).")
    if len(largest_issue):
        issue = largest_issue.iloc[0]
        bullets.append(f"Largest issue: {bold(_esc(issue.get('company_name', '')))} "
                       f"({_esc(issue.get('board', ''))}, {_esc(issue.get('sector', ''))}) at Rs. {_cr(issue.get('size_cr', 0))} Cr.")
    if t12_count:
        bullets.append(f"Trailing 12-month market: {bold(f'Rs. {_cr(t12_raised)} Cr')} across "
                       f"{bold(str(t12_count))} IPOs.")
    if raised:
        bullets.append(f"Market concentration: top 3 issues captured {bold(f'{top3_share:.1f}%')} of total mobilisation; "
                       f"top 5 captured {bold(f'{top5_share:.1f}%')}.")

    story.append(SectionHead("02 - Highlights", "Key Highlights & Observations"))
    for b in bullets:
        story.append(Paragraph(f'<font color="{ACCENT_HEX}">&bull;</font>&nbsp;&nbsp;{b}', bullet_style))
    story.append(Spacer(1, 5 * mm))

    # ======================================================
    # 03 MONTH-ON-MONTH COMPARISON
    # ======================================================
    cur_s = _stats_block(mdf)
    prv_s = _stats_block(prev_df)
    prev_label = fmt_month(prev_period) if prev_period else "Previous Month"
    mom_rows = [
        ["Total IPOs", str(prv_s["count"]), str(cur_s["count"]), _growth(cur_s["count"], prv_s["count"])],
        ["Mobilised (Rs. Cr)", _cr(prv_s["raised"]), _cr(cur_s["raised"]), _growth(cur_s["raised"], prv_s["raised"])],
        ["Main Board IPOs", str(prv_s["mb_count"]), str(cur_s["mb_count"]), _growth(cur_s["mb_count"], prv_s["mb_count"])],
        ["Main Board Mobilised (Rs. Cr)", _cr(prv_s["mb_raised"]), _cr(cur_s["mb_raised"]), _growth(cur_s["mb_raised"], prv_s["mb_raised"])],
        ["SME IPOs", str(prv_s["sme_count"]), str(cur_s["sme_count"]), _growth(cur_s["sme_count"], prv_s["sme_count"])],
        ["SME Mobilised (Rs. Cr)", _cr(prv_s["sme_raised"]), _cr(cur_s["sme_raised"]), _growth(cur_s["sme_raised"], prv_s["sme_raised"])],
        ["Avg Issue Size (Rs. Cr)", _cr(prv_s["avg"]), _cr(cur_s["avg"]), _growth(cur_s["avg"], prv_s["avg"])],
        ["Median Issue Size (Rs. Cr)", _cr(prv_s["median"]), _cr(cur_s["median"]), _growth(cur_s["median"], prv_s["median"])],
    ]

    story.append(KeepTogether([
        SectionHead("03 - MoM", "Month-on-Month Comparison"),
        nice_table(
            ["Metric", prev_label, month_label, "MoM Change"],
            mom_rows,
            [AVAIL * x for x in (0.34, 0.20, 0.20, 0.26)],
            right=(1, 2, 3)
        ),
    ]))
    insert_ai("comparison", "previous month", "month-on-month")
    story.append(Spacer(1, 3 * mm))

    # ======================================================
    # 04 YEAR-ON-YEAR COMPARISON
    # ======================================================
    yoy_s = _stats_block(yoy_df)
    yoy_label = fmt_month(yoy_period) if yoy_period else "Same Month Last Year"
    yoy_rows = [
        ["Total IPOs", str(yoy_s["count"]), str(cur_s["count"]), _growth(cur_s["count"], yoy_s["count"])],
        ["Mobilised (Rs. Cr)", _cr(yoy_s["raised"]), _cr(cur_s["raised"]), _growth(cur_s["raised"], yoy_s["raised"])],
        ["Main Board IPOs", str(yoy_s["mb_count"]), str(cur_s["mb_count"]), _growth(cur_s["mb_count"], yoy_s["mb_count"])],
        ["Main Board Mobilised (Rs. Cr)", _cr(yoy_s["mb_raised"]), _cr(cur_s["mb_raised"]), _growth(cur_s["mb_raised"], yoy_s["mb_raised"])],
        ["SME IPOs", str(yoy_s["sme_count"]), str(cur_s["sme_count"]), _growth(cur_s["sme_count"], yoy_s["sme_count"])],
        ["SME Mobilised (Rs. Cr)", _cr(yoy_s["sme_raised"]), _cr(cur_s["sme_raised"]), _growth(cur_s["sme_raised"], yoy_s["sme_raised"])],
        ["Avg Issue Size (Rs. Cr)", _cr(yoy_s["avg"]), _cr(cur_s["avg"]), _growth(cur_s["avg"], yoy_s["avg"])],
        ["Median Issue Size (Rs. Cr)", _cr(yoy_s["median"]), _cr(cur_s["median"]), _growth(cur_s["median"], yoy_s["median"])],
    ]

    story.append(KeepTogether([
        SectionHead("04 - YoY", "Year-on-Year Comparison"),
        nice_table(
            ["Metric", yoy_label, month_label, "YoY Change"],
            yoy_rows,
            [AVAIL * x for x in (0.34, 0.20, 0.20, 0.26)],
            right=(1, 2, 3)
        ),
    ]))
    insert_ai("year", "annual", "yoy")
    story.append(Spacer(1, 3 * mm))

    # ======================================================
    # 05 MOMENTUM
    # ======================================================
    story.append(SectionHead("05 - Momentum", "Listing Momentum (Trailing 12 Months)"))
    if not cumulative.empty:
        story.append(RLImage(chart_trend(cumulative, period), width=AVAIL, height=_img_height(TREND_FS)))
        story.append(Paragraph("Bars: capital mobilised (Rs. Cr, left axis). Line: number of listings (right axis). "
                               "Highlighted bar = report month.", caption_style))
    else:
        story.append(Paragraph("No momentum data available.", caption_style))
    insert_ai("momentum")
    story.append(Spacer(1, 3 * mm))

    # ======================================================
    # 06 BOARDS
    # ======================================================
    if not mdf.empty:
        board_df = (mdf.groupby("board").agg(count=("size_cr", "size"), raised=("size_cr", "sum"))
                    .reindex(["Main Board", "SME"]).fillna(0))
    else:
        board_df = pd.DataFrame({"count": [0, 0], "raised": [0.0, 0.0]},
                                index=pd.Index(["Main Board", "SME"], name="board"))

    story.append(SectionHead("06 - Boards", "Main Board vs SME"))
    if board_df["count"].sum() > 0:
        story.append(RLImage(chart_boards_bar(board_df), width=AVAIL, height=_img_height(SECTOR_FS)))
    board_rows = []
    for board_name, row in board_df.iterrows():
        bc = int(float(row.get("count", 0) or 0)); br = float(row.get("raised", 0) or 0)
        bavg = br / bc if bc else 0; bshare = (br / raised * 100) if raised else 0
        board_rows.append([_esc(board_name), str(bc), _cr(br), _cr(bavg), f"{bshare:.1f}%"])

    story.append(Spacer(1, 2 * mm))
    story.append(nice_table(
        ["Board", "Issues", "Raised (Rs. Cr)", "Avg Size", "Share of Raised"],
        board_rows,
        [AVAIL * x for x in (0.30, 0.14, 0.24, 0.16, 0.16)],
        right=(1, 2, 3, 4)
    ))
    insert_ai("board")
    story.append(Spacer(1, 3 * mm))

    # ======================================================
    # 07 SECTORS
    # ======================================================
    story.append(SectionHead("07 - Sectors", "Sector Watch"))
    if not sector_df.empty:
        story.append(RLImage(chart_sectors(sector_df), width=AVAIL, height=_img_height(SECTOR_FS)))
    sector_rows = []
    for _, row in sector_df.head(12).iterrows():
        sector_rows.append([
            _esc(row.get("sector", "")),
            str(int(float(row.get("n", 0) or 0))),
            _cr(row.get("r", 0)),
            f"{float(row.get('share', 0) or 0):.1f}%",
            _cr(row.get("avg", 0))
        ])
    story.append(Spacer(1, 2 * mm))
    story.append(nice_table(
        ["Sector", "IPOs", "Raised (Rs. Cr)", "Share %", "Avg Size"],
        sector_rows,
        [AVAIL * x for x in (0.34, 0.12, 0.20, 0.14, 0.20)],
        right=(1, 2, 3, 4)
    ))
    if sec_mom_rows:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph("<b>Sector mobilisation - current vs previous month</b>", caption_style))
        story.append(Spacer(1, 1.5 * mm))
        story.append(nice_table(
            ["Sector", f"{prev_label} (Rs. Cr)", f"{month_label} (Rs. Cr)", "MoM Change"],
            sec_mom_rows,
            [AVAIL * x for x in (0.34, 0.20, 0.20, 0.26)],
            right=(1, 2, 3)
        ))
    insert_ai("sector", "geographic")
    story.append(Spacer(1, 3 * mm))

    # ======================================================
    # 08 GEOGRAPHY
    # ======================================================
    story.append(SectionHead("08 - Geography", "Top IPO Cities"))
    city_rows = []
    for _, row in city_df.iterrows():
        city_rows.append([_esc(row.get("city", "")), str(int(float(row.get("n", 0) or 0))),
                          _cr(row.get("r", 0)), f"{float(row.get('share', 0) or 0):.1f}%"])
    if city_rows:
        story.append(nice_table(
            ["City", "IPOs", "Raised (Rs. Cr)", "Share %"],
            city_rows,
            [AVAIL * x for x in (0.46, 0.14, 0.24, 0.16)],
            right=(1, 2, 3)
        ))
    else:
        story.append(Paragraph("No city data available.", caption_style))
    story.append(Spacer(1, 5 * mm))

    # ======================================================
    # 09 SIZE DISTRIBUTION
    # ======================================================
    story.append(SectionHead("09 - Issue Size", "Issue Size Distribution"))
    if count:
        story.append(RLImage(chart_size_dist(size_df), width=AVAIL, height=_img_height(SIZE_FS)))
    size_rows = []
    for _, row in size_df.iterrows():
        n_val = int(float(row.get("n", 0) or 0))
        r_val = float(row.get("r", 0) or 0)
        share = (r_val / raised * 100) if raised else 0
        size_rows.append([_esc(row.get("bucket", "")), str(n_val), _cr(r_val), f"{share:.1f}%"])
    story.append(Spacer(1, 2 * mm))
    story.append(nice_table(
        ["Size Band", "IPOs", "Raised (Rs. Cr)", "Share %"],
        size_rows,
        [AVAIL * x for x in (0.46, 0.14, 0.24, 0.16)],
        right=(1, 2, 3)
    ))
    insert_ai("size")
    story.append(Spacer(1, 3 * mm))

    # ======================================================
    # 10 WEEKLY DISTRIBUTION
    # ======================================================
    story.append(SectionHead("10 - Weekly Flow", "Weekly Listing Distribution"))
    if not weekly_df.empty:
        story.append(RLImage(chart_weekly(weekly_df), width=AVAIL, height=_img_height(WEEKLY_FS)))
        story.append(Paragraph("Weekly breakdown of capital mobilised during the report month. "
                               "Highlighted bar = highest mobilisation week.", caption_style))
    else:
        story.append(Paragraph("No weekly distribution data available.", caption_style))
    story.append(Spacer(1, 5 * mm))

    # ======================================================
    # 11 MAIN BOARD LISTING
    # ======================================================
    story.append(SectionHead("11 - Main Board", f"Main Board IPOs ({len(main_board_df)})"))
    if not main_board_df.empty:
        story.append(board_listing_table(main_board_df.sort_values("size_cr", ascending=False)))
    else:
        story.append(Paragraph("No Main Board IPOs listed this month.", caption_style))
    story.append(Spacer(1, 5 * mm))

    # ======================================================
    # 12 SME LISTING
    # ======================================================
    story.append(SectionHead("12 - SME", f"SME IPOs ({len(sme_df)})"))
    if not sme_df.empty:
        story.append(board_listing_table(sme_df.sort_values("size_cr", ascending=False)))
    else:
        story.append(Paragraph("No SME IPOs listed this month.", caption_style))
    story.append(Spacer(1, 5 * mm))

    # ======================================================
    # 13 LEAGUE TABLES & CONCENTRATION
    # ======================================================
    story.append(SectionHead("13 - League Tables", "Top Issues, Bankers & Market Concentration"))
    story.append(Paragraph("<b>Notable issues this month</b>", caption_style))
    story.append(Spacer(1, 1.5 * mm))
    top_issues = mdf.sort_values("size_cr", ascending=False).head(6) if not mdf.empty else pd.DataFrame()
    top_rows = []
    for _, row in top_issues.iterrows():
        top_rows.append([_esc(row.get("company_name", "")), _esc(row.get("board", "")),
                         _esc(row.get("sector", "")), _cr(row.get("size_cr", 0))])
    story.append(nice_table(
        ["Company", "Board", "Sector", "Size (Rs. Cr)"],
        top_rows,
        [AVAIL * x for x in (0.36, 0.18, 0.30, 0.16)],
        right=(3,)
    ))
    story.append(Spacer(1, 4 * mm))

    # Concentration metrics
    story.append(Paragraph("<b>Market concentration metrics</b>", caption_style))
    story.append(Spacer(1, 1.5 * mm))
    conc_rows = [
        ["Top 3 Issues Share", f"{top3_share:.1f}%", f"Rs. {_cr(top3_raised)} Cr"],
        ["Top 5 Issues Share", f"{top5_share:.1f}%", f"Rs. {_cr(top5_raised)} Cr"],
        ["HHI Index (0-10,000)", f"{hhi:.0f}", "Concentration" + (" High" if hhi > 2500 else " Moderate" if hhi > 1500 else " Low")],
        ["Largest Issue", largest_name, f"Rs. {_cr(largest_size)} Cr"],
        ["Smallest Issue", smallest_name, f"Rs. {_cr(smallest_size)} Cr"],
    ]
    story.append(nice_table(
        ["Metric", "Value", "Detail"],
        conc_rows,
        [AVAIL * x for x in (0.40, 0.25, 0.35)],
        right=(1, 2),
        bold_cols=(0,)
    ))
    story.append(Spacer(1, 4 * mm))

    # Merchant banker league table
    story.append(Paragraph("<b>Merchant banker league table (by mobilisation)</b>", caption_style))
    story.append(Spacer(1, 1.5 * mm))
    if not mdf.empty and "merchant_banker" in mdf.columns:
        banker_df = mdf.copy()
        banker_df["merchant_banker"] = banker_df["merchant_banker"].astype(str)
        banker_df = banker_df.assign(banker=banker_df["merchant_banker"].str.split(r",|;|&")).explode("banker")
        banker_df["banker"] = banker_df["banker"].astype(str).str.strip()
        banker_df = banker_df[~banker_df["banker"].str.lower().isin(["", "nan", "none", "null"])]
        if not banker_df.empty:
            banker_df = (banker_df.groupby("banker").agg(n=("size_cr", "size"), r=("size_cr", "sum"))
                         .sort_values("r", ascending=False).head(10).reset_index())
        else:
            banker_df = pd.DataFrame(columns=["banker", "n", "r"])
    else:
        banker_df = pd.DataFrame(columns=["banker", "n", "r"])
    banker_rows = []
    for _, row in banker_df.iterrows():
        banker_rows.append([_esc(row.get("banker", "")), str(int(float(row.get("n", 0) or 0))), _cr(row.get("r", 0))])
    story.append(nice_table(
        ["Merchant Banker", "Issues", "Raised (Rs. Cr)"],
        banker_rows,
        [AVAIL * x for x in (0.60, 0.15, 0.25)],
        right=(1, 2)
    ))
    story.append(Spacer(1, 5 * mm))

    # ======================================================
    # 14 AI STRATEGIC OUTLOOK
    # ======================================================
    story.append(SectionHead("14 - AI Intelligence", "AI Strategic Outlook"))
    if ai_sections:
        exec_blocks = find_ai_section(ai_sections, "executive")
        if exec_blocks:
            story.append(Paragraph("EXECUTIVE SUMMARY", ai_heading_style))
            for b in exec_blocks:
                txt = _md_to_html(b["text"])
                if b["type"] == "bullet":
                    story.append(Paragraph(f'<font color="{ACCENT_HEX}">&bull;</font>&nbsp;&nbsp;{_esc_html(txt)}', ai_bullet_style))
                else:
                    story.append(Paragraph(_esc_html(txt), ai_para_style))
        insert_ai("risk", "watchpoint")
        insert_ai("outlook", "forward")
    else:
        story.append(Paragraph(
            "AI analysis is unavailable. Set the <b>GROQ_API_KEY</b> environment variable "
            "(or enter the key in the dashboard) to enable AI-driven market intelligence.",
            caption_style))
    story.append(Spacer(1, 5 * mm))

    # ======================================================
    # 15 METHODOLOGY & DISCLAIMER
    # ======================================================
    story.append(SectionHead("15 - Methodology", "Data Sources & Methodology"))
    story.append(Paragraph(
        "This report aggregates publicly available IPO data from IPOPlatform.com tracker database. "
        "Figures are presented in Indian Rupees Crore (Rs. Cr). Issues with undisclosed sizes are excluded from totals. "
        "Month-on-month comparisons use calendar months. Year-on-year comparisons match the same calendar month in the prior year. "
        "Weekly distribution is based on ISO calendar weeks of listing dates. Market concentration is measured using the "
        "Herfindahl-Hirschman Index (HHI) on issue-size shares. AI commentary is generated by a large language model and is "
        "intended for informational purposes only; it does not constitute investment advice or an offer to sell securities.",
        body
    ))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        f"Source: IPOPlatform.com tracker database. Compiled and presented by {BRAND_NAME}. "
        "All rights reserved. Redistribution without written permission is prohibited.",
        note_style
    ))

    doc.build(story)
    return buffer.getvalue()