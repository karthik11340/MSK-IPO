import io
import os
import re
import numpy as np
import pandas as pd
import requests
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


# ==========================================================
# BRAND CONFIG  ->  MS KAPITAL (BLUE THEME)
# ==========================================================
BRAND_NAME = "MS KAPITAL"
BRAND_TAG = "IPO Market Intelligence  •  Capital Markets Research  •  Investment Advisory"

INK_HEX     = "#12263F"
ACCENT_HEX  = "#2563EB"
BLUE2_HEX   = "#7FA8E0"
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
BOARD_FS  = (7.4, 3.2)
SECTOR_FS = (7.4, 3.2)
SIZE_FS   = (7.4, 2.6)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#999999",
    "axes.labelcolor": "#555555",
    "figure.facecolor": "white"
})


# ==========================================================
# GROQ CONFIG  (AI trend-analysis commentary)
# ==========================================================
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

YOY_SYSTEM_PROMPT = (
    "You are a capital-markets analyst writing a short year-on-year commentary "
    "for an annual IPO market report published by MS KAPITAL. Use ONLY the "
    "figures provided - never invent numbers, company names, or events not in "
    "the data. Write one short paragraph (70-110 words) on how mobilisation, "
    "issue count and the Main Board / SME mix changed versus the prior year, "
    "and what that shift suggests about the market's direction. Neutral, "
    "analytical tone. No buy/sell language or investment recommendations. "
    "Format your response in markdown sections: start with '# Executive Summary', "
    "then include '# Risk & Watchpoints' and '# Forward Outlook' sections. "
    "Use bullet points (- ) under each section. Keep each section concise."
)

MULTIYEAR_SYSTEM_PROMPT = (
    "You are a capital-markets analyst writing the closing multi-year trend "
    "analysis for an annual IPO market report published by MS KAPITAL. Use "
    "ONLY the figures provided - never invent numbers, company names, or "
    "events not in the data. Write exactly two short paragraphs (roughly "
    "100-150 words total), separated by a blank line: the first on the "
    "trajectory of capital mobilisation and issue count across the years "
    "given - note any acceleration, slowdown, peak or trough; the second on "
    "where the most recent year sits within that multi-year context. "
    "Neutral, analytical tone. No buy/sell language or investment "
    "recommendations. Format your response in markdown sections: start with "
    "'# Executive Summary', then include '# Risk & Watchpoints' and "
    "'# Forward Outlook' sections. Use bullet points (- ) under each section."
)


# ==========================================================
# HELPERS
# ==========================================================
def _cr(value):
    try:
        if value is None or pd.isna(value):
            return "0"
        return f"{float(value):,.0f}"
    except Exception:
        return "0"


def _pdf_safe(text):
    """Converts Unicode characters that Helvetica/WinAnsi cannot print."""
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
    """Plain text escape - no HTML allowed."""
    if value is None:
        return ""
    return escape(_pdf_safe(str(value)))


def _esc_html(value):
    """Escape text but preserve allowed HTML formatting tags."""
    if value is None:
        return ""
    s = _pdf_safe(str(value))
    allowed_tags = []
    tag_pattern = re.compile(r'<(/?)(b|i|br)(\s*/?>|\s*>)|<(/?)font(\s[^>]*)?>')
    def store_tag(match):
        allowed_tags.append(match.group(0))
        return f"##TAG_{len(allowed_tags)-1}##"
    s = tag_pattern.sub(store_tag, s)
    s = escape(s)
    for i, tag in enumerate(allowed_tags):
        s = s.replace(f"##TAG_{i}##", tag)
    return s


def _md_to_html(text):
    """Convert markdown bold **text** and italic *text* to HTML font tags."""
    if not text:
        return text
    text = re.sub(r'\*\*(.+?)\*\*', r'<font name="Helvetica-Bold"></font>', text)
    text = re.sub(r'\*(.+?)\*', r'<i></i>', text)
    return text


def bold(text):
    """EXPLICIT BOLD - Reliable across all reportlab environments."""
    return f'<font name="Helvetica-Bold">{_esc(text)}</font>'


def bold_html(text):
    """Explicit bold for text that may contain other HTML."""
    return f'<font name="Helvetica-Bold">{_esc_html(text)}</font>'


def _img_height(figsize):
    return AVAIL * (figsize[1] / figsize[0])


def _growth(curr, prev):
    try:
        if prev in (0, None) or np.isnan(prev):
            return "—"
        g = (curr - prev) / prev * 100.0
        sign = "+" if g >= 0 else ""
        return f"{sign}{g:.1f}%"
    except Exception:
        return "—"


def _growth_html(curr, prev):
    """Growth with explicit color coding."""
    try:
        if prev in (0, None) or np.isnan(prev):
            return "—"
        g = (curr - prev) / prev * 100.0
        color = ACCENT_HEX if g >= 0 else "#DC2626"
        arrow = "▲" if g >= 0 else "▼"
        return f'<font color="{color}">{arrow} {"+" if g >= 0 else ""}{g:.1f}%</font>'
    except Exception:
        return "—"


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
    def __init__(self, year_label):
        Flowable.__init__(self)
        self.year_label = year_label
        self.height = 52 * mm

    def wrap(self, aw, ah):
        return aw, self.height

    def draw(self):
        c = self.canv
        top = self.height
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(ACCENT)
        c.drawString(0, top - 6 * mm, "ANNUAL CAPITAL MARKETS REVIEW")
        c.setFont("Helvetica-Bold", 30)
        c.setFillColor(INK)
        c.drawString(0, top - 18 * mm, "IPO YEARLY ANALYSIS")
        c.setFont("Helvetica", 11)
        c.setFillColor(GREY)
        c.drawString(0, top - 26 * mm, f"In-Depth Annual Market Study  —  Calendar Year {self.year_label}")
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
        self.stats = stats
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


def _style_twin_axes(ax1, ax2):
    for a in (ax1, ax2):
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
    ax1.spines["bottom"].set_color("#BBBBBB")
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, color="#EEEEEE")


def chart_monthly_progress(monthly):
    fig, ax1 = plt.subplots(figsize=TREND_FS, dpi=160)
    if monthly.empty:
        return _png(fig)
    monthly = monthly.copy()
    monthly["raised"] = pd.to_numeric(monthly["raised"], errors="coerce").fillna(0)
    monthly["count"] = pd.to_numeric(monthly["count"], errors="coerce").fillna(0)
    x = np.arange(len(monthly))
    ax1.bar(x, monthly["raised"], color=INK_HEX, width=0.62, zorder=3)
    for i, v in enumerate(monthly["raised"].values):
        ax1.annotate(_cr(v), (i, v), ha="center", va="bottom", fontsize=6, color="#555555")
    ax2 = ax1.twinx()
    ax2.plot(x, monthly["count"].values, color=ACCENT_HEX, marker="o", ms=4, lw=1.8, zorder=5)
    labels = pd.to_datetime(monthly["period"].astype(str) + "-01", errors="coerce").dt.strftime("%b")
    labels = labels.fillna(monthly["period"].astype(str))
    ax1.set_xticks(x); ax1.set_xticklabels(labels.tolist(), fontsize=7)
    ax1.set_ylabel("Rs. Cr", fontsize=7.5)
    ax2.set_ylabel("IPO Count", fontsize=7.5, color=ACCENT_HEX)
    ax2.tick_params(labelsize=7, colors=ACCENT_HEX)
    _style_twin_axes(ax1, ax2)
    return _png(fig)


def chart_multi_year(yearly, highlight_year=None):
    fig, ax1 = plt.subplots(figsize=TREND_FS, dpi=160)
    if yearly.empty:
        return _png(fig)
    yearly = yearly.copy()
    yearly["raised"] = pd.to_numeric(yearly["raised"], errors="coerce").fillna(0)
    yearly["count"] = pd.to_numeric(yearly["count"], errors="coerce").fillna(0)
    x = np.arange(len(yearly))
    years = yearly["year"].astype(str).tolist()
    bar_colors = [ACCENT_HEX if (highlight_year and y == str(highlight_year)) else INK_HEX for y in years]
    ax1.bar(x, yearly["raised"], color=bar_colors, width=0.6, zorder=3)
    for i, v in enumerate(yearly["raised"].values):
        ax1.annotate(_cr(v), (i, v), ha="center", va="bottom", fontsize=6, color="#555555")
    ax2 = ax1.twinx()
    ax2.plot(x, yearly["count"].values, color=ACCENT_HEX, marker="o", ms=4, lw=1.8, zorder=5)
    ax1.set_xticks(x); ax1.set_xticklabels(years, fontsize=7)
    ax1.set_ylabel("Rs. Cr", fontsize=7.5)
    ax2.set_ylabel("IPO Count", fontsize=7.5, color=ACCENT_HEX)
    ax2.tick_params(labelsize=7, colors=ACCENT_HEX)
    _style_twin_axes(ax1, ax2)
    return _png(fig)


def chart_boards_bar(board_df):
    """BAR CHART (replaces pie chart) - Main Board vs SME mobilisation."""
    fig, ax = plt.subplots(figsize=BOARD_FS, dpi=160)
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
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EEEEEE")
    return _png(fig)


def chart_sectors_year(sector_df):
    fig, ax = plt.subplots(figsize=SECTOR_FS, dpi=160)
    if sector_df.empty:
        return _png(fig)
    val_col = "raised" if "raised" in sector_df.columns else "r"
    label_col = "sector" if "sector" in sector_df.columns else sector_df.columns[0]
    sector_df = sector_df.copy()
    sector_df[val_col] = pd.to_numeric(sector_df[val_col], errors="coerce").fillna(0)
    sector_df = sector_df.sort_values(val_col, ascending=True).tail(10)
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


# ==========================================================
# DATA AGGREGATION
# ==========================================================
def clean_dataset(df):
    if df is None:
        df = pd.DataFrame()
    d = df.copy()
    required = ["period", "size_cr", "board", "sector", "company_name", "merchant_banker", "city"]
    for col in required:
        if col not in d.columns:
            d[col] = ""
    text_columns = ["board", "sector", "company_name", "merchant_banker", "city"]
    for col in text_columns:
        d[col] = d[col].fillna("").astype(str).str.strip().replace("nan", "")
    d["sector"] = d["sector"].replace("", "Unknown")
    d["board"] = d["board"].replace("", "Unknown")
    d["size_cr"] = pd.to_numeric(d["size_cr"], errors="coerce").fillna(0)
    d["period"] = d["period"].astype(str).str.strip()
    invalid = ["", "nan", "nat", "none", "null"]
    d = d[~d["period"].str.lower().isin(invalid)].copy()
    return d


def year_metrics(d, year):
    sub = d[d["period"].astype(str).str[:4] == str(year)]
    mb = sub[sub["board"] == "Main Board"]
    sme = sub[sub["board"] == "SME"]
    return {
        "count": int(len(sub)),
        "raised": float(sub["size_cr"].sum()) if len(sub) else 0.0,
        "mb_count": int(len(mb)),
        "mb_raised": float(mb["size_cr"].sum()) if len(mb) else 0.0,
        "sme_count": int(len(sme)),
        "sme_raised": float(sme["size_cr"].sum()) if len(sme) else 0.0,
        "avg": float(sub["size_cr"].mean()) if len(sub) else 0.0,
        "median": float(sub["size_cr"].median()) if len(sub) else 0.0,
        "largest": float(sub["size_cr"].max()) if len(sub) else 0.0,
        "sectors": int(sub["sector"].replace("", np.nan).nunique()),
        "cities": int(sub["city"].replace("", np.nan).nunique()),
    }


# ==========================================================
# AI TREND ANALYSIS  (GROQ)
# ==========================================================
def _groq_chat(user_prompt, system=None, model=None, api_key=None,
               temperature=0.4, max_tokens=450, timeout=20):
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_prompt})
    resp = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or GROQ_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _yoy_data_summary(ctx):
    return (
        f"Current year: {ctx['year']}\n"
        f"Current year totals: {ctx['count']} IPOs, Rs. {_cr(ctx['raised'])} Cr raised\n"
        f"Previous year ({ctx['prev_year']}) totals: {ctx['prev_count']} IPOs, "
        f"Rs. {_cr(ctx['prev_raised'])} Cr raised\n"
        f"Main Board {ctx['year']}: {ctx['mb_count']} issues, Rs. {_cr(ctx['mb_raised'])} Cr "
        f"(prior year: {ctx['prev_mb_count']} issues, Rs. {_cr(ctx['prev_mb_raised'])} Cr)\n"
        f"SME {ctx['year']}: {ctx['sme_count']} issues, Rs. {_cr(ctx['sme_raised'])} Cr "
        f"(prior year: {ctx['prev_sme_count']} issues, Rs. {_cr(ctx['prev_sme_raised'])} Cr)\n"
        f"Average issue size {ctx['year']}: Rs. {_cr(ctx['avg'])} Cr "
        f"(prior year: Rs. {_cr(ctx['prev_avg'])} Cr)\n\n"
        "Write the year-on-year commentary using only these figures."
    )


def _fallback_yoy_analysis(ctx):
    raised_g = _growth(ctx["raised"], ctx["prev_raised"])
    count_g = _growth(ctx["count"], ctx["prev_count"])
    text = (
        f"# Executive Summary\n"
        f"Capital mobilisation in {ctx['year']} moved {raised_g} year-on-year to "
        f"Rs. {_cr(ctx['raised'])} Cr, while issue count moved {count_g} to "
        f"{ctx['count']} IPOs (from {ctx['prev_count']} in {ctx['prev_year']}).\n\n"
    )
    if ctx["count"]:
        sme_share = ctx["sme_count"] / ctx["count"] * 100
        text += (
            f"# Risk & Watchpoints\n"
            f"- SME issues accounted for {sme_share:.0f}% of total issue count in {ctx['year']}.\n\n"
        )
    text += (
        f"# Forward Outlook\n"
        f"- Monitor Main Board pipeline and SME platform traction for directional signals."
    )
    return text


def generate_yoy_analysis(ctx, api_key=None, model=None):
    try:
        summary = _yoy_data_summary(ctx)
        text = _groq_chat(summary, system=YOY_SYSTEM_PROMPT, api_key=api_key, model=model)
        if not text:
            raise RuntimeError("Empty response from Groq.")
        return text, True, (model or GROQ_MODEL)
    except Exception:
        return _fallback_yoy_analysis(ctx), False, None


def _multiyear_data_summary(ctx):
    lines = []
    for _, row in ctx["multi_year"].iterrows():
        lines.append(f"- {row['year']}: {int(row['count'])} IPOs, Rs. {_cr(row['raised'])} Cr raised")
    history_block = "\n".join(lines) if lines else "No multi-year history available."
    return (
        f"Most recent year in this report: {ctx['year']}\n\n"
        f"Yearly history (oldest to newest):\n{history_block}\n\n"
        "Write the multi-year trend analysis using only these figures."
    )


def _fallback_multiyear_analysis(ctx):
    my = ctx["multi_year"]
    if my.empty or len(my) < 2:
        return (
            f"# Executive Summary\n"
            f"Limited historical data is available prior to {ctx['year']}; a longer track "
            "record will allow richer multi-year trend analysis in future editions.\n\n"
            f"# Forward Outlook\n"
            f"- Build multi-year dataset for deeper trajectory analysis."
        )
    first = my.iloc[0]
    last = my.iloc[-1]
    peak = my.loc[my["raised"].idxmax()]
    total_g = _growth(last["raised"], first["raised"])
    return (
        f"# Executive Summary\n"
        f"- Over the {len(my)}-year period from {first['year']} to {last['year']}, annual capital "
        f"mobilisation moved from Rs. {_cr(first['raised'])} Cr to Rs. {_cr(last['raised'])} Cr "
        f"({total_g} cumulative), with {peak['year']} the strongest year on record at "
        f"Rs. {_cr(peak['raised'])} Cr raised.\n"
        f"- {ctx['year']} closed the period with {int(last['count'])} IPOs raising "
        f"Rs. {_cr(last['raised'])} Cr, the most recent data point in this trend.\n\n"
        f"# Risk & Watchpoints\n"
        f"- Watch for concentration risk if peak years were driven by a few mega issues.\n\n"
        f"# Forward Outlook\n"
        f"- Compare upcoming pipeline against the {peak['year']} peak to gauge market health."
    )


def generate_multiyear_analysis(ctx, api_key=None, model=None):
    try:
        summary = _multiyear_data_summary(ctx)
        text = _groq_chat(summary, system=MULTIYEAR_SYSTEM_PROMPT, api_key=api_key, model=model)
        if not text:
            raise RuntimeError("Empty response from Groq.")
        return text, True, (model or GROQ_MODEL)
    except Exception:
        return _fallback_multiyear_analysis(ctx), False, None


# ==========================================================
# MAIN YEARLY PDF BUILDER
# ==========================================================
def build_yearly_pdf(df, year, groq_api_key=None, groq_model=None, include_ai_trend=True):
    d = clean_dataset(df)
    year = str(year)
    try:
        prev_year = str(int(year) - 1)
    except Exception:
        prev_year = ""

    ydf = d[d["period"].astype(str).str[:4] == year].copy()
    curr = year_metrics(d, year)
    prev = year_metrics(d, prev_year) if prev_year else {}

    # Monthly progression within the year
    monthly = (
        ydf.groupby("period")
        .agg(raised=("size_cr", "sum"), count=("period", "size"))
        .reset_index().sort_values("period")
    ) if not ydf.empty else pd.DataFrame(columns=["period", "raised", "count"])

    # Sector analysis
    if not ydf.empty:
        sector_df = ydf.groupby("sector").agg(n=("size_cr", "size"), r=("size_cr", "sum")).reset_index()
        sector_df["n"] = pd.to_numeric(sector_df["n"], errors="coerce").fillna(0).astype(int)
        sector_df["r"] = pd.to_numeric(sector_df["r"], errors="coerce").fillna(0)
        sector_df = sector_df.sort_values("r", ascending=False)
        sector_df["share"] = (sector_df["r"] / curr["raised"] * 100) if curr["raised"] else 0
        sector_df["avg"] = sector_df["r"] / sector_df["n"].replace(0, np.nan)
    else:
        sector_df = pd.DataFrame(columns=["sector", "n", "r", "share", "avg"])

    # City analysis
    if not ydf.empty:
        city_df = ydf.groupby("city").agg(n=("size_cr", "size"), r=("size_cr", "sum")).reset_index()
        city_df["r"] = pd.to_numeric(city_df["r"], errors="coerce").fillna(0)
        city_df = city_df.sort_values("r", ascending=False).head(10)
        city_df["share"] = (city_df["r"] / curr["raised"] * 100) if curr["raised"] else 0
    else:
        city_df = pd.DataFrame(columns=["city", "n", "r", "share"])

    # Size distribution
    if not ydf.empty:
        ydf["_bucket"] = ydf["size_cr"].apply(_size_bucket)
        size_df = (ydf.groupby("_bucket").agg(n=("size_cr", "size"), r=("size_cr", "sum"))
                   .reindex(BUCKET_ORDER).fillna(0).reset_index().rename(columns={"_bucket": "bucket"}))
        size_df["n"] = size_df["n"].astype(int)
    else:
        size_df = pd.DataFrame({"bucket": BUCKET_ORDER, "n": [0] * 5, "r": [0.0] * 5})

    # Merchant bankers
    if not ydf.empty and "merchant_banker" in ydf.columns:
        banker_df = ydf.copy()
        banker_df["merchant_banker"] = banker_df["merchant_banker"].astype(str)
        banker_df = banker_df.assign(banker=banker_df["merchant_banker"].str.split(r",|;|&")).explode("banker")
        banker_df["banker"] = banker_df["banker"].astype(str).str.strip()
        banker_df = banker_df[~banker_df["banker"].str.lower().isin(["", "nan", "none", "null"])]
        if not banker_df.empty:
            banker_df = (banker_df.groupby("banker").agg(n=("size_cr", "size"), r=("size_cr", "sum"))
                         .sort_values("r", ascending=False).head(15).reset_index())
        else:
            banker_df = pd.DataFrame(columns=["banker", "n", "r"])
    else:
        banker_df = pd.DataFrame(columns=["banker", "n", "r"])

    # Top issues
    top_issues = ydf.sort_values("size_cr", ascending=False).head(20) if not ydf.empty else pd.DataFrame()

    # Multi-year context (trailing 5 years)
    if not d.empty:
        d2 = d.copy()
        d2["year"] = d2["period"].astype(str).str[:4]
        multi_year = (d2.groupby("year").agg(raised=("size_cr", "sum"), count=("period", "size"))
                      .reset_index().sort_values("year").tail(5))
    else:
        multi_year = pd.DataFrame(columns=["year", "raised", "count"])

    # Market concentration metrics
    top3_raised = float(ydf.sort_values("size_cr", ascending=False).head(3)["size_cr"].sum()) if not ydf.empty else 0.0
    top5_raised = float(ydf.sort_values("size_cr", ascending=False).head(5)["size_cr"].sum()) if not ydf.empty else 0.0
    top3_share = (top3_raised / curr["raised"] * 100) if curr["raised"] else 0.0
    top5_share = (top5_raised / curr["raised"] * 100) if curr["raised"] else 0.0
    hhi = float((ydf["size_cr"] / curr["raised"] * 100).pow(2).sum()) if curr["raised"] and not ydf.empty else 0.0
    largest_issue = ydf.sort_values("size_cr", ascending=False).head(1) if not ydf.empty else pd.DataFrame()
    smallest_issue = ydf.sort_values("size_cr", ascending=True).head(1) if not ydf.empty else pd.DataFrame()
    largest_name = _esc(largest_issue["company_name"].iloc[0]) if len(largest_issue) else "-"
    smallest_name = _esc(smallest_issue["company_name"].iloc[0]) if len(smallest_issue) else "-"
    largest_size = float(largest_issue["size_cr"].iloc[0]) if len(largest_issue) else 0.0
    smallest_size = float(smallest_issue["size_cr"].iloc[0]) if len(smallest_issue) else 0.0

    # ---------- AI Generation ----------
    yoy_text, yoy_used_ai, yoy_model = None, False, None
    my_text, my_used_ai, my_model = None, False, None

    if include_ai_trend and prev:
        yoy_ctx = {
            "year": year, "prev_year": prev_year,
            "count": curr["count"], "raised": curr["raised"],
            "prev_count": prev["count"], "prev_raised": prev["raised"],
            "mb_count": curr["mb_count"], "mb_raised": curr["mb_raised"],
            "prev_mb_count": prev["mb_count"], "prev_mb_raised": prev["mb_raised"],
            "sme_count": curr["sme_count"], "sme_raised": curr["sme_raised"],
            "prev_sme_count": prev["sme_count"], "prev_sme_raised": prev["sme_raised"],
            "avg": curr["avg"], "prev_avg": prev["avg"],
        }
        yoy_text, yoy_used_ai, yoy_model = generate_yoy_analysis(yoy_ctx, api_key=groq_api_key, model=groq_model)

    if include_ai_trend and not multi_year.empty:
        multiyear_ctx = {"year": year, "multi_year": multi_year}
        my_text, my_used_ai, my_model = generate_multiyear_analysis(multiyear_ctx, api_key=groq_api_key, model=groq_model)

    yoy_sections = parse_ai_sections(yoy_text) if yoy_text else {}
    my_sections = parse_ai_sections(my_text) if my_text else {}

    # ---------- Build document ----------
    buffer = io.BytesIO()
    doc = BaseDocTemplate(buffer, pagesize=A4, leftMargin=M_L, rightMargin=M_R, topMargin=M_T, bottomMargin=M_B,
                          title=f"IPO Yearly Analysis - {year}")

    def chrome(canvas, doc_obj):
        canvas.saveState()
        canvas.setFillColor(ACCENT)
        canvas.rect(M_L, H - 11 * mm, 3 * mm, 3 * mm, stroke=0, fill=1)
        canvas.setFillColor(INK)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(M_L + 5 * mm, H - 10.6 * mm, BRAND_NAME)
        canvas.setFillColor(GREY)
        canvas.setFont("Helvetica", 7.2)
        canvas.drawRightString(W - M_R, H - 10.4 * mm, "IPO YEARLY ANALYSIS  |  ANNUAL CAPITAL MARKETS REVIEW")
        canvas.setStrokeColor(colors.HexColor("#D8DEE7")); canvas.setLineWidth(0.6)
        canvas.line(M_L, H - 13 * mm, W - M_R, H - 13 * mm)
        canvas.setFont("Helvetica", 6.8); canvas.setFillColor(GREY)
        canvas.drawString(M_L, 9 * mm, BRAND_TAG)
        canvas.drawRightString(W - M_R, 9 * mm, f"Page {doc_obj.page}")
        canvas.restoreState()

    frame = Frame(M_L, M_B, AVAIL, H - M_T - M_B)
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=chrome)])

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

    def insert_ai(ai_sections, *keywords):
        blocks = find_ai_section(ai_sections, *keywords)
        if not blocks:
            return
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph('<font color="#2563EB">&raquo;</font>&nbsp;&nbsp;AI INSIGHT', ai_label_style))
        for block in blocks:
            txt = _md_to_html(block["text"])
            if block["type"] == "heading":
                story.append(Paragraph(_esc_html(txt), ai_heading_style))
            elif block["type"] == "bullet":
                story.append(Paragraph(f'<font color="{ACCENT_HEX}">&bull;</font>&nbsp;&nbsp;{_esc_html(txt)}', ai_bullet_style))
            else:
                story.append(Paragraph(_esc_html(txt), ai_para_style))
        story.append(Spacer(1, 3 * mm))

    # ======================================================
    # COVER PAGE
    # ======================================================
    story.append(CoverHead(year))
    cover_stats = [
        ("Total IPOs", curr["count"]),
        ("Mobilised", f"Rs. {_cr(curr['raised'])} Cr"),
        ("Main Board", curr["mb_count"]),
        ("SME Board", curr["sme_count"]),
    ]
    story.append(KeyStatsBox(cover_stats))
    story.append(Spacer(1, 4 * mm))

    exec_summary = (
        f"Calendar Year {bold(year)} witnessed {bold(str(curr['count']))} IPOs mobilising "
        f"{bold(f'Rs. {_cr(curr["raised"])} Cr')} in aggregate. The Main Board contributed "
        f"{bold(str(curr['mb_count']))} issues raising {bold(f'Rs. {_cr(curr["mb_raised"])} Cr')}, "
        f"while the SME platform listed {bold(str(curr['sme_count']))} issues raising "
        f"{bold(f'Rs. {_cr(curr["sme_raised"])} Cr')}. This annual review provides an in-depth analysis "
        f"of market momentum, board dynamics, sector and geographic concentration, issue-size distribution, "
        f"intermediary activity, market concentration metrics and the year's landmark issues."
    )
    story.append(Paragraph(exec_summary, body))
    story.append(Spacer(1, 6 * mm))

    # ======================================================
    # 01 SNAPSHOT
    # ======================================================
    kpis = [
        (f"Rs. {_cr(curr['raised'])} Cr", "Total Mobilised"),
        (f"{curr['count']}", "Total IPOs"),
        (f"Rs. {_cr(curr['avg'])} Cr", "Average Issue Size"),
        (f"Rs. {_cr(curr['median'])} Cr", "Median Issue Size"),
        (f"{curr['mb_count']}  •  Rs. {_cr(curr['mb_raised'])} Cr", "Main Board (Issues / Mobilised)"),
        (f"{curr['sme_count']}  •  Rs. {_cr(curr['sme_raised'])} Cr", "SME (Issues / Mobilised)"),
        (f"Rs. {_cr(curr['largest'])} Cr", "Largest Issue"),
        (f"{curr['sectors']}", "Active Sectors"),
        (f"{curr['cities']}", "Active Cities"),
    ]
    story.append(KeepTogether([
        SectionHead("01 — Snapshot", f"Annual Snapshot: {year}"),
        Spacer(1, 3 * mm),
        kpi_table(kpis),
        Spacer(1, 5 * mm),
    ]))

    # ======================================================
    # 02 YEAR-ON-YEAR
    # ======================================================
    story.append(SectionHead("02 — Year-on-Year", f"Growth Analysis: {prev_year} vs {year}"))
    if prev:
        yoy_rows = [
            ["Total IPOs", str(prev["count"]), str(curr["count"]), _growth(curr["count"], prev["count"])],
            ["Total Mobilised (Rs. Cr)", _cr(prev["raised"]), _cr(curr["raised"]), _growth(curr["raised"], prev["raised"])],
            ["Main Board IPOs", str(prev["mb_count"]), str(curr["mb_count"]), _growth(curr["mb_count"], prev["mb_count"])],
            ["SME IPOs", str(prev["sme_count"]), str(curr["sme_count"]), _growth(curr["sme_count"], prev["sme_count"])],
            ["Average Issue Size (Rs. Cr)", _cr(prev["avg"]), _cr(curr["avg"]), _growth(curr["avg"], prev["avg"])],
            ["Median Issue Size (Rs. Cr)", _cr(prev["median"]), _cr(curr["median"]), _growth(curr["median"], prev["median"])],
        ]
        story.append(nice_table(
            ["Metric", prev_year, year, "YoY Change"],
            yoy_rows,
            [AVAIL * x for x in (0.40, 0.18, 0.18, 0.24)],
            right=(1, 2, 3)
        ))

        yoy_bullets = []
        raised_g = _growth(curr["raised"], prev["raised"])
        count_g = _growth(curr["count"], prev["count"])
        yoy_bullets.append(f"Capital mobilisation moved {bold(raised_g)} year-on-year, while issue count moved {bold(count_g)}.")
        if curr["count"]:
            sme_share_count = curr["sme_count"] / curr["count"] * 100
            yoy_bullets.append(f"SME issues accounted for {bold(f'{sme_share_count:.0f}%')} of total issue count in {year}.")
        for bullet in yoy_bullets:
            story.append(Paragraph(f'<font color="{ACCENT_HEX}">■</font>&nbsp;&nbsp;{bullet}', bullet_style))

        if yoy_sections:
            insert_ai(yoy_sections, "executive")
            insert_ai(yoy_sections, "risk", "watchpoint")
            insert_ai(yoy_sections, "outlook", "forward")
        else:
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph(
                "AI commentary unavailable. Set GROQ_API_KEY to enable AI-driven year-on-year analysis.",
                caption_style))
    else:
        story.append(Paragraph("No prior-year data available for comparison.", caption_style))
    story.append(Spacer(1, 5 * mm))

    # ======================================================
    # 03 MONTHLY PROGRESSION
    # ======================================================
    story.append(SectionHead("03 — Monthly Progression", f"How {year} Unfolded, Month by Month"))
    if not monthly.empty:
        story.append(RLImage(chart_monthly_progress(monthly), width=AVAIL, height=_img_height(TREND_FS)))
        peak = monthly.loc[monthly["raised"].idxmax()]
        peak_label = pd.to_datetime(peak["period"] + "-01").strftime("%B")
        story.append(Paragraph(
            f"Bars: capital mobilised (Rs. Cr, left axis). Line: number of listings (right axis). "
            f"The strongest month by mobilisation was {bold(peak_label)} (Rs. {_cr(peak['raised'])} Cr).",
            caption_style))
    else:
        story.append(Paragraph("No monthly data available.", caption_style))
    story.append(Spacer(1, 5 * mm))

    # ======================================================
    # 04 BOARD ANALYSIS  (BAR CHART)
    # ======================================================
    if not ydf.empty:
        board_df = (ydf.groupby("board").agg(count=("size_cr", "size"), raised=("size_cr", "sum"))
                    .reindex(["Main Board", "SME"]).fillna(0))
    else:
        board_df = pd.DataFrame({"count": [0, 0], "raised": [0.0, 0.0]},
                                index=pd.Index(["Main Board", "SME"], name="board"))

    story.append(SectionHead("04 — Board Analysis", "Main Board vs SME"))
    if board_df["count"].sum() > 0:
        story.append(RLImage(chart_boards_bar(board_df), width=AVAIL, height=_img_height(BOARD_FS)))
    board_rows = []
    for board_name, row in board_df.iterrows():
        bc = int(float(row.get("count", 0) or 0)); br = float(row.get("raised", 0) or 0)
        bavg = br / bc if bc else 0; bshare = (br / curr["raised"] * 100) if curr["raised"] else 0
        board_rows.append([_esc(board_name), str(bc), _cr(br), _cr(bavg), f"{bshare:.1f}%"])
    story.append(nice_table(
        ["Board", "Issues", "Raised (Rs. Cr)", "Avg Size", "Share of Raised"],
        board_rows,
        [AVAIL * x for x in (0.30, 0.14, 0.24, 0.16, 0.16)],
        right=(1, 2, 3, 4)
    ))
    if curr["count"]:
        sme_count_share = curr["sme_count"] / curr["count"] * 100
        sme_raised_share = (curr["sme_raised"] / curr["raised"] * 100) if curr["raised"] else 0
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(
            f'<font color="{ACCENT_HEX}">■</font>&nbsp;&nbsp;The SME platform contributed {bold(f"{sme_count_share:.0f}%")} of issue '
            f'count but only {bold(f"{sme_raised_share:.0f}%")} of mobilisation, reflecting smaller ticket sizes relative to the Main Board.',
            bullet_style))
    story.append(Spacer(1, 5 * mm))

    # ======================================================
    # 05 SECTOR ANALYSIS
    # ======================================================
    story.append(SectionHead("05 — Sector Analysis", "Sectoral Distribution & Concentration"))
    if not sector_df.empty:
        story.append(RLImage(chart_sectors_year(sector_df), width=AVAIL, height=_img_height(SECTOR_FS)))
    sector_rows = []
    for _, row in sector_df.head(12).iterrows():
        sector_rows.append([
            _esc(row.get("sector", "")),
            str(int(float(row.get("n", 0) or 0))),
            _cr(row.get("r", 0)),
            f"{float(row.get('share', 0) or 0):.1f}%",
            _cr(row.get("avg", 0))
        ])
    story.append(nice_table(
        ["Sector", "IPOs", "Raised (Rs. Cr)", "Share %", "Avg Size"],
        sector_rows,
        [AVAIL * x for x in (0.34, 0.12, 0.20, 0.14, 0.20)],
        right=(1, 2, 3, 4)
    ))
    if not sector_df.empty and len(sector_df) >= 3:
        top3 = sector_df.head(3)
        top3_share = top3["share"].sum()
        top3_names = ", ".join([bold(_esc(s)) for s in top3["sector"].tolist()])
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(
            f'<font color="{ACCENT_HEX}">■</font>&nbsp;&nbsp;Sector concentration: the top three sectors ({top3_names}) '
            f'together account for {bold(f"{top3_share:.1f}%")} of total mobilisation in {year}.',
            bullet_style))
    story.append(Spacer(1, 5 * mm))

    # ======================================================
    # 06 GEOGRAPHY
    # ======================================================
    story.append(SectionHead("06 — Geography", "Top IPO Cities"))
    city_rows = []
    for _, row in city_df.iterrows():
        city_rows.append([
            _esc(row.get("city", "")),
            str(int(float(row.get("n", 0) or 0))),
            _cr(row.get("r", 0)),
            f"{float(row.get('share', 0) or 0):.1f}%"
        ])
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
    # 07 SIZE DISTRIBUTION
    # ======================================================
    story.append(SectionHead("07 — Issue Size", "Issue Size Distribution"))
    if curr["count"]:
        story.append(RLImage(chart_size_dist(size_df), width=AVAIL, height=_img_height(SIZE_FS)))
    size_rows = []
    for _, row in size_df.iterrows():
        size_rows.append([_esc(row.get("bucket", "")), str(int(float(row.get("n", 0) or 0))), _cr(row.get("r", 0))])
    story.append(nice_table(
        ["Size Band", "IPOs", "Raised (Rs. Cr)"],
        size_rows,
        [AVAIL * x for x in (0.52, 0.18, 0.30)],
        right=(1, 2)
    ))
    story.append(Spacer(1, 5 * mm))

    # ======================================================
    # 08 MARKET CONCENTRATION
    # ======================================================
    story.append(SectionHead("08 — Concentration", "Market Concentration & Records"))
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
    if curr["count"]:
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(
            f'<font color="{ACCENT_HEX}">■</font>&nbsp;&nbsp;The top 3 issues captured {bold(f"{top3_share:.1f}%")} of total '
            f'annual mobilisation. HHI of {bold(f"{hhi:.0f}")} indicates '
            f'{"high" if hhi > 2500 else "moderate" if hhi > 1500 else "low"} concentration.',
            bullet_style))
    story.append(Spacer(1, 5 * mm))

    # ======================================================
    # 09 MERCHANT BANKERS
    # ======================================================
    story.append(SectionHead("09 — Intermediaries", "Merchant Banker League Table"))
    banker_rows = []
    for _, row in banker_df.iterrows():
        banker_rows.append([
            _esc(row.get("banker", "")),
            str(int(float(row.get("n", 0) or 0))),
            _cr(row.get("r", 0))
        ])
    if banker_rows:
        story.append(nice_table(
            ["Merchant Banker", "Issues", "Raised (Rs. Cr)"],
            banker_rows,
            [AVAIL * x for x in (0.60, 0.15, 0.25)],
            right=(1, 2)
        ))
    else:
        story.append(Paragraph("No merchant banker data available.", caption_style))
    story.append(Spacer(1, 5 * mm))

    # ======================================================
    # 10 TOP ISSUES
    # ======================================================
    story.append(SectionHead("10 — Landmark Issues", f"Top IPOs of {year}"))
    top_rows = []
    for _, row in top_issues.iterrows():
        top_rows.append([
            _esc(row.get("company_name", "")),
            _esc(row.get("board", "")),
            _esc(row.get("sector", "")),
            _cr(row.get("size_cr", 0))
        ])
    if top_rows:
        story.append(nice_table(
            ["Company", "Board", "Sector", "Size (Rs. Cr)"],
            top_rows,
            [AVAIL * x for x in (0.36, 0.18, 0.30, 0.16)],
            right=(3,)
        ))
    else:
        story.append(Paragraph("No issue data available.", caption_style))
    story.append(Spacer(1, 5 * mm))

    # ======================================================
    # 11 MULTI-YEAR CONTEXT
    # ======================================================
    story.append(SectionHead("11 — Multi-Year Context", "Trailing 5-Year Market Trend"))
    if not multi_year.empty:
        story.append(RLImage(chart_multi_year(multi_year, year), width=AVAIL, height=_img_height(TREND_FS)))
        my_rows = []
        for _, row in multi_year.iterrows():
            my_rows.append([_esc(row["year"]), str(int(row["count"])), _cr(row["raised"])])
        story.append(nice_table(
            ["Year", "IPOs", "Mobilised (Rs. Cr)"],
            my_rows,
            [AVAIL * x for x in (0.34, 0.28, 0.38)],
            right=(1, 2)
        ))

        if my_sections:
            insert_ai(my_sections, "executive")
            insert_ai(my_sections, "risk", "watchpoint")
            insert_ai(my_sections, "outlook", "forward")
        else:
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph(
                "AI multi-year commentary unavailable. Set GROQ_API_KEY to enable AI-driven trend analysis.",
                caption_style))
    else:
        story.append(Paragraph("No multi-year data available.", caption_style))
    story.append(Spacer(1, 6 * mm))

    # ======================================================
    # 12 METHODOLOGY & DISCLAIMER
    # ======================================================
    story.append(SectionHead("12 — Methodology", "Data Sources & Methodology"))
    story.append(Paragraph(
        "This report aggregates publicly available IPO data from IPOPlatform.com tracker database. "
        "Figures are presented in Indian Rupees Crore (Rs. Cr). Issues with undisclosed sizes are excluded from totals. "
        "Year-on-year comparisons use calendar years. Market concentration is measured using the "
        "Herfindahl-Hirschman Index (HHI) on issue-size shares, where values above 2,500 indicate high concentration, "
        "1,500-2,500 moderate concentration, and below 1,500 low concentration. AI commentary is generated by a large "
        "language model and is intended for informational purposes only; it does not constitute investment advice or an offer to sell securities.",
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