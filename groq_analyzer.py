"""
groq_analyzer.py
================
AI engine for the MS Kapital IPO dashboard & PDF reports.

- Reads GROQ_API_KEY from the environment (or accepts a key argument).
- Tries several Groq models in order (so a deprecated model never breaks output).
- Returns rich, markdown-formatted market analysis for monthly & yearly reports.
- Run this file directly to test your connection:   python groq_analyzer.py
"""

import os
import numpy as np
import pandas as pd
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Models tried in order. If one is deprecated / fails, the next is used.
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "gemma2-9b-it",
]

LAST_ERROR = None
LAST_MODEL_USED = None


# ==========================================================
# API KEY
# ==========================================================
def get_groq_api_key():
    key = os.getenv("GROQ_API_KEY", "").strip()
    # Common copy-paste mistakes
    return key.replace("Bearer ", "").replace('"', "").replace("'", "").strip()


def get_last_error():
    return LAST_ERROR


def get_last_model():
    return LAST_MODEL_USED


# ==========================================================
# ERROR EXPLANATIONS
# ==========================================================
def _explain_status(code, body=""):
    if code == 401:
        return "Invalid API key (401). It should start with 'gsk_'."
    if code == 403:
        return "Access forbidden (403). Your key may be disabled."
    if code == 404:
        return "Model not found (404). The model is deprecated - trying another."
    if code == 413:
        return "Request too large (413). Reduce max_tokens / prompt size."
    if code == 429:
        return "Rate limit exceeded (429). Free-tier limit - wait a minute and retry."
    if code >= 500:
        return f"Groq server error ({code}). Try again shortly."
    return f"HTTP {code}: {body[:200]}"


# ==========================================================
# LOW-LEVEL CALL
# ==========================================================
def call_groq_single(prompt, system_prompt=None, api_key=None, model=None,
                     temperature=0.4, max_tokens=3000):
    """Single-model call. Returns (text, error_message)."""
    api_key = (api_key or get_groq_api_key()).strip()
    if not api_key:
        return None, "No API key found. Set GROQ_API_KEY or enter it in the dashboard."

    model = model or GROQ_MODELS[0]
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}

    try:
        r = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=90)
    except requests.exceptions.Timeout:
        return None, "Request timed out after 90s. Check internet / firewall."
    except requests.exceptions.ConnectionError:
        return None, "Could not reach api.groq.com. Check internet / proxy / firewall."
    except Exception as e:
        return None, f"Network error: {e}"

    if r.status_code != 200:
        return None, _explain_status(r.status_code, r.text)

    try:
        return r.json()["choices"][0]["message"]["content"].strip(), None
    except Exception as e:
        return None, f"Unexpected response format: {e}"


def call_groq(prompt, system_prompt=None, api_key=None,
              temperature=0.4, max_tokens=3000):
    """Tries every model in GROQ_MODELS until one works. Returns text or None."""
    global LAST_ERROR, LAST_MODEL_USED
    api_key = (api_key or get_groq_api_key()).strip()
    if not api_key:
        LAST_ERROR = "No API key found. Set GROQ_API_KEY or enter it in the dashboard."
        return None

    errors = []
    for model in GROQ_MODELS:
        text, err = call_groq_single(prompt, system_prompt, api_key, model,
                                     temperature, max_tokens)
        if text:
            LAST_ERROR = None
            LAST_MODEL_USED = model
            return text
        errors.append(f"[{model}] {err}")

    LAST_ERROR = " | ".join(errors)
    return None


# ==========================================================
# CONNECTION TEST  (python groq_analyzer.py)
# ==========================================================
def test_groq_connection(api_key=None):
    api_key = (api_key or get_groq_api_key()).strip()
    result = {
        "key_found": bool(api_key),
        "key_preview": (api_key[:7] + "..." + api_key[-4:]) if len(api_key) > 12 else "(too short)",
        "models_tried": [], "working_model": None, "sample_response": None, "errors": []
    }
    if not api_key:
        result["errors"].append("GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys")
        return result

    for model in GROQ_MODELS:
        text, err = call_groq_single("Reply with exactly: OK", api_key=api_key,
                                     model=model, temperature=0, max_tokens=10)
        if text:
            result["models_tried"].append({"model": model, "status": "WORKING"})
            result["working_model"] = model
            result["sample_response"] = text
            return result
        result["models_tried"].append({"model": model, "status": f"FAILED - {err}"})
        result["errors"].append(f"{model}: {err}")
    return result


# ==========================================================
# DATA HELPERS
# ==========================================================
def _cr(v):
    try:
        if v is None or np.isnan(v): return "0"
        return f"{float(v):,.0f}"
    except Exception:
        return "0"


def _growth(c, p):
    try:
        if p in (0, None) or np.isnan(p): return "N/A"
        g = (c - p) / p * 100.0
        return f"{'+' if g >= 0 else ''}{g:.1f}%"
    except Exception:
        return "N/A"


def _month_label(p):
    try:
        return pd.Period(p, freq="M").strftime("%B %Y")
    except Exception:
        return str(p)


def _clean_dataset(df):
    if df is None: df = pd.DataFrame()
    d = df.copy()
    for col in ["period", "size_cr", "board", "sector", "company_name", "city"]:
        if col not in d.columns: d[col] = ""
    d["size_cr"] = pd.to_numeric(d["size_cr"], errors="coerce").fillna(0)
    d["period"] = d["period"].astype(str).str.strip()
    return d[~d["period"].str.lower().isin(["", "nan", "nat", "none", "null"])].copy()


def _period_stats(sub):
    n = len(sub)
    mb = sub[sub["board"] == "Main Board"]
    sme = sub[sub["board"] == "SME"]
    return {
        "count": n, "raised": float(sub["size_cr"].sum()) if n else 0.0,
        "mb_count": len(mb), "mb_raised": float(mb["size_cr"].sum()) if len(mb) else 0.0,
        "sme_count": len(sme), "sme_raised": float(sme["size_cr"].sum()) if len(sme) else 0.0,
        "avg": float(sub["size_cr"].mean()) if n else 0.0,
        "median": float(sub["size_cr"].median()) if n else 0.0,
        "largest": float(sub["size_cr"].max()) if n else 0.0,
    }


MONTHLY_SYSTEM = (
    "You are a senior capital-markets analyst at MS Kapital, an Indian corporate affairs and "
    "investment advisory. Write sharp, quantitative IPO market analysis. Format strictly in markdown: "
    "'## ' for section headings and '- ' for bullet points. All money values are Indian Rupees in Crore "
    "(Rs. Cr). Always compare with the previous month and cite the actual numbers."
)

YEARLY_SYSTEM = (
    "You are a senior capital-markets analyst at MS Kapital, an Indian corporate affairs and investment "
    "advisory. Write sharp, quantitative annual IPO market analysis. Format strictly in markdown: '## ' for "
    "section headings and '- ' for bullet points. All money values are Indian Rupees in Crore (Rs. Cr). "
    "Always discuss multi-year trends and previous years, citing the actual numbers."
)


# ==========================================================
# MONTHLY INSIGHTS (with previous-month + 6-month context)
# ==========================================================
def generate_monthly_insights(df, period, api_key=None):
    d = _clean_dataset(df)
    if d.empty: return None
    period = str(period)
    mdf = d[d["period"] == period]

    try: prev_period = (pd.Period(period, freq="M") - 1).strftime("%Y-%m")
    except Exception: prev_period = ""
    prev_df = d[d["period"] == prev_period] if prev_period else d.iloc[0:0]

    curr, prev = _period_stats(mdf), _period_stats(prev_df)

    ps = sorted(d["period"].unique())
    idx = ps.index(period) if period in ps else -1
    last6 = ps[max(0, idx - 5): idx + 1] if idx >= 0 else []
    trend = [f"  - {_month_label(p)}: Rs. {_cr(d[d['period']==p]['size_cr'].sum())} Cr across {len(d[d['period']==p])} IPOs"
             for p in last6]

    cs = mdf.groupby("sector")["size_cr"].sum() if not mdf.empty else pd.Series(dtype=float)
    pv = prev_df.groupby("sector")["size_cr"].sum() if not prev_df.empty else pd.Series(dtype=float)
    secs = [f"  - {s}: Rs. {_cr(float(cs.get(s,0)))} Cr now vs Rs. {_cr(float(pv.get(s,0)))} Cr previous month"
            for s in cs.sort_values(ascending=False).head(5).index]

    top_city = "N/A"
    if not mdf.empty:
        ct = mdf.groupby("city")["size_cr"].sum().sort_values(ascending=False)
        if len(ct): top_city = ct.index[0]

    prompt = f"""Analyze the Indian IPO market for {_month_label(period)} and compare it with the previous month ({_month_label(prev_period)}).

CURRENT MONTH ({_month_label(period)})
- Total IPOs: {curr['count']}
- Total mobilised: Rs. {_cr(curr['raised'])} Cr
- Main Board: {curr['mb_count']} issues, Rs. {_cr(curr['mb_raised'])} Cr
- SME: {curr['sme_count']} issues, Rs. {_cr(curr['sme_raised'])} Cr
- Average issue size: Rs. {_cr(curr['avg'])} Cr
- Median issue size: Rs. {_cr(curr['median'])} Cr
- Largest issue: Rs. {_cr(curr['largest'])} Cr
- Top city: {top_city}

PREVIOUS MONTH ({_month_label(prev_period)})
- Total IPOs: {prev['count']}
- Total mobilised: Rs. {_cr(prev['raised'])} Cr
- Main Board: {prev['mb_count']} issues, Rs. {_cr(prev['mb_raised'])} Cr
- SME: {prev['sme_count']} issues, Rs. {_cr(prev['sme_raised'])} Cr
- Average issue size: Rs. {_cr(prev['avg'])} Cr

MONTH-ON-MONTH CHANGE
- Mobilisation: {_growth(curr['raised'], prev['raised'])}
- Issue count: {_growth(curr['count'], prev['count'])}
- Average issue size: {_growth(curr['avg'], prev['avg'])}

LAST 6-MONTH TREND
{chr(10).join(trend) or '  - No trend data'}

TOP SECTORS (CURRENT vs PREVIOUS MONTH)
{chr(10).join(secs) or '  - No sector data'}

Write a detailed analyst report with these exact sections:
## Executive Summary
## Comparison with Previous Month
## Market Momentum
## Board Dynamics
## Sector & Geographic Trends
## Issue Size Analysis
## Risks & Watchpoints
## Outlook

Use '- ' bullets for key takeaways under each section. Be specific, quantitative and concise."""

    return call_groq(prompt, system_prompt=MONTHLY_SYSTEM, api_key=api_key)


# ==========================================================
# YEARLY INSIGHTS (with multi-year + YoY + board-mix context)
# ==========================================================
def generate_yearly_insights(df, year, api_key=None):
    d = _clean_dataset(df)
    if d.empty: return None
    year = str(year)
    try: prev_year = str(int(year) - 1)
    except Exception: prev_year = ""

    d2 = d.copy(); d2["year"] = d2["period"].astype(str).str[:4]

    def ys(y):
        sub = d2[d2["year"] == str(y)]
        s = _period_stats(sub)
        s["sectors"] = int(sub["sector"].replace("", np.nan).nunique())
        s["cities"] = int(sub["city"].replace("", np.nan).nunique())
        return s

    curr = ys(year)
    prev = ys(prev_year) if prev_year else {}

    last5 = sorted(d2["year"].unique())[-5:]
    multi = [f"  - {y}: Rs. {_cr(ys(y)['raised'])} Cr across {ys(y)['count']} IPOs (avg Rs. {_cr(ys(y)['avg'])} Cr)"
             for y in last5]

    cs = d2[d2["year"] == year].groupby("sector")["size_cr"].sum()
    pv = d2[d2["year"] == prev_year].groupby("sector")["size_cr"].sum() if prev_year else pd.Series(dtype=float)
    syoy = [f"  - {s}: Rs. {_cr(float(cs.get(s,0)))} Cr in {year} vs Rs. {_cr(float(pv.get(s,0)))} Cr in {prev_year} ({_growth(float(cs.get(s,0)), float(pv.get(s,0)))})"
            for s in cs.sort_values(ascending=False).head(6).index]

    bmix = []
    for y in last5:
        sub = d2[d2["year"] == y]
        mb = int((sub["board"] == "Main Board").sum()); sme = int((sub["board"] == "SME").sum()); t = len(sub)
        bmix.append(f"  - {y}: Main Board {mb}, SME {sme}" + (f" (SME share {sme/t*100:.0f}%)" if t else ""))

    top_city = "N/A"
    ydf = d2[d2["year"] == year]
    if not ydf.empty:
        ct = ydf.groupby("city")["size_cr"].sum().sort_values(ascending=False)
        if len(ct): top_city = ct.index[0]

    prompt = f"""Provide an in-depth analysis of the Indian IPO market for Calendar Year {year}, with strong emphasis on previous years' trends.

CURRENT YEAR ({year})
- Total IPOs: {curr['count']}
- Total mobilised: Rs. {_cr(curr['raised'])} Cr
- Main Board: {curr['mb_count']} issues, Rs. {_cr(curr['mb_raised'])} Cr
- SME: {curr['sme_count']} issues, Rs. {_cr(curr['sme_raised'])} Cr
- Average issue size: Rs. {_cr(curr['avg'])} Cr
- Median issue size: Rs. {_cr(curr['median'])} Cr
- Largest issue: Rs. {_cr(curr['largest'])} Cr
- Top city: {top_city}
- Active sectors: {curr['sectors']}, Active cities: {curr['cities']}

PREVIOUS YEAR ({prev_year})
- Total IPOs: {prev.get('count', 0)}
- Total mobilised: Rs. {_cr(prev.get('raised', 0.0))} Cr
- Average issue size: Rs. {_cr(prev.get('avg', 0.0))} Cr

YEAR-ON-YEAR CHANGE
- Mobilisation: {_growth(curr['raised'], prev.get('raised', 0.0))}
- Issue count: {_growth(curr['count'], prev.get('count', 0))}
- Average issue size: {_growth(curr['avg'], prev.get('avg', 0.0))}

MULTI-YEAR TREND (LAST 5 YEARS)
{chr(10).join(multi) or '  - No multi-year data'}

SECTOR MOVEMENT (YoY)
{chr(10).join(syoy) or '  - No sector data'}

BOARD MIX OVER YEARS
{chr(10).join(bmix) or '  - No board data'}

Write a detailed annual review with these exact sections:
## Annual Executive Summary
## Multi-Year Trend Review
## Year-on-Year Analysis
## Board Composition Trends
## Sectoral & Geographic Shifts
## Key Risks & Structural Watchpoints
## Forward Outlook

Use '- ' bullets for key takeaways under each section. Be specific, quantitative and concise."""

    return call_groq(prompt, system_prompt=YEARLY_SYSTEM, api_key=api_key)


# ==========================================================
# SELF-TEST
# ==========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GROQ API DIAGNOSTIC")
    print("=" * 60)
    key = get_groq_api_key()
    if key:
        print(f"[OK] Found GROQ_API_KEY: {key[:7]}...{key[-4:]}")
    else:
        print("[!] GROQ_API_KEY not found in environment.")
        key = input("Paste your Groq API key (starts with gsk_): ").strip()
        if not key:
            print("No key provided. Exiting.")
            raise SystemExit
        os.environ["GROQ_API_KEY"] = key

    print("\nTesting connection...\n")
    res = test_groq_connection(key)
    print(f"Key found      : {res['key_found']}")
    print(f"Key preview    : {res['key_preview']}\n")
    print("Models tested:")
    for m in res["models_tried"]:
        print(f"  - {m['model']:30s} : {m['status']}")

    if res["working_model"]:
        print(f"\n[SUCCESS] Working model: {res['working_model']}")
        print(f"Sample response: {res['sample_response']}")
        print("\nGroq is working - PDFs and dashboard will now include AI analysis.")
    else:
        print("\n[FAILED] No model worked. Errors:")
        for e in res["errors"]:
            print(f"  - {e}")
        print("\nCommon fixes:")
        print("  1. Key must start with 'gsk_' (from console.groq.com/keys)")
        print("  2. No extra spaces / quotes around the key")
        print("  3. Ensure internet / firewall allows api.groq.com")
        print("  4. Free tier has rate limits - wait a minute and retry")