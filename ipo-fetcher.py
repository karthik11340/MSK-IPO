import os
import re
import json
import html
from datetime import datetime, date

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


# ==========================================================
# CONFIG
# ==========================================================
BASE_URL = "https://www.ipoplatform.com/main-board/index"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest"
}

EXCEL_FILE = r"C:\Users\KARTHIKKPMSHoldings\OneDrive - MS Chartered Accountants\Desktop\PROJECTS\IPO MARKET\IPO_Database - Copy.xlsx"

MASTER_CSV = os.path.join(
    os.path.dirname(EXCEL_FILE),
    "IPO_Master.csv"
)

STATE_FILE = "tracker_state.json"
PAGE_SIZE = 100

# Leave empty to ask date while running.
# Examples:
# FROM_DATE = "2026-01-01"
# FROM_DATE = "ALL"
FROM_DATE = ""


MASTER_COLUMNS = [
    "id",
    "board",
    "company_name",
    "sector",
    "merchant_banker",
    "city",
    "size_cr",
    "open_date",
    "close_date",
    "listing_date",
    "period",
    "year",
    "month",
    "ipo_url"
]


# ==========================================================
# EXCEL TEXT SANITIZER (FIXED FOR PANDAS 2.0+ & XML)
# ==========================================================
# Matches any character not allowed in XML 1.0 (which openpyxl enforces)
ILLEGAL_XML_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1F\uD800-\uDFFF\uFFFE\uFFFF]')


def sanitize_text(value):
    """
    Removes characters that openpyxl/Excel rejects.
    Also converts HTML entities like &#039; into normal characters.
    """
    if value is None:
        return ""

    if not isinstance(value, str):
        return value

    value = html.unescape(value)
    value = ILLEGAL_XML_RE.sub("", value)

    return value.strip()


def sanitize_dataframe(df):
    """
    Cleans all text/string columns so Excel writing does not fail.
    Compatible with Pandas 2.0+ StringDtype.
    """
    df = df.copy()

    # Clean column names
    df.columns = [sanitize_text(str(c)) for c in df.columns]

    for col in df.columns:
        # Catch both old 'object' and new 'string' dtypes
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            def clean_val(v):
                if isinstance(v, str):
                    return sanitize_text(v)
                return v
            
            df[col] = df[col].apply(clean_val)

    return df


def normalize_id_series(series):
    """
    Normalizes IDs so 1, 1.0 and '1' are treated similarly.
    """
    if series is None:
        return pd.Series(dtype="string")

    s = series.astype(str).str.strip()

    numeric = pd.to_numeric(s, errors="coerce")

    if len(s) == 0:
        return s

    if numeric.notna().mean() >= 0.8:
        return numeric.astype("Int64").astype(str).str.strip()

    return s


# ==========================================================
# BASIC HELPERS
# ==========================================================
def clean_html(value):
    if value is None:
        return ""

    value = str(value)

    if not value:
        return ""

    soup = BeautifulSoup(value, "html.parser")
    text = soup.get_text(" ", strip=True)

    return html.unescape(text)


def extract_url(html_value):
    if html_value is None:
        return ""

    html_text = str(html_value)

    match = re.search(r'href=["\'](.*?)["\']', html_text, re.I)

    if match:
        return match.group(1)

    match = re.search(r'(https?://[^\s"\'<>]+)', html_text)

    if match:
        return match.group(1)

    return ""


def parse_size_to_cr(value):
    if value is None:
        return np.nan

    text = str(value).lower().strip()

    if not text:
        return np.nan

    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", text)

    if not numbers:
        return np.nan

    number = float(numbers[0].replace(",", ""))

    if "lakh" in text or "lac" in text:
        number = number / 100.0

    elif "billion" in text or re.search(r"\d\s*bn\b", text):
        number = number * 100.0

    return round(number, 2)


def parse_user_date(value):
    value = str(value).strip()

    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass

    raise ValueError(f"Could not parse date: '{value}'. Please use YYYY-MM-DD.")


# ==========================================================
# STATE
# ==========================================================
def default_state():
    return {
        "MainBoard": 0,
        "SME": 0,
        "last_run_date": ""
    }


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)


def load_state():
    if not os.path.exists(STATE_FILE):
        state = default_state()
        save_state(state)
        return state

    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
    except Exception:
        state = default_state()

    base = default_state()

    for key in ["MainBoard", "SME"]:
        value = state.get(key, 0)

        if isinstance(value, dict):
            value = value.get("last_id", 0)

        try:
            base[key] = int(value)
        except Exception:
            base[key] = 0

    base["last_run_date"] = str(state.get("last_run_date", "") or "")

    return base


def get_start_date(state):
    fixed = str(FROM_DATE or "").strip()

    if fixed.lower() == "all":
        return None

    if fixed:
        return parse_user_date(fixed)

    default_date = state.get("last_run_date", "")

    while True:
        prompt = "Enter start date (YYYY-MM-DD) "

        if default_date:
            prompt += f"[Enter = last run date: {default_date}, ALL = all records]: "
        else:
            prompt += "[Enter/ALL = all records]: "

        try:
            raw = input(prompt).strip()
        except EOFError:
            raw = default_date or "ALL"

        if raw == "" and default_date:
            raw = default_date

        if raw == "" or raw.lower() == "all":
            return None

        try:
            return parse_user_date(raw)
        except ValueError as err:
            print(err)


# ==========================================================
# DOWNLOAD DATA
# ==========================================================
def fetch_all(ipo_type):
    all_rows = []
    start = 0
    draw = 1

    while True:
        params = {
            "draw": draw,
            "start": start,
            "length": PAGE_SIZE,
            "search[value]": "",
            "search[regex]": "false",
            "ipo_type": ipo_type,
            "selected_year": "all"
        }

        print(f"Fetching {ipo_type}: start={start}, page_size={PAGE_SIZE}")

        response = requests.get(
            BASE_URL,
            params=params,
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

        try:
            js = response.json()
        except ValueError:
            print(f"{ipo_type}: API did not return JSON.")
            break

        rows = js.get("data", [])

        if not isinstance(rows, list):
            print(f"{ipo_type}: Unexpected API response format.")
            break

        if len(rows) == 0:
            break

        all_rows.extend(rows)

        try:
            total = int(js.get("recordsTotal", 0))
        except Exception:
            total = len(all_rows)

        print(f"{ipo_type}: Downloaded {len(all_rows)} of {total} rows")

        start += PAGE_SIZE
        draw += 1

        if total and start >= total:
            break

        if len(rows) < PAGE_SIZE:
            break

    return all_rows


# ==========================================================
# CLEAN RAW DATAFRAME (FIXED FOR PANDAS 2.0+)
# ==========================================================
def prepare_dataframe(rows):
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    if "company_link" in df.columns:
        df["ipo_url"] = df["company_link"].apply(extract_url)

    html_columns = [
        "company_link",
        "sector",
        "merchant_banker",
        "city",
        "city_sme",
        "listing_year",
        "ipo_year_mb",
        "ipo_year_sme"
    ]

    for col in html_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: clean_html(x) if isinstance(x, str) else x)

    # Clean any other string column containing HTML tags
    for col in df.columns:
        if col == "ipo_url":
            continue
            
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            try:
                if df[col].astype(str).str.contains(r"<[^>]+>", regex=True, na=False).any():
                    df[col] = df[col].apply(lambda x: clean_html(x) if isinstance(x, str) else x)
            except Exception:
                pass

    return df


# ==========================================================
# DATE FILTER
# ==========================================================
def apply_date_filter(df, start_date):
    if df.empty or start_date is None:
        return df

    date_col = None

    for col in df.columns:
        if "date" in str(col).lower():
            date_col = col
            break

    if date_col:
        parsed = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        filtered = df[parsed >= pd.Timestamp(start_date)].copy()

        print(
            f"Date filter applied on column '{date_col}': "
            f"{len(filtered)} rows kept out of {len(df)}"
        )

        return filtered

    year_col = None

    for col in df.columns:
        if "year" in str(col).lower():
            year_col = col
            break

    if year_col:
        years = (
            df[year_col]
            .astype(str)
            .str.extract(r"(\d{4})", expand=False)
        )

        years = pd.to_numeric(years, errors="coerce")

        filtered = df[years >= start_date.year].copy()

        print(
            f"Year filter applied on column '{year_col}': "
            f"{len(filtered)} rows kept out of {len(df)}"
        )

        return filtered

    print("No date/year column found. Using all fetched rows.")
    return df


# ==========================================================
# EXCEL HELPERS
# ==========================================================
def read_existing_sheet(sheet_name):
    if not os.path.exists(EXCEL_FILE):
        return pd.DataFrame()

    try:
        return pd.read_excel(
            EXCEL_FILE,
            sheet_name=sheet_name
        )
    except Exception as e:
        print(f"Could not read existing sheet '{sheet_name}': {e}")
        return pd.DataFrame()


def write_sheet(sheet_name, df):
    folder = os.path.dirname(EXCEL_FILE)

    if folder:
        os.makedirs(folder, exist_ok=True)

    # Remove illegal Excel characters before writing
    df = sanitize_dataframe(df)

    try:
        if os.path.exists(EXCEL_FILE):
            with pd.ExcelWriter(
                EXCEL_FILE,
                engine="openpyxl",
                mode="a",
                if_sheet_exists="replace"
            ) as writer:
                df.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False
                )
        else:
            with pd.ExcelWriter(
                EXCEL_FILE,
                engine="openpyxl"
            ) as writer:
                df.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False
                )

    except PermissionError:
        print(f"Cannot write to {EXCEL_FILE}. Please close the Excel file and rerun.")
        raise


# ==========================================================
# UPDATE SHEET
# ==========================================================
def update_sheet(sheet_name, ipo_type, state, start_date):
    print(f"\nUpdating {sheet_name}...")

    rows = fetch_all(ipo_type)
    df = prepare_dataframe(rows)

    if df.empty:
        print("No data fetched.")
        return

    df = apply_date_filter(df, start_date)

    if df.empty:
        print("No rows satisfy the date filter.")
        return

    existing = read_existing_sheet(sheet_name)

    if "id" in df.columns and not existing.empty and "id" in existing.columns:
        existing_ids = set(normalize_id_series(existing["id"]))
        new_ids = normalize_id_series(df["id"])
        new_df = df[~new_ids.isin(existing_ids)].copy()
    else:
        new_df = df.copy()

    if "id" in new_df.columns:
        new_df["_dedupe"] = normalize_id_series(new_df["id"])
        new_df.drop_duplicates(subset=["_dedupe"], keep="last", inplace=True)
        new_df.drop(columns=["_dedupe"], inplace=True)
    else:
        new_df.drop_duplicates(inplace=True)

    print(f"New IPOs to add: {len(new_df)}")

    if new_df.empty:
        print("No new records to write. Duplicates avoided.")
        return

    combined = pd.concat(
        [existing, new_df],
        ignore_index=True
    )

    if "id" in combined.columns:
        combined["_dedupe"] = normalize_id_series(combined["id"])
        combined.drop_duplicates(subset=["_dedupe"], keep="last", inplace=True)
        combined.drop(columns=["_dedupe"], inplace=True)
    else:
        combined.drop_duplicates(inplace=True)

    if "id" in combined.columns:
        numeric_ids = pd.to_numeric(combined["id"], errors="coerce")

        if len(combined) and numeric_ids.notna().mean() > 0.8:
            combined["_sort"] = numeric_ids
            combined.sort_values("_sort", ascending=False, inplace=True)
            combined.drop(columns=["_sort"], inplace=True)
        else:
            combined["_sort"] = normalize_id_series(combined["id"])
            combined.sort_values("_sort", ascending=False, inplace=True)
            combined.drop(columns=["_sort"], inplace=True)

    write_sheet(sheet_name, combined)

    if "id" in combined.columns:
        numeric_ids = pd.to_numeric(combined["id"], errors="coerce")

        if numeric_ids.notna().any():
            state[ipo_type] = int(numeric_ids.max())

    print(f"Total records in {sheet_name}: {len(combined)}")


# ==========================================================
# STRUCTURE DATA FOR DASHBOARD / PDF
# ==========================================================
def normalize_dataframe(df, board):
    if df.empty:
        return pd.DataFrame(columns=MASTER_COLUMNS)

    out = df.copy()

    def col(name, default=""):
        if name in out.columns:
            return out[name]
        return pd.Series([default] * len(out), index=out.index)

    out["board"] = board

    out["company_name"] = (
        col("company_link")
        .astype(str)
        .str.strip()
        .replace("nan", "")
    )

    out["sector"] = (
        col("sector")
        .astype(str)
        .str.strip()
        .replace("nan", "")
        .replace("", "Unknown")
    )

    out["merchant_banker"] = (
        col("merchant_banker")
        .astype(str)
        .str.strip()
        .replace("nan", "")
    )

    city = (
        col("city")
        .astype(str)
        .str.strip()
        .replace("nan", "")
    )

    city_sme = (
        col("city_sme")
        .astype(str)
        .str.strip()
        .replace("nan", "")
    )

    out["city"] = city.where(city != "", city_sme)

    size_col = None

    for c in out.columns:
        if "size" in str(c).lower():
            size_col = c
            break

    if size_col:
        out["size_cr"] = col(size_col).apply(parse_size_to_cr)
    else:
        out["size_cr"] = np.nan

    def dcol(*keywords):
        for c in out.columns:
            col_name = str(c).lower()

            if all(keyword in col_name for keyword in keywords):
                return c

        return None

    open_col = dcol("open")
    close_col = dcol("close")
    listing_col = dcol("listing", "date")

    if open_col:
        open_dt = pd.to_datetime(col(open_col), errors="coerce", dayfirst=True)
    else:
        open_dt = pd.Series(pd.NaT, index=out.index)

    if close_col:
        close_dt = pd.to_datetime(col(close_col), errors="coerce", dayfirst=True)
    else:
        close_dt = pd.Series(pd.NaT, index=out.index)

    if listing_col:
        listing_dt = pd.to_datetime(col(listing_col), errors="coerce", dayfirst=True)
    else:
        listing_dt = pd.Series(pd.NaT, index=out.index)

    period_dt = open_dt.combine_first(listing_dt).combine_first(close_dt)

    if period_dt.notna().mean() < 0.5:
        year_text = (
            col("listing_year")
            .astype(str)
            .str.extract(r"(\d{4})", expand=False)
        )

        year_dt = pd.to_datetime(year_text + "-01-01", errors="coerce")
        period_dt = period_dt.combine_first(year_dt)

    out["open_date"] = open_dt.dt.strftime("%Y-%m-%d").fillna("")
    out["close_date"] = close_dt.dt.strftime("%Y-%m-%d").fillna("")
    out["listing_date"] = listing_dt.dt.strftime("%Y-%m-%d").fillna("")

    out["period"] = period_dt.dt.strftime("%Y-%m").fillna("")
    out["year"] = period_dt.dt.year.fillna(0).astype(int)
    out["month"] = period_dt.dt.month.fillna(0).astype(int)

    out["ipo_url"] = (
        col("ipo_url")
        .astype(str)
        .str.strip()
        .replace("nan", "")
    )

    for c in MASTER_COLUMNS:
        if c not in out.columns:
            out[c] = ""

    return out[MASTER_COLUMNS]


def build_master():
    frames = []

    for sheet_name, board in [("MainBoard", "Main Board"), ("SME", "SME")]:
        try:
            raw = pd.read_excel(
                EXCEL_FILE,
                sheet_name=sheet_name
            )
        except Exception as e:
            print(f"Could not read sheet '{sheet_name}' while building master: {e}")
            continue

        if raw.empty:
            continue

        frames.append(normalize_dataframe(raw, board))

    if not frames:
        master = pd.DataFrame(columns=MASTER_COLUMNS)
        master = sanitize_dataframe(master)
        master.to_csv(MASTER_CSV, index=False, encoding="utf-8-sig")
        print(f"Empty master dataset written: {MASTER_CSV}")
        return master

    master = pd.concat(frames, ignore_index=True)

    if "id" in master.columns and "board" in master.columns:
        id_present = (
            master["id"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": "", "none": ""})
            .ne("")
            .mean() > 0.5
        )

        if id_present:
            master["_dedupe"] = (
                master["board"].astype(str).str.strip()
                + "|"
                + normalize_id_series(master["id"])
            )
            master.drop_duplicates(subset=["_dedupe"], keep="last", inplace=True)
            master.drop(columns=["_dedupe"], inplace=True)
        else:
            master.drop_duplicates(inplace=True)

    master.sort_values(
        "period",
        ascending=False,
        inplace=True,
        na_position="last"
    )

    master = sanitize_dataframe(master)

    master.to_csv(
        MASTER_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nMaster dataset written: {MASTER_CSV}")
    print(f"Total structured rows: {len(master)}")

    return master


# ==========================================================
# MAIN
# ==========================================================
def main():
    state = load_state()

    start_date = get_start_date(state)

    print("\nStart date selected:", start_date if start_date else "ALL")

    try:
        update_sheet(
            sheet_name="MainBoard",
            ipo_type="MainBoard",
            state=state,
            start_date=start_date
        )

        update_sheet(
            sheet_name="SME",
            ipo_type="SME",
            state=state,
            start_date=start_date
        )

    except requests.exceptions.RequestException as e:
        print(f"Network/API error: {e}")
        return

    state["last_run_date"] = date.today().isoformat()
    save_state(state)

    build_master()

    print("\nDone!")
    print("Excel File :", EXCEL_FILE)
    print("Master CSV :", MASTER_CSV)


if __name__ == "__main__":
    main()