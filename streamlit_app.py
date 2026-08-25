import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import groq_analyzer
except Exception:
    groq_analyzer = None

try:
    import report_builder as report_builder_mod
    build_monthly_pdf = report_builder_mod.build_monthly_pdf
except Exception:
    report_builder_mod = None
    build_monthly_pdf = None

try:
    import yearly_report_builder as yearly_report_builder_mod
    build_yearly_pdf = yearly_report_builder_mod.build_yearly_pdf
except Exception:
    yearly_report_builder_mod = None
    build_yearly_pdf = None


# ==========================================================
# CONFIG
# ==========================================================
MASTER_CSV = r"IPO_Master.csv"
if not os.path.exists(MASTER_CSV):
    _local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IPO_Master.csv")
    if os.path.exists(_local):
        MASTER_CSV = _local

INK, ACCENT = "#12263F", "#2563EB"

BUCKETS = ["Mega  (Rs. 1,000+ Cr)", "Large  (Rs. 500-1,000 Cr)", "Medium  (Rs. 100-500 Cr)",
           "Small  (Rs. 10-100 Cr)", "Micro  (Below Rs. 10 Cr)"]

st.set_page_config(page_title="IPO Market Intelligence | MS Kapital", page_icon="📊", layout="wide")


# ==========================================================
# HELPERS
# ==========================================================
def fmt_month(p):
    try: return pd.Period(p, freq="M").strftime("%B %Y")
    except Exception: return str(p)


def _bucket(x):
    try: x = float(x)
    except Exception: return "Micro  (Below Rs. 10 Cr)"
    if x >= 1000: return BUCKETS[0]
    if x >= 500: return BUCKETS[1]
    if x >= 100: return BUCKETS[2]
    if x >= 10: return BUCKETS[3]
    return BUCKETS[4]


def read_csv_safe(path):
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin-1", "utf-16"]:
        try:
            return pd.read_csv(path, dtype={"id": str}, encoding=enc)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise ValueError("IPO_Master.csv is empty.")
    raise ValueError("Could not read IPO_Master.csv.")


@st.cache_data(ttl=600)
def load_master():
    df = read_csv_safe(MASTER_CSV)
    df["size_cr"] = pd.to_numeric(df.get("size_cr"), errors="coerce")
    df["period"] = df["period"].astype(str).str.strip()
    df = df[~df["period"].str.lower().isin(["", "nan", "nat", "none", "null"])]
    for col in ["board", "sector", "merchant_banker", "company_name", "city", "ipo_url", "listing_date"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).str.strip().replace("nan", "")
    df["sector"] = df["sector"].replace("", "Unknown")
    df["board"] = df["board"].replace("", "Unknown")
    df["city"] = df["city"].replace("", "Unknown")
    df["size_band"] = df["size_cr"].apply(_bucket)
    return df


# ==========================================================
# INITIAL LOAD
# ==========================================================
if not os.path.exists(MASTER_CSV):
    st.error("IPO_Master.csv not found. Run `python ipo-fetcher.py` first.")
    st.stop()

df = load_master()
periods = sorted(df["period"].unique(), reverse=True)
all_years = sorted({p[:4] for p in periods}, reverse=True)
if not periods:
    st.error("No valid periods in IPO_Master.csv.")
    st.stop()


# ==========================================================
# HEADER
# ==========================================================
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:flex-end;border-bottom:3px solid {INK};
padding-bottom:8px;margin-bottom:16px">
<div><span style="background:{ACCENT};color:#fff;padding:3px 10px;font-weight:700;letter-spacing:1px">MS KAPITAL</span>
<h1 style="margin:8px 0 0">IPO Market Intelligence</h1></div>
<div style="color:#888">Monthly & Annual Capital Markets Brief • AI-Enhanced</div></div>""",
            unsafe_allow_html=True)


# ==========================================================
# SIDEBAR - GLOBAL FILTERS
# ==========================================================
def _banker_options():
    b = (df.assign(bk=df["merchant_banker"].str.split(r",|;|&")).explode("bk"))
    b["bk"] = b["bk"].str.strip()
    b = b[~b["bk"].str.lower().isin(["", "nan", "none", "null"])]
    return sorted(b.groupby("bk")["size_cr"].sum().sort_values(ascending=False).head(30).index.tolist())


with st.sidebar:
    st.header("Filters")
    period = st.selectbox("Report month", periods, format_func=fmt_month, key="f_period")
    board_options = sorted(df["board"].unique())
    boards = st.multiselect("Board", board_options,
                            default=[b for b in ["Main Board", "SME"] if b in board_options] or board_options,
                            key="f_boards")
    sectors = st.multiselect("Sector", sorted(df["sector"].unique()), key="f_sectors")
    city_options = df["city"].value_counts().head(30).index.tolist()
    cities = st.multiselect("City (top 30)", city_options, key="f_cities")
    bands = st.multiselect("Issue size band", BUCKETS, key="f_bands")
    bankers = st.multiselect("Merchant banker (top 30)", _banker_options(), key="f_bankers")
    search = st.text_input("Search company", "", key="f_search")
    st.divider()
    if st.button("Reset filters"):
        for k in ["f_boards", "f_sectors", "f_cities", "f_bands", "f_bankers", "f_search"]:
            st.session_state.pop(k, None)
        st.rerun()
    st.divider()
    st.subheader("AI Settings")
    groq_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""),
                             help="Free key at console.groq.com")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    st.caption(f"Data rows: {len(df):,}  •  Latest: {fmt_month(periods[0])}")


# ==========================================================
# APPLY GLOBAL FILTERS
# ==========================================================
if not boards:
    boards = board_options
fdf = df[df["board"].isin(boards)].copy()
if sectors: fdf = fdf[fdf["sector"].isin(sectors)]
if cities: fdf = fdf[fdf["city"].isin(cities)]
if bands: fdf = fdf[fdf["size_band"].isin(bands)]
if bankers:
    fdf = fdf[fdf["merchant_banker"].apply(
        lambda x: any(bk in [s.strip() for s in str(x).split(",")] + [s.strip() for s in str(x).split(";")] + [s.strip() for s in str(x).split("&")] for bk in bankers))]
if search:
    fdf = fdf[fdf["company_name"].str.contains(search, case=False, na=False)]

m_df = fdf[fdf["period"] == period]
prev = (pd.Period(period, freq="M") - 1).strftime("%Y-%m")
p_df = fdf[fdf["period"] == prev]

st.caption(f"Filters active: {len(fdf):,} rows match  •  {len(m_df)} IPOs in {fmt_month(period)}")


# ==========================================================
# KPI METRICS (MONTH)
# ==========================================================
raised, count = m_df["size_cr"].sum(), len(m_df)
p_raised, p_count = p_df["size_cr"].sum(), len(p_df)
c = st.columns(6)
c[0].metric("Mobilised (Rs. Cr)", f"{raised:,.0f}",
            f"{((raised - p_raised) / p_raised) * 100:.1f}% MoM" if p_raised else None)
c[1].metric("Total IPOs", count, f"{((count - p_count) / p_count) * 100:.1f}% MoM" if p_count else None)
c[2].metric("Main Board", int((m_df["board"] == "Main Board").sum()))
c[3].metric("SME", int((m_df["board"] == "SME").sum()))
c[4].metric("Avg Size (Rs. Cr)", f"{m_df['size_cr'].mean():,.0f}" if count else "0")
c[5].metric("Median Size (Rs. Cr)", f"{m_df['size_cr'].median():,.0f}" if count else "0")


# ==========================================================
# MOMENTUM (12 MONTHS)
# ==========================================================
st.subheader("Listing Momentum — Trailing 12 Months")
trend = (fdf[fdf["period"] <= period].groupby("period")
         .agg(raised=("size_cr", "sum"), n=("period", "size"))
         .reset_index().sort_values("period").tail(12))
if not trend.empty:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=trend["period"], y=trend["raised"], name="Rs. Cr", marker_color=INK), secondary_y=False)
    fig.add_trace(go.Scatter(x=trend["period"], y=trend["n"], name="IPOs", mode="lines+markers",
                             line=dict(color=ACCENT, width=2)), secondary_y=True)
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=40), legend=dict(orientation="h", y=1.08))
    fig.update_yaxes(title_text="Rs. Cr", secondary_y=False)
    fig.update_yaxes(title_text="IPO Count", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No trend data for the selected filters.")


# ==========================================================
# YEARLY ANALYSIS (multi-year overview)
# ==========================================================
st.subheader("Yearly Analysis")
yb = fdf.assign(year=fdf["period"].astype(str).str[:4])
ya = (yb.groupby("year").agg(raised=("size_cr", "sum"), n=("period", "size"))
      .reset_index().sort_values("year"))
mb_y = (yb[yb["board"] == "Main Board"].groupby("year")
        .agg(mb_n=("size_cr", "size"), mb_r=("size_cr", "sum")).reset_index())
sme_y = (yb[yb["board"] == "SME"].groupby("year")
         .agg(sme_n=("size_cr", "size"), sme_r=("size_cr", "sum")).reset_index())
yt = (ya.merge(mb_y, on="year", how="left").merge(sme_y, on="year", how="left")
      .fillna(0).sort_values("year"))
yt["avg"] = yt.apply(lambda r: r["raised"] / r["n"] if r["n"] else 0, axis=1)
if not yt.empty:
    yf = make_subplots(specs=[[{"secondary_y": True}]])
    yf.add_trace(go.Bar(x=yt["year"], y=yt["raised"], name="Rs. Cr", marker_color=INK), secondary_y=False)
    yf.add_trace(go.Scatter(x=yt["year"], y=yt["n"], name="IPOs", mode="lines+markers",
                            line=dict(color=ACCENT, width=2)), secondary_y=True)
    yf.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=40), legend=dict(orientation="h", y=1.08))
    st.plotly_chart(yf, use_container_width=True)
    st.dataframe(yt.rename(columns={"year": "Year", "n": "Total IPOs", "raised": "Mobilised (Rs. Cr)",
                                    "mb_n": "MB IPOs", "mb_r": "MB (Rs. Cr)", "sme_n": "SME IPOs",
                                    "sme_r": "SME (Rs. Cr)", "avg": "Avg Size"}),
                 use_container_width=True, hide_index=True)
    with st.expander("Quarterly breakdown (all years)"):
        qq = pd.to_datetime(yb["period"] + "-01", errors="coerce").dt.quarter
        qt = (yb.assign(q=qq).groupby(["year", "q"])
              .agg(n=("size_cr", "size"), r=("size_cr", "sum")).reset_index())
        st.dataframe(qt.rename(columns={"year": "Year", "q": "Quarter", "n": "IPOs", "r": "Rs. Cr"}),
                     use_container_width=True, hide_index=True)
else:
    st.info("No yearly data for the selected filters.")


# ==========================================================
# YEAR EXPLORER  (OWN filters + detailed year-wise statistics)
# ==========================================================
st.divider()
st.subheader("📅 Year Explorer — Year-wise Statistics")
st.caption("This section has its OWN filters, independent of the sidebar. Pick a year, then refine below.")

sel_year = st.selectbox("Select year", all_years, key="ye_year")

ye_board_opts = sorted(df["board"].unique())
ye_sector_opts = sorted(df["sector"].unique())
ye_city_opts = df["city"].value_counts().head(30).index.tolist()
ye_banker_opts = _banker_options()

with st.expander("🎛️ Year Explorer filters", expanded=True):
    r1a, r1b, r1c = st.columns(3)
    ye_boards = r1a.multiselect("Board", ye_board_opts, default=ye_board_opts, key="ye_boards")
    ye_sectors = r1b.multiselect("Sector", ye_sector_opts, key="ye_sectors")
    ye_cities = r1c.multiselect("City (top 30)", ye_city_opts, key="ye_cities")
    r2a, r2b, r2c = st.columns(3)
    ye_bands = r2a.multiselect("Issue size band", BUCKETS, key="ye_bands")
    ye_bankers = r2b.multiselect("Merchant banker (top 30)", ye_banker_opts, key="ye_bankers")
    ye_search = r2c.text_input("Search company", "", key="ye_search")
    if st.button("Reset Year Explorer filters", key="ye_reset"):
        for k in ["ye_boards", "ye_sectors", "ye_cities", "ye_bands", "ye_bankers", "ye_search"]:
            st.session_state.pop(k, None)
        st.rerun()


def _ye_has_banker(row_val, chosen):
    parts = ([s.strip() for s in str(row_val).split(",")] +
             [s.strip() for s in str(row_val).split(";")] +
             [s.strip() for s in str(row_val).split("&")])
    return any(b in parts for b in chosen)


def _ye_apply(frame, yr):
    out = frame[frame["period"].astype(str).str[:4] == yr]
    if ye_boards: out = out[out["board"].isin(ye_boards)]
    if ye_sectors: out = out[out["sector"].isin(ye_sectors)]
    if ye_cities: out = out[out["city"].isin(ye_cities)]
    if ye_bands: out = out[out["size_band"].isin(ye_bands)]
    if ye_bankers:
        out = out[out.apply(lambda r: _ye_has_banker(r["merchant_banker"], ye_bankers), axis=1)]
    if ye_search:
        out = out[out["company_name"].str.contains(ye_search, case=False, na=False)]
    return out


yex = _ye_apply(df, sel_year)
try:
    prev_sel = str(int(sel_year) - 1)
except Exception:
    prev_sel = ""
yex_prev = _ye_apply(df, prev_sel) if prev_sel else pd.DataFrame()

yr_raised, yr_count = yex["size_cr"].sum(), len(yex)
pr_raised = yex_prev["size_cr"].sum() if not yex_prev.empty else 0
pr_count = len(yex_prev)

st.caption(f"{yr_count} IPOs match in {sel_year}  •  Rs. {yr_raised:,.0f} Cr mobilised")

if yr_count:
    k = st.columns(7)
    k[0].metric("Mobilised (Rs. Cr)", f"{yr_raised:,.0f}",
                f"{((yr_raised - pr_raised) / pr_raised) * 100:.1f}% YoY" if pr_raised else None)
    k[1].metric("Total IPOs", yr_count,
                f"{((yr_count - pr_count) / pr_count) * 100:.1f}% YoY" if pr_count else None)
    k[2].metric("Main Board", int((yex["board"] == "Main Board").sum()))
    k[3].metric("SME", int((yex["board"] == "SME").sum()))
    k[4].metric("Avg Size", f"{yex['size_cr'].mean():,.0f}")
    k[5].metric("Median Size", f"{yex['size_cr'].median():,.0f}")
    k[6].metric("Largest", f"{yex['size_cr'].max():,.0f}")

    ym = (yex.groupby("period").agg(raised=("size_cr", "sum"), n=("size_cr", "size"))
          .reset_index().sort_values("period"))
    best = ym.loc[ym["raised"].idxmax()]
    lgst = yex.loc[yex["size_cr"].idxmax()]
    ysec = yex.groupby("sector")["size_cr"].sum().sort_values(ascending=False)
    ycty = yex.groupby("city")["size_cr"].sum().sort_values(ascending=False)
    ybank = (yex.assign(banker=yex["merchant_banker"].str.split(r",|;|&")).explode("banker")
             .assign(banker=lambda x: x["banker"].str.strip()))
    ybank = ybank[~ybank["banker"].str.lower().isin(["", "nan", "none", "null"])]
    ybank = ybank.groupby("banker")["size_cr"].sum().sort_values(ascending=False)
    sme_share = (yex["board"] == "SME").sum() / yr_count * 100

    st.markdown("**Key takeaways**")
    for b in [
        f"**{sel_year}** mobilised **Rs. {yr_raised:,.0f} Cr** across **{yr_count}** IPOs" +
        (f" ({((yr_raised - pr_raised) / pr_raised) * 100:+.1f}% YoY)." if pr_raised else "."),
        f"Best month: **{fmt_month(best['period'])}** (Rs. {best['raised']:,.0f} Cr, {int(best['n'])} IPOs).",
        f"Largest issue: **{lgst['company_name']}** (Rs. {lgst['size_cr']:,.0f} Cr, {lgst['board']}).",
        f"Top sector: **{ysec.index[0]}** ({(ysec.iloc[0] / yr_raised * 100):.1f}% share).",
        f"Top city: **{ycty.index[0]}** (Rs. {ycty.iloc[0]:,.0f} Cr).",
        f"Top merchant banker: **{ybank.index[0]}** (Rs. {ybank.iloc[0]:,.0f} Cr).",
        f"SME accounted for **{sme_share:.0f}%** of issue count.",
    ]:
        st.markdown(f"- {b}")

    fig_ym = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ym.add_trace(go.Bar(x=ym["period"].str[5:], y=ym["raised"], name="Rs. Cr", marker_color=INK), secondary_y=False)
    fig_ym.add_trace(go.Scatter(x=ym["period"].str[5:], y=ym["n"], name="IPOs", mode="lines+markers",
                                line=dict(color=ACCENT, width=2)), secondary_y=True)
    fig_ym.update_layout(template="plotly_white", title=f"Monthly progression - {sel_year}",
                         margin=dict(l=10, r=10, t=40), legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig_ym, use_container_width=True)

    e1, e2, e3 = st.columns(3)
    with e1:
        yq = pd.to_datetime(yex["period"] + "-01", errors="coerce").dt.quarter
        qt_y = (yex.assign(q=yq).groupby("q").agg(n=("size_cr", "size"), r=("size_cr", "sum"))
                .reindex([1, 2, 3, 4]).fillna(0).reset_index())
        st.markdown("**Quarterly**")
        st.dataframe(qt_y.rename(columns={"q": "Quarter", "n": "IPOs", "r": "Rs. Cr"}),
                     use_container_width=True, hide_index=True)
    with e2:
        yb2 = yex.groupby("board").agg(n=("size_cr", "size"), r=("size_cr", "sum")).reset_index()
        st.markdown("**Board split**")
        st.dataframe(yb2.rename(columns={"board": "Board", "n": "IPOs", "r": "Rs. Cr"}),
                     use_container_width=True, hide_index=True)
    with e3:
        ysz = (yex.groupby("size_band").agg(n=("size_cr", "size"), r=("size_cr", "sum"))
               .reindex(BUCKETS).fillna(0).reset_index())
        st.markdown("**Size bands**")
        st.dataframe(ysz.rename(columns={"size_band": "Band", "n": "IPOs", "r": "Rs. Cr"}),
                     use_container_width=True, hide_index=True)

    g1, g2 = st.columns(2)
    with g1:
        ys2 = (yex.groupby("sector").agg(n=("size_cr", "size"), r=("size_cr", "sum"))
               .reset_index().sort_values("r", ascending=False).head(10))
        st.plotly_chart(go.Figure(
            go.Bar(y=ys2["sector"][::-1], x=ys2["r"][::-1], orientation="h", marker_color=INK,
                   text=ys2["r"][::-1].map(lambda v: f"{v:,.0f}"), textposition="outside"),
            layout=dict(template="plotly_white", title=f"Top sectors - {sel_year} (Rs. Cr)",
                        margin=dict(l=10, r=10, t=40))), use_container_width=True)
    with g2:
        yct = (yex.groupby("city").agg(n=("size_cr", "size"), r=("size_cr", "sum"))
               .reset_index().sort_values("r", ascending=False).head(8))
        st.plotly_chart(go.Figure(
            go.Bar(y=yct["city"][::-1], x=yct["r"][::-1], orientation="h", marker_color=ACCENT,
                   text=yct["r"][::-1].map(lambda v: f"{v:,.0f}"), textposition="outside"),
            layout=dict(template="plotly_white", title=f"Top cities - {sel_year} (Rs. Cr)",
                        margin=dict(l=10, r=10, t=40))), use_container_width=True)

    with st.expander("Sector detail table"):
        ysd = (yex.groupby("sector").agg(n=("size_cr", "size"), r=("size_cr", "sum"))
               .reset_index().sort_values("r", ascending=False))
        ysd["share"] = (ysd["r"] / yr_raised * 100)
        ysd["avg"] = ysd["r"] / ysd["n"]
        st.dataframe(ysd.rename(columns={"sector": "Sector", "n": "IPOs", "r": "Rs. Cr",
                                         "share": "Share %", "avg": "Avg Size"}).round(1),
                     use_container_width=True, hide_index=True)

    h1, h2 = st.columns(2)
    with h1:
        st.markdown(f"**Top issues of {sel_year}**")
        st.dataframe(yex.sort_values("size_cr", ascending=False)
                     [["company_name", "board", "sector", "city", "size_cr", "merchant_banker"]].head(15),
                     use_container_width=True, hide_index=True)
    with h2:
        st.markdown(f"**Top merchant bankers - {sel_year}**")
        st.dataframe(ybank.reset_index().rename(columns={"banker": "Banker", "size_cr": "Rs. Cr"}).head(15),
                     use_container_width=True, hide_index=True)

    st.download_button(f"⬇️ Export {sel_year} filtered data (CSV)", yex.to_csv(index=False),
                       file_name=f"MS_Kapital_IPO_{sel_year}.csv", mime="text/csv", key=f"ye_csv_{sel_year}")
else:
    st.info(f"No data for {sel_year} with the selected Year Explorer filters.")


# ==========================================================
# BOARD & SECTOR (month)
# ==========================================================
l, r = st.columns(2)
bd = m_df.groupby("board").agg(n=("size_cr", "size"), r=("size_cr", "sum")).reset_index()
if not bd.empty:
    l.plotly_chart(go.Figure(
        go.Bar(x=bd["board"], y=bd["r"], marker_color=[INK, ACCENT][:len(bd)],
               text=bd["r"].map(lambda v: f"{v:,.0f}"), textposition="outside"),
        layout=dict(template="plotly_white", title=f"Board split - {fmt_month(period)} (Rs. Cr)")),
        use_container_width=True)
else:
    l.info("No board data this month.")
sd = (m_df.groupby("sector").agg(n=("size_cr", "size"), r=("size_cr", "sum"))
      .reset_index().sort_values("r", ascending=False))
if not sd.empty:
    top = sd.head(8)[::-1]
    r.plotly_chart(go.Figure(
        go.Bar(y=top["sector"], x=top["r"], orientation="h", marker_color=INK,
               text=top["r"].map(lambda v: f"{v:,.0f}"), textposition="outside"),
        layout=dict(template="plotly_white", title="Top sectors (Rs. Cr)")),
        use_container_width=True)
else:
    r.info("No sector data this month.")
with st.expander("Sector movement vs previous month"):
    cs = m_df.groupby("sector")["size_cr"].sum()
    pv = p_df.groupby("sector")["size_cr"].sum()
    mover = pd.DataFrame({fmt_month(prev): pv, fmt_month(period): cs}).fillna(0)
    mover["MoM %"] = mover.apply(lambda x: ((x.iloc[1] - x.iloc[0]) / x.iloc[0] * 100) if x.iloc[0] else None, axis=1)
    mover = mover.sort_values(mover.columns[1], ascending=False)
    st.dataframe(mover.round(1), use_container_width=True)


# ==========================================================
# CITY & SIZE BAND (month)
# ==========================================================
l2, r2 = st.columns(2)
cd = (m_df.groupby("city").agg(n=("size_cr", "size"), r=("size_cr", "sum"))
      .reset_index().sort_values("r", ascending=False).head(8)[::-1])
if not cd.empty:
    l2.plotly_chart(go.Figure(
        go.Bar(y=cd["city"], x=cd["r"], orientation="h", marker_color=ACCENT,
               text=cd["r"].map(lambda v: f"{v:,.0f}"), textposition="outside"),
        layout=dict(template="plotly_white", title="Top cities (Rs. Cr)")),
        use_container_width=True)
else:
    l2.info("No city data this month.")
sz = (m_df.groupby("size_band").agg(n=("size_cr", "size"), r=("size_cr", "sum"))
      .reindex(BUCKETS).fillna(0).reset_index())
if count:
    r2.plotly_chart(go.Figure(
        go.Bar(x=sz["size_band"], y=sz["n"], marker_color=ACCENT, text=sz["n"], textposition="outside"),
        layout=dict(template="plotly_white", title="Issue size distribution (count)", xaxis=dict(tickangle=-20))),
        use_container_width=True)
else:
    r2.info("No size data this month.")


# ==========================================================
# TABLES (month)
# ==========================================================
t1, t2 = st.columns(2)
t1.subheader("Top issues")
top = m_df.sort_values("size_cr", ascending=False)[["company_name", "board", "sector", "city", "size_cr"]].head(10)
t1.dataframe(top, use_container_width=True, hide_index=True)
t2.subheader("Merchant banker league table")
bk = (m_df.assign(banker=m_df["merchant_banker"].str.split(r",|;|&")).explode("banker")
      .assign(banker=lambda x: x["banker"].str.strip()))
bk = bk[~bk["banker"].str.lower().isin(["", "nan", "none", "null"])]
bk = (bk.groupby("banker").agg(n=("size_cr", "size"), r=("size_cr", "sum"))
      .sort_values("r", ascending=False).reset_index().head(10))
t2.dataframe(bk, use_container_width=True, hide_index=True)


# ==========================================================
# FULL LISTING (month)
# ==========================================================
st.subheader(f"Full Listing - {fmt_month(period)}")
if not m_df.empty:
    listing = (m_df.sort_values("size_cr", ascending=False)
               [["company_name", "board", "sector", "city", "size_cr", "merchant_banker", "listing_date", "ipo_url"]])
    st.dataframe(listing, use_container_width=True, hide_index=True, height=420,
                 column_config={
                     "company_name": st.column_config.TextColumn("Company"),
                     "board": st.column_config.TextColumn("Board"),
                     "sector": st.column_config.TextColumn("Sector"),
                     "city": st.column_config.TextColumn("City"),
                     "size_cr": st.column_config.NumberColumn("Size (Rs. Cr)", format="%.0f"),
                     "merchant_banker": st.column_config.TextColumn("Merchant Banker"),
                     "listing_date": st.column_config.TextColumn("Listing Date"),
                     "ipo_url": st.column_config.LinkColumn("IPO URL", display_text="Open"),
                 })
else:
    st.info("No listings for the selected filters.")
st.download_button("⬇️ Export filtered data (CSV)", fdf.to_csv(index=False),
                   file_name=f"MS_Kapital_IPO_filtered_{period}.csv", mime="text/csv")


# ==========================================================
# AI MARKET INTELLIGENCE
# ==========================================================
st.divider()
st.subheader("AI Market Intelligence")
if groq_analyzer:
    tab_m, tab_y = st.tabs(["Monthly AI", "Yearly AI"])
    with tab_m:
        if st.button("Generate Monthly AI Analysis", key="gm"):
            with st.spinner("Calling Groq..."):
                st.session_state["ai_m"] = groq_analyzer.generate_monthly_insights(df, period, groq_key or None)
        if st.session_state.get("ai_m"):
            st.markdown(st.session_state["ai_m"])
        else:
            st.info("Click to generate an AI analysis for the selected month.")
    with tab_y:
        sy = st.selectbox("Year", all_years, key="ai_year")
        if st.button("Generate Yearly AI Analysis", key="gy"):
            with st.spinner("Calling Groq..."):
                st.session_state["ai_y"] = groq_analyzer.generate_yearly_insights(df, sy, groq_key or None)
        if st.session_state.get("ai_y"):
            st.markdown(st.session_state["ai_y"])
        else:
            st.info("Click to generate an AI analysis for the selected year.")
else:
    st.warning("groq_analyzer.py not found - AI disabled.")


# ==========================================================
# PDF EXPORTS  (two options: WITH AI / WITHOUT AI)
# ==========================================================
st.divider()
apply_filters_pdf = st.checkbox("Apply dashboard filters to PDF reports", value=False)
pdf_data = fdf if apply_filters_pdf else df


def _sig(extra):
    return (extra, period, tuple(boards), tuple(sectors), tuple(cities),
            tuple(bands), tuple(bankers), search, apply_filters_pdf)


def _build_monthly(data, period_, with_ai):
    if report_builder_mod is None:
        raise RuntimeError("report_builder.py not found")
    if with_ai:
        return report_builder_mod.build_monthly_pdf(data, period_)
    old = report_builder_mod.groq_analyzer
    report_builder_mod.groq_analyzer = None
    try:
        return report_builder_mod.build_monthly_pdf(data, period_)
    finally:
        report_builder_mod.groq_analyzer = old


def _build_yearly(data, year_, with_ai):
    if yearly_report_builder_mod is None:
        raise RuntimeError("yearly_report_builder.py not found")
    if with_ai:
        return yearly_report_builder_mod.build_yearly_pdf(data, year_)
    old = yearly_report_builder_mod.groq_analyzer
    yearly_report_builder_mod.groq_analyzer = None
    try:
        return yearly_report_builder_mod.build_yearly_pdf(data, year_)
    finally:
        yearly_report_builder_mod.groq_analyzer = old


# ---------- MONTHLY ----------
st.subheader(f"Monthly Market Update PDF - {fmt_month(period)}")
if build_monthly_pdf:
    mA, mB = st.columns(2)
    with mA:
        st.markdown("**Option 1 — With AI analysis**")
        if st.button("Generate (with AI)", key="gen_m_ai"):
            with st.spinner("Building monthly report with AI..."):
                try:
                    st.session_state["m_pdf_ai"] = {"sig": _sig("M_AI"),
                                                    "bytes": _build_monthly(pdf_data, period, True)}
                except Exception as e:
                    st.error(f"Monthly PDF (with AI) failed: {e}")
        mp_ai = st.session_state.get("m_pdf_ai")
        if mp_ai and mp_ai["sig"] == _sig("M_AI"):
            st.download_button("⬇️ Download Monthly PDF (with AI)", mp_ai["bytes"],
                               file_name=f"MS_Kapital_IPO_Monthly_{period}_with_AI.pdf",
                               mime="application/pdf", key="dl_m_ai")
    with mB:
        st.markdown("**Option 2 — Without AI analysis**")
        if st.button("Generate (without AI)", key="gen_m_noai"):
            with st.spinner("Building monthly report (no AI)..."):
                try:
                    st.session_state["m_pdf_noai"] = {"sig": _sig("M_NOAI"),
                                                      "bytes": _build_monthly(pdf_data, period, False)}
                except Exception as e:
                    st.error(f"Monthly PDF (no AI) failed: {e}")
        mp_no = st.session_state.get("m_pdf_noai")
        if mp_no and mp_no["sig"] == _sig("M_NOAI"):
            st.download_button("⬇️ Download Monthly PDF (without AI)", mp_no["bytes"],
                               file_name=f"MS_Kapital_IPO_Monthly_{period}_no_AI.pdf",
                               mime="application/pdf", key="dl_m_noai")
else:
    st.error("report_builder.py not found.")


# ---------- YEARLY ----------
st.subheader("Yearly Market Update PDF")
if build_yearly_pdf:
    sel_y = st.selectbox("Report year", all_years, key="pdf_year")
    yA, yB = st.columns(2)
    with yA:
        st.markdown("**Option 1 — With AI analysis**")
        if st.button("Generate (with AI)", key="gen_y_ai"):
            with st.spinner("Building yearly report with AI..."):
                try:
                    st.session_state["y_pdf_ai"] = {"sig": _sig(f"Y{sel_y}_AI"),
                                                    "bytes": _build_yearly(pdf_data, sel_y, True)}
                except Exception as e:
                    st.error(f"Yearly PDF (with AI) failed: {e}")
        yp_ai = st.session_state.get("y_pdf_ai")
        if yp_ai and yp_ai["sig"] == _sig(f"Y{sel_y}_AI"):
            st.download_button("⬇️ Download Yearly PDF (with AI)", yp_ai["bytes"],
                               file_name=f"MS_Kapital_IPO_Yearly_{sel_y}_with_AI.pdf",
                               mime="application/pdf", key="dl_y_ai")
    with yB:
        st.markdown("**Option 2 — Without AI analysis**")
        if st.button("Generate (without AI)", key="gen_y_noai"):
            with st.spinner("Building yearly report (no AI)..."):
                try:
                    st.session_state["y_pdf_noai"] = {"sig": _sig(f"Y{sel_y}_NOAI"),
                                                      "bytes": _build_yearly(pdf_data, sel_y, False)}
                except Exception as e:
                    st.error(f"Yearly PDF (no AI) failed: {e}")
        yp_no = st.session_state.get("y_pdf_noai")
        if yp_no and yp_no["sig"] == _sig(f"Y{sel_y}_NOAI"):
            st.download_button("⬇️ Download Yearly PDF (without AI)", yp_no["bytes"],
                               file_name=f"MS_Kapital_IPO_Yearly_{sel_y}_no_AI.pdf",
                               mime="application/pdf", key="dl_y_noai")
else:
    st.error("yearly_report_builder.py not found.")
