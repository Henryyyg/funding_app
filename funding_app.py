import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime
import streamlit as st

# -----------------------
# Global Headers (must be defined before fetch_ops_csv uses it)
# -----------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "text/csv,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# -----------------------
# Simple formatters used by headlines
# -----------------------
def fmt_bln_2dp(x):
    if pd.isna(x):
        return "n/a"
    return f"{float(x):.2f}bln"

def fmt_usd_bln_two_dp(x_bln):
    if pd.isna(x_bln):
        return ""
    return f"{float(x_bln):.2f}bln"

# -----------------------
# NY Fed endpoints
# -----------------------
NYFED_READ = "https://markets.newyorkfed.org/read"
PRODUCT_CODE_OPS = 70
PRODUCT_CODE_REFERENCE_RATES = 50  # NY Fed reference rates product

# Reference rate event codes (NY Fed)
EVENTS = {
    "EFFR": 500,
    "OBFR": 505,
    "TGCR": 510,
    "BGCR": 515,
    "SOFR": 520,
}
LIMIT = 10

# -----------------------
# Shared ops helpers
# -----------------------
def _first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _first_non_nan(row, candidates):
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return row[c]
    return np.nan

def fetch_ops_csv(operation_type: str, limit: int = 200) -> pd.DataFrame:
    """
    operation_type: "Repo" or "Reverse Repo"
    Returns raw ops dataframe from NY Fed read endpoint.
    """
    params = {
        "format": "csv",
        "operationTypes": operation_type,
        "productCode": str(PRODUCT_CODE_OPS),
        "sort": "postDt:-1,'data.closeTm':-1",
        "limit": str(limit),
        "startPosition": "0",
    }
    r = requests.get(NYFED_READ, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    df.columns = [str(c).strip() for c in df.columns]
    return df

def add_am_pm_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses Close Time (preferred) else Release Time to label AM/PM by hour>=12.
    """
    df = df.copy()
    time_col = _first_existing_col(df, ["Close Time", "Release Time"])
    if time_col:
        t = pd.to_datetime(df[time_col], errors="coerce")
        df["AM/PM"] = np.where((pd.notna(t) & (t.dt.hour >= 12)), "PM", "AM")
    else:
        df["AM/PM"] = ""  # fallback
    return df

# -----------------------
# Helpers (reference rates)
# -----------------------
def fmt_date(d: pd.Timestamp) -> str:
    # "February 17th"
    d = pd.to_datetime(d).date()
    day = d.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{d.strftime('%B')} {day}{suffix}"

def fmt_volume_from_bln(x_bln: float) -> str:
    if pd.isna(x_bln):
        return "USD n/a"

    x = float(x_bln)

    # trillions
    if x >= 1000:
        return f"USD {x/1000:.3f}".rstrip("0").rstrip(".") + "tln"

    # less than 1bln → 2dp
    if x < 1:
        return f"USD {x:.2f}".rstrip("0").rstrip(".") + "bln"

    # 1bln+ → 1dp
    return f"USD {x:.1f}".rstrip("0").rstrip(".") + "bln"

def fetch_reference_rate(event_code: int, limit: int = 10) -> pd.DataFrame:
    params = {
        "eventCodes": str(event_code),
        "productCode": str(PRODUCT_CODE_REFERENCE_RATES),
        "format": "csv",
        "limit": str(limit),
        "sort": "postDt:-1",
        "startPosition": "0",
    }
    r = requests.get(NYFED_READ, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    df.columns = [str(c).strip() for c in df.columns]

    col_date = "Effective Date"
    col_type = "Rate Type"
    col_rate = "Rate (%)"
    col_vol  = "Volume ($Billions)"
    col_tfrom = "Target Rate From (%)"
    col_tto   = "Target Rate To (%)"

    for c in [col_date, col_type, col_rate]:
        if c not in df.columns:
            raise ValueError(f"Missing expected column '{c}' in NY Fed response. Got: {list(df.columns)}")

    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    df[col_rate] = pd.to_numeric(df[col_rate], errors="coerce")
    df[col_vol]  = pd.to_numeric(df[col_vol], errors="coerce") if col_vol in df.columns else np.nan

    if col_tfrom in df.columns:
        df[col_tfrom] = pd.to_numeric(df[col_tfrom], errors="coerce")
    else:
        df[col_tfrom] = np.nan

    if col_tto in df.columns:
        df[col_tto] = pd.to_numeric(df[col_tto], errors="coerce")
    else:
        df[col_tto] = np.nan

    out = pd.DataFrame({
        "effectiveDate": df[col_date],
        "type": df[col_type].astype(str).str.strip(),
        "percentRate": df[col_rate],
        "volume_bln": df[col_vol] if col_vol in df.columns else np.nan,
        "target_from": df[col_tfrom],
        "target_to": df[col_tto],
    })

    out = out.dropna(subset=["effectiveDate"]).sort_values("effectiveDate", ascending=False).head(limit)
    return out

def headline_line(label: str, df: pd.DataFrame) -> str:
    if df is None or df.empty or len(df) < 2:
        return f"{label}: Pending"

    latest = df.iloc[0]
    prev = df.iloc[1]

    return (
        f"{label} at {latest['percentRate']:.2f}% (prev. {prev['percentRate']:.2f}%), "
        f"volumes at {fmt_volume_from_bln(latest['volume_bln'])} (prev. {fmt_volume_from_bln(prev['volume_bln'])}) "
        f"on {fmt_date(latest['effectiveDate'])}"
    )

def make_table(dfs: dict, tickers: list) -> pd.DataFrame:
    frames = []
    for k in tickers:
        d = dfs[k].copy()
        d = d[["effectiveDate", "percentRate", "volume_bln"]]
        d = d.rename(columns={
            "percentRate": f"{k} Rate (%)",
            "volume_bln": f"{k} Volume ($bln)",
        })
        frames.append(d)

    merged = frames[0]
    for f in frames[1:]:
        merged = pd.merge(merged, f, on="effectiveDate", how="outer")

    merged = merged.sort_values("effectiveDate", ascending=False).head(LIMIT)

    disp = merged.copy()
    disp["Date"] = disp["effectiveDate"].dt.date.astype(str)
    disp = disp.drop(columns=["effectiveDate"])

    cols = ["Date"] + [c for c in disp.columns if c != "Date"]
    disp = disp[cols]

    for c in disp.columns:
        if c.endswith("Volume ($bln)"):
            disp[c] = disp[c].apply(lambda x: np.nan if pd.isna(x) else float(x))
    return disp

def format_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c.endswith("Rate (%)"):
            out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        if c.endswith("Volume ($bln)"):
            out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda x: "" if pd.isna(x) else fmt_volume_from_bln(x))
    return out

# -----------------------
# RRP table + headline
# -----------------------
def build_rrp_table(last_n_dates: int = 10) -> pd.DataFrame:
    df = fetch_ops_csv("Reverse Repo", limit=500)

    if "Auction Status" in df.columns:
        df = df[df["Auction Status"].astype(str).str.strip().str.lower() == "results"]

    date_col = "Operation Date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if "Operation Method" in df.columns:
        df = df[~df["Operation Method"].astype(str).str.contains("small value", case=False, na=False)]
    if "Note" in df.columns:
        df = df[~df["Note"].astype(str).str.contains("small value", case=False, na=False)]

    submitted_col = _first_existing_col(df, ["Total Amt Submitted ($Billions)"])
    accepted_col  = _first_existing_col(df, ["Total Amt Accepted ($Billions)"])
    cpty_col      = _first_existing_col(df, ["Participating Counterparties"])
    if submitted_col: df[submitted_col] = pd.to_numeric(df[submitted_col], errors="coerce")
    if accepted_col:  df[accepted_col]  = pd.to_numeric(df[accepted_col], errors="coerce")
    if cpty_col:      df[cpty_col]      = pd.to_numeric(df[cpty_col], errors="coerce")

    rate_candidates = [
        "Tsy Offering Rate(%)",
        "Tsy Award Rate (%)",
        "Tsy Stop-Out Rate(%)",
        "Tsy Weighted Average Rate (%)",
        "Tsy Minimum Bid Rate(%)",
        "Tsy Maximum Bid Rate(%)",
    ]
    for c in rate_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(date_col, ascending=False)
    dates = df[date_col].dt.normalize().drop_duplicates().head(last_n_dates).tolist()
    df = df[df[date_col].dt.normalize().isin(dates)].copy()

    if accepted_col:
        df = df.sort_values([date_col, accepted_col], ascending=[False, False]).drop_duplicates(subset=[date_col], keep="first")
    else:
        df = df.drop_duplicates(subset=[date_col], keep="first")

    out = pd.DataFrame({
        "Date": df[date_col].dt.date.astype(str),
        "Rate (%)": df.apply(lambda r: _first_non_nan(r, rate_candidates), axis=1),
        "Amt Submitted": df[submitted_col] if submitted_col else np.nan,
        "Amt Accepted": df[accepted_col] if accepted_col else np.nan,
        "Counterparties": df[cpty_col] if cpty_col else np.nan,
    })

    out["Rate (%)"] = pd.to_numeric(out["Rate (%)"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    out["Amt Submitted"] = pd.to_numeric(out["Amt Submitted"], errors="coerce").map(lambda x: "" if pd.isna(x) else fmt_volume_from_bln(x))
    out["Amt Accepted"]  = pd.to_numeric(out["Amt Accepted"], errors="coerce").map(lambda x: "" if pd.isna(x) else fmt_volume_from_bln(x))
    out["Counterparties"] = pd.to_numeric(out["Counterparties"], errors="coerce").map(lambda x: "" if pd.isna(x) else str(int(x)))

    return out

def build_rrp_headline():
    df = fetch_ops_csv("Reverse Repo", limit=600)

    if "Auction Status" in df.columns:
        df = df[df["Auction Status"].astype(str).str.strip().str.lower() == "results"]

    dcol = "Operation Date"
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).sort_values(dcol, ascending=False)

    if "Operation Method" in df.columns:
        df = df[~df["Operation Method"].astype(str).str.contains("small value", case=False, na=False)]
    if "Note" in df.columns:
        df = df[~df["Note"].astype(str).str.contains("small value", case=False, na=False)]

    acc_col = _first_existing_col(df, ["Total Amt Accepted ($Billions)"])
    cpty_col = _first_existing_col(df, ["Participating Counterparties"])

    if acc_col is None or cpty_col is None or df.empty:
        return "NY Fed RRP op: Pending"

    df[acc_col] = pd.to_numeric(df[acc_col], errors="coerce")
    df[cpty_col] = pd.to_numeric(df[cpty_col], errors="coerce")

    df = df.sort_values([dcol, acc_col], ascending=[False, False]) \
           .drop_duplicates(subset=[dcol], keep="first")

    if len(df) < 2:
        return "NY Fed RRP op: Pending"

    latest = df.iloc[0]
    prev = df.iloc[1]

    latest_acc = fmt_bln_2dp(latest[acc_col])
    prev_acc = fmt_bln_2dp(prev[acc_col])

    latest_cpty = int(latest[cpty_col]) if pd.notna(latest[cpty_col]) else "n/a"
    prev_cpty = int(prev[cpty_col]) if pd.notna(prev[cpty_col]) else "n/a"

    date_txt = fmt_date(latest[dcol])

    return (
        f"NY Fed RRP op demand at {latest_acc} (prev. {prev_acc}) "
        f"across {latest_cpty} counterparties (prev. {prev_cpty}) "
        f"on {date_txt}"
    )

# --------------------------
# Repo table + headlines (AM/PM)
# --------------------------
def build_repo_table(last_n_days: int = 10) -> pd.DataFrame:
    df = fetch_ops_csv("Repo", limit=800)

    if "Auction Status" in df.columns:
        df = df[df["Auction Status"].astype(str).str.strip().str.lower() == "results"]

    date_col = "Operation Date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if "Operation Method" in df.columns:
        df = df[~df["Operation Method"].astype(str).str.contains("small value", case=False, na=False)]
    if "Note" in df.columns:
        df = df[~df["Note"].astype(str).str.contains("small value", case=False, na=False)]

    df = add_am_pm_bucket(df)

    tsy_col = _first_existing_col(df, ["Tsy Amt Accepted ($Billions)", "Total Tsy Settle Amt Accepted ($Billions)"])
    agy_col = _first_existing_col(df, ["Agy Amt Accepted ($Billions)", "Total Agy Settle Amt Accepted ($Billions)"])
    mbs_col = _first_existing_col(df, ["Mbs Amt Accepted ($Billions)", "Total Mbs Settle Amt Accepted ($Billions)"])

    for c in [tsy_col, agy_col, mbs_col]:
        if c:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rate_candidates = [
        "Minimum Bid Rate(%)",
        "Offering Rate(%)",
        "Stop-Out Rate(%)",
        "Award Rate (%)",
        "Weighted Average Rate (%)",
        "Tsy Minimum Bid Rate(%)",
        "Tsy Offering Rate(%)",
        "Tsy Stop-Out Rate(%)",
        "Tsy Award Rate (%)",
        "Tsy Weighted Average Rate (%)",
    ]
    for c in rate_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values([date_col], ascending=False)
    days = df[date_col].dt.normalize().drop_duplicates().head(last_n_days).tolist()
    df = df[df[date_col].dt.normalize().isin(days)].copy()

    key_cols = [date_col, "AM/PM"]
    if "Operation Id" in df.columns:
        key_cols = ["Operation Id"]

    df = df.sort_values([date_col, tsy_col] if tsy_col else [date_col],
                        ascending=[False, False] if tsy_col else [False])
    df = df.drop_duplicates(subset=key_cols, keep="first")

    date_label = df[date_col].dt.date.astype(str) + " " + df["AM/PM"].astype(str)

    out = pd.DataFrame({
        "Date": date_label,
        "Rate (%)": df.apply(lambda r: _first_non_nan(r, rate_candidates), axis=1),
        "Tsy Accepted": df[tsy_col] if tsy_col else np.nan,
        "Agy Accepted": df[agy_col] if agy_col else np.nan,
        "MBS Accepted": df[mbs_col] if mbs_col else np.nan,
    })

    out["Rate (%)"] = pd.to_numeric(out["Rate (%)"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    for c in ["Tsy Accepted", "Agy Accepted", "MBS Accepted"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda x: "" if pd.isna(x) else fmt_usd_bln_two_dp(x))

    return out.sort_values("Date", ascending=False)

def build_repo_headlines_am_pm():
    df = fetch_ops_csv("Repo", limit=900)

    if "Auction Status" in df.columns:
        df = df[df["Auction Status"].astype(str).str.strip().str.lower() == "results"]

    dcol = "Operation Date"
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol])

    # exclude SVEs
    if "Operation Method" in df.columns:
        df = df[~df["Operation Method"].astype(str)
                .str.contains("small value", case=False, na=False)]
    if "Note" in df.columns:
        df = df[~df["Note"].astype(str)
                .str.contains("small value", case=False, na=False)]

    # Bucket AM / PM
    time_col = _first_existing_col(df, ["Close Time", "Release Time"])
    if time_col:
        t = pd.to_datetime(df[time_col], errors="coerce")
        df["AM/PM"] = np.where((pd.notna(t) & (t.dt.hour >= 12)), "PM", "AM")
    else:
        df["AM/PM"] = "AM"

    df = df.sort_values(dcol, ascending=False)
    latest_day = df.iloc[0][dcol].normalize()
    day = df[df[dcol].dt.normalize() == latest_day].copy()

    tsy_col = _first_existing_col(day, ["Tsy Amt Accepted ($Billions)", "Total Tsy Settle Amt Accepted ($Billions)"])
    agy_col = _first_existing_col(day, ["Agy Amt Accepted ($Billions)", "Total Agy Settle Amt Accepted ($Billions)"])
    mbs_col = _first_existing_col(day, ["Mbs Amt Accepted ($Billions)", "Total Mbs Settle Amt Accepted ($Billions)"])
    tot_col = _first_existing_col(day, ["Total Amt Accepted ($Billions)", "Total US Dollar Settle Amt Accepted ($Billions)"])

    for c in [tsy_col, agy_col, mbs_col, tot_col]:
        if c and c in day.columns:
            day[c] = pd.to_numeric(day[c], errors="coerce")

    rate_candidates = [
        "Minimum Bid Rate(%)", "Offering Rate(%)", "Stop-Out Rate(%)",
        "Award Rate (%)", "Weighted Average Rate (%)",
        "Tsy Minimum Bid Rate(%)", "Tsy Offering Rate(%)",
        "Tsy Stop-Out Rate(%)", "Tsy Award Rate (%)",
        "Tsy Weighted Average Rate (%)",
    ]
    for c in rate_candidates:
        if c in day.columns:
            day[c] = pd.to_numeric(day[c], errors="coerce")

    if tot_col:
        day = (
            day.sort_values([dcol, "AM/PM", tot_col],
                            ascending=[False, True, False])
               .drop_duplicates(subset=["AM/PM"], keep="first")
        )
    else:
        day = day.drop_duplicates(subset=["AM/PM"], keep="first")

    out = {"AM": None, "PM": None}

    for _, r in day.iterrows():
        total = fmt_bln_2dp(r[tot_col]) if (tot_col and pd.notna(r.get(tot_col))) else "0.00bln"
        tsy = fmt_bln_2dp(r[tsy_col]) if (tsy_col and pd.notna(r.get(tsy_col))) else "0.00bln"
        agy = fmt_bln_2dp(r[agy_col]) if (agy_col and pd.notna(r.get(agy_col))) else "0.00bln"
        mbs = fmt_bln_2dp(r[mbs_col]) if (mbs_col and pd.notna(r.get(mbs_col))) else "0.00bln"

        rate_val = _first_non_nan(r, rate_candidates)
        rate_txt = f"{float(rate_val):.2f}%" if pd.notna(rate_val) else "n/a"

        date_txt = fmt_date(r[dcol])

        out[r["AM/PM"]] = (
            f"{r['AM/PM']} NY Fed Repo op demand at {total} "
            f"(Tsy: {tsy}, Agency: {agy}, MBS: {mbs}) "
            f"at {rate_txt} on {date_txt}"
        )

    final = []
    for slot in ["AM", "PM"]:
        if out[slot] is None:
            final.append(f"{slot} NY Fed Repo Op: Pending ({fmt_date(latest_day)})")
        else:
            final.append(out[slot])

    return final

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Funding Snapshot", layout="wide")
st.title("Funding Snapshot")

with st.spinner("Pulling NY Fed reference rates..."):
    dfs = {k: fetch_reference_rate(v, LIMIT) for k, v in EVENTS.items()}

effr = dfs["EFFR"]
target_from = effr.iloc[0]["target_from"] if not effr.empty else np.nan
target_to = effr.iloc[0]["target_to"] if not effr.empty else np.nan
target_date = effr.iloc[0]["effectiveDate"] if not effr.empty else pd.NaT

colA, colB = st.columns([1, 3])
with colA:
    if pd.notna(target_from) and pd.notna(target_to):
        st.metric("Fed funds target range", f"{target_from:.2f}%–{target_to:.2f}%")
        st.caption(f"From NY Fed EFFR table (as of {fmt_date(target_date)})")
    else:
        st.metric("Fed funds target range", "n/a")
        st.caption("Target range not available in latest EFFR row.")

with colB:
    st.caption("Overnight Rates: EFFR, OBFR · Secured Rates: TGCR, BGCR, SOFR")

overnight_keys = ["EFFR", "OBFR"]
secured_keys = ["TGCR", "BGCR", "SOFR"]

# Auto highlight helpers
def highlight_outside_target(val, lower, upper):
    try:
        v = float(val)
    except:
        return ""
    if pd.isna(lower) or pd.isna(upper):
        return ""
    if v > upper:
        return "background-color: #ffdddd"
    if v < lower:
        return "background-color: #dde8ff"
    return ""

st.subheader("Unsecured Funding Rates (last 10, T-1)")
overnight_tbl = make_table(dfs, overnight_keys)
overnight_display = format_table_for_display(overnight_tbl)

if pd.notna(target_from) and pd.notna(target_to):
    styled = overnight_display.style.applymap(
        lambda v: highlight_outside_target(v, target_from, target_to),
        subset=["EFFR Rate (%)", "OBFR Rate (%)"]
    )
    st.dataframe(styled, use_container_width=True)
else:
    st.dataframe(overnight_display, use_container_width=True)

st.subheader("Secured Funding Rates (last 10, T-1)")
secured_tbl = make_table(dfs, secured_keys)
st.dataframe(format_table_for_display(secured_tbl), use_container_width=True)

st.subheader("NY Fed Reverse Repo Operations (T-0)")
rrp_tbl = build_rrp_table(last_n_dates=10)
st.dataframe(rrp_tbl, use_container_width=True)

st.subheader("NY Fed Repo Operations (T-0)")
repo_tbl = build_repo_table(last_n_days=10)
st.dataframe(repo_tbl, use_container_width=True)

with st.expander("Rate & Operations Definitions"):
    st.markdown("""
### Reference Rates
**EFFR** – Effective Federal Funds Rate: Volume-weighted median rate of overnight unsecured federal funds transactions.  
**OBFR** – Overnight Bank Funding Rate: Volume-weighted median rate of overnight federal funds and Eurodollar transactions.  
**TGCR** – Tri-Party General Collateral Rate: Rate on overnight Treasury repo cleared via the tri-party platform.  
**BGCR** – Broad General Collateral Rate: Broader Treasury GC repo rate including FICC-cleared bilateral trades.  
**SOFR** – Secured Overnight Financing Rate: Broad measure of overnight Treasury repo financing costs.

---
### Federal Reserve Policy
**Fed Funds Target Range** – The FOMC’s target range for the federal funds rate. Displayed for reference against EFFR and OBFR.

---
### NY Fed Operations
**RRP (Reverse Repo Operation)** – Overnight facility where the Fed borrows cash from counterparties in exchange for Treasury collateral.  
Displayed: rate, total submitted, total accepted, and participating counterparties.  

**Repo Operation** – Overnight facility where the Fed lends cash against Treasury, Agency, or MBS collateral.  
Displayed: rate and amounts accepted by collateral type (Tsy, Agency, MBS). Typically two operations per day (AM and PM).

---
### Notes
• Reference rate data is effective T-1 (published with a one-day lag).  
• Repo and RRP operations reflect same-day published results.  
• Small Value Exercises (SVEs) are excluded from displayed operations.
""")

# -----------------------
# Headlines (structured)
# -----------------------

t1_lines = [
    headline_line("EFFR", dfs["EFFR"]),
    headline_line("OBFR", dfs["OBFR"]),
    headline_line("TGCR", dfs["TGCR"]),
    headline_line("BGCR", dfs["BGCR"]),
    headline_line("SOFR", dfs["SOFR"]),
]

t0_lines = [
    build_rrp_headline(),
]
t0_lines.extend(build_repo_headlines_am_pm())

headline_block = (
    "T-1 Reference Rates\n"
    + "\n".join(t1_lines)
    + "\n\n"
    + "T-0 NY Fed Operations\n"
    + "\n".join(t0_lines)
)

with st.expander("Headlines (wrap format)", expanded=False):
    st.code(headline_block, language="text")
