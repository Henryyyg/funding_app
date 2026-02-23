import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime
import streamlit as st

# -----------------------
# Config
# -----------------------
NYFED_READ = "https://markets.newyorkfed.org/read"
PRODUCT_CODE_REFERENCE_RATES = 50  # NY Fed reference rates product
LIMIT = 10

# Reference rate event codes (NY Fed)
EVENTS = {
    "EFFR": 500,
    "OBFR": 505,
    "TGCR": 510,
    "BGCR": 515,
    "SOFR": 520,
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "text/csv,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# -----------------------
# Helpers
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
    if x >= 1000:
        return f"USD {x/1000:.3f}tln".rstrip("0").rstrip(".").replace("tln", "tln")
    # keep 1dp for bln, but if < 1 then 2dp (e.g. 0.13bln)
    if x < 1:
        return f"USD {x:.2f}bln".rstrip("0").rstrip(".") + "bln"
    return f"USD {x:.1f}bln".rstrip("0").rstrip(".") + "bln"

def fetch_reference_rate(event_code: int, limit: int = 10) -> pd.DataFrame:
    """
    Pulls last `limit` observations for one NY Fed reference rate.
    Returns a standardized df with columns: effectiveDate, type, percentRate, volume_bln, target_from, target_to
    """
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

    # Expected columns (from NY Fed export):
    # Effective Date, Rate Type, Rate (%), Volume ($Billions), Target Rate From (%), Target Rate To (%)
    # Some rates won't have target cols populated; EFFR usually does.
    # Normalize
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
    # df expected sorted desc by effectiveDate
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
    """
    dfs: {"EFFR": df, ...}
    returns wide table with 10 rows (dates) and columns like 'EFFR Rate', 'EFFR Vol'
    """
    frames = []
    for k in tickers:
        d = dfs[k].copy()
        d = d[["effectiveDate", "percentRate", "volume_bln"]]
        d = d.rename(columns={
            "percentRate": f"{k} Rate (%)",
            "volume_bln": f"{k} Volume ($bln)",
        })
        frames.append(d)

    # Outer join on effectiveDate
    merged = frames[0]
    for f in frames[1:]:
        merged = pd.merge(merged, f, on="effectiveDate", how="outer")

    merged = merged.sort_values("effectiveDate", ascending=False).head(LIMIT)

    # Pretty formatting for display
    disp = merged.copy()
    disp["Date"] = disp["effectiveDate"].dt.date.astype(str)
    disp = disp.drop(columns=["effectiveDate"])

    # Put Date first
    cols = ["Date"] + [c for c in disp.columns if c != "Date"]
    disp = disp[cols]

    # format volumes as USD bln/tln strings (keep rate numeric)
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
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Funding Snapshot", layout="wide")
st.title("Funding Snapshot")

with st.spinner("Pulling NY Fed reference rates..."):
    dfs = {k: fetch_reference_rate(v, LIMIT) for k, v in EVENTS.items()}

# Fed funds target range (from latest EFFR row)
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

# Tables
overnight_keys = ["EFFR", "OBFR"]
secured_keys = ["TGCR", "BGCR", "SOFR"]

st.subheader("Overnight Rates (last 10)")
overnight_tbl = make_table(dfs, overnight_keys)
st.dataframe(format_table_for_display(overnight_tbl), use_container_width=True)

st.subheader("Secured Rates (last 10)")
secured_tbl = make_table(dfs, secured_keys)
st.dataframe(format_table_for_display(secured_tbl), use_container_width=True)

# Headlines (your wrap-style)
st.subheader("Headlines")
headlines = [
    headline_line("EFFR", dfs["EFFR"]),
    headline_line("OBFR", dfs["OBFR"]),
    headline_line("TGCR", dfs["TGCR"]),
    headline_line("BGCR", dfs["BGCR"]),
    headline_line("SOFR", dfs["SOFR"]),
]
st.code("\n".join(headlines), language="text")