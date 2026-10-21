"""
ACCT 3343 · Analysis App
Screening Consumer Discretionary Stocks: Red Flags vs Hidden Value By Ava Hunter 

SELL side (Chapters 3–4): Scaled total accruals (STA), Scaled net operating assets (SNOA),
                          Beneish M-score / PMAN, Altman Z.  A firm is a SELL candidate when it
                          raises at least N of these red flags.
BUY side  (Chapter 2):    Gross profitability (GPA) and Book-to-market (B/M), ranked within the
                          screened universe and averaged into one combined score.
"""
import math

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Consumer Discretionary Screener", layout="wide")

# --------------------------------------------------------------------------------------
# Companies that may never be recommended (e.g. my 60 Second Pitch company).
# Put the ticker(s) here; they are removed before anything is calculated.
EXCLUDE = []          # e.g. ["NKE"]
# --------------------------------------------------------------------------------------


# ======================================================================================
# 1 · LOAD DATA AND CALCULATE MEASURES
# Measures don't depend on the sidebar, so they're calculated once and cached.
# Missing inputs give a missing measure: there is no fillna(0) on any statement line.
# ======================================================================================
def safe_div(a, b):
    """a / b, but missing when b is missing or zero."""
    b = b.where(b != 0)
    return a / b


def norm_cdf(x):
    """Standard normal CDF, used to turn the Beneish M-score into a probability (PMAN)."""
    return x.apply(lambda v: 0.5 * (1 + math.erf(v / math.sqrt(2))) if pd.notna(v) else np.nan)


@st.cache_data
def load_data(path="universe.csv"):
    df = pd.read_csv(path)
    df = df[~df["ticker"].isin(EXCLUDE)].copy()
    df["industry"] = df["industry"].fillna("Unclassified")
    df["usd_reporter"] = df["financial_currency"].fillna("USD").eq("USD")

    ta = df["total_assets"].where(df["total_assets"] > 0)
    ta_lag = df["total_assets_lag"].where(df["total_assets_lag"] > 0)

    # Gross profit: use the reported line; if Yahoo has none, revenue − cost of revenue
    for s in ("", "_lag"):
        df["gp" + s] = df["gross_profit" + s].fillna(df["revenue" + s] - df["cogs" + s])

    # ---------- BUY measures (Chapter 2) ----------
    # GPA = Gross profit / Total assets   (Novy-Marx: higher = better quality)
    df["GPA"] = safe_div(df["gp"], ta)
    # B/M = Book equity / Market cap      (higher = cheaper)
    # Negative book equity makes B/M meaningless, so it is left missing.
    # Book value and market cap must be in the same currency, so B/M is only for USD reporters.
    book = df["common_equity"].where(df["common_equity"] > 0)
    df["BM"] = safe_div(book, df["market_cap"]).where(df["usd_reporter"])

    # ---------- SELL measures (Chapters 3–4) ----------
    # STA = (Net income − Operating cash flow) / Total assets
    df["STA"] = safe_div(df["net_income"] - df["operating_cash_flow"], ta)

    # SNOA = (Operating assets − Operating liabilities) / Lagged total assets
    #   Operating assets      = Total assets − Cash & short-term investments
    #   Operating liabilities = Total assets − Total debt − Minority interest
    #                           − Preferred stock − Common equity
    # Yahoo omits the minority-interest and preferred-stock lines when a company has none,
    # so only those two lines are treated as 0 when absent (a stated assumption).
    # Yahoo's Total debt includes lease liabilities, so store leases count as financing.
    mi = df["minority_interest"].fillna(0)
    ps = df["preferred_stock"].fillna(0)
    op_assets = df["total_assets"] - df["cash_sti"]
    op_liabs = df["total_assets"] - df["total_debt"] - mi - ps - df["common_equity"]
    df["SNOA"] = safe_div(op_assets - op_liabs, ta_lag)

    # Beneish M-score (8 variables, current year t vs prior year t−1)
    rev, rev_l = df["revenue"], df["revenue_lag"]
    dsri = safe_div(safe_div(df["receivables"], rev), safe_div(df["receivables_lag"], rev_l))
    gmi = safe_div(safe_div(df["gp_lag"], rev_l), safe_div(df["gp"], rev))
    aq = 1 - safe_div(df["current_assets"] + df["net_ppe"], ta)
    aq_l = 1 - safe_div(df["current_assets_lag"] + df["net_ppe_lag"], ta_lag)
    aqi = safe_div(aq, aq_l)
    sgi = safe_div(rev, rev_l)
    dep = safe_div(df["depreciation"], df["depreciation"] + df["net_ppe"])
    dep_l = safe_div(df["depreciation_lag"], df["depreciation_lag"] + df["net_ppe_lag"])
    depi = safe_div(dep_l, dep)
    sgai = safe_div(safe_div(df["sga"], rev), safe_div(df["sga_lag"], rev_l))
    lev = safe_div(df["current_liabilities"] + df["long_term_debt"], ta)
    lev_l = safe_div(df["current_liabilities_lag"] + df["long_term_debt_lag"], ta_lag)
    lvgi = safe_div(lev, lev_l)
    tata = safe_div(df["net_income"] - df["operating_cash_flow"], ta)
    df = df.assign(DSRI=dsri, GMI=gmi, AQI=aqi, SGI=sgi, DEPI=depi, SGAI=sgai, LVGI=lvgi, TATA=tata)
    df["M_score"] = (-4.84 + 0.920 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
                     + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    df["PMAN"] = norm_cdf(df["M_score"])

    # Altman Z (original public-company model)
    #   1.2·WC/TA + 1.4·RE/TA + 3.3·EBIT/TA + 0.6·MVE/TL + 1.0·Sales/TA
    # MVE/TL mixes market and book values, so Z is only calculated for USD reporters.
    wc = df["current_assets"] - df["current_liabilities"]
    df["Altman_Z"] = (1.2 * safe_div(wc, ta) + 1.4 * safe_div(df["retained_earnings"], ta)
                      + 3.3 * safe_div(df["ebit"], ta)
                      + 0.6 * safe_div(df["market_cap"], df["total_liabilities"])
                      + 1.0 * safe_div(rev, ta)).where(df["usd_reporter"])

    df["mcap_m"] = df["market_cap"] / 1e6
    return df


df_all = load_data()

# ======================================================================================
# 2 · SIDEBAR FILTERS  (every widget below changes who is flagged)
# ======================================================================================
st.sidebar.header("Universe")
industries = sorted(df_all["industry"].unique())
chosen_ind = st.sidebar.multiselect("Industries to include", industries, default=industries)
min_cap = st.sidebar.number_input("Minimum market cap ($ millions)", 0, 50_000, 300, step=50)
usd_only = st.sidebar.checkbox("US-dollar reporters only", value=True,
                               help="Foreign filers report in other currencies, so B/M and Altman Z "
                                    "can't be compared for them.")

st.sidebar.header("SELL: red flags")
sta_pct = st.sidebar.slider("Flag STA above percentile", 50, 99, 80)
snoa_pct = st.sidebar.slider("Flag SNOA above percentile", 50, 99, 80)
m_cut = st.sidebar.slider("Flag Beneish M-score above", -3.0, 0.0, -1.78, 0.01,
                          help="−1.78 is Beneish's cut-off; −2.22 is the stricter alternative.")
z_cut = st.sidebar.slider("Flag Altman Z below", 0.0, 4.0, 1.81, 0.01,
                          help="Below 1.81 is Altman's distress zone; 1.81–2.99 is the grey zone.")
n_flags = st.sidebar.slider("Red flags required for SELL", 1, 4, 2)

st.sidebar.header("BUY: quality + value")
buy_pct = st.sidebar.slider("Combined GPA + B/M rank at least (percentile)", 50, 99, 90)
gpa_floor = st.sidebar.slider("Minimum GPA percentile", 0, 90, 50,
                              help="Stops a very cheap but low-quality firm qualifying on value alone.")
excl_sell = st.sidebar.checkbox("Never BUY a company that is also a SELL candidate", value=True)

# ======================================================================================
# 3 · SCREEN
# Percentiles are ranked within the screened universe, so the filters above move them.
# ======================================================================================
df = df_all[df_all["industry"].isin(chosen_ind) & (df_all["mcap_m"] >= min_cap)].copy()
if usd_only:
    df = df[df["usd_reporter"]]

for m in ["STA", "SNOA", "GPA", "BM"]:
    df[m + "_pct"] = df[m].rank(pct=True) * 100

df["flag_STA"] = df["STA_pct"] >= sta_pct
df["flag_SNOA"] = df["SNOA_pct"] >= snoa_pct
df["flag_Beneish"] = df["M_score"] > m_cut
df["flag_AltmanZ"] = df["Altman_Z"] < z_cut
flag_cols = ["flag_STA", "flag_SNOA", "flag_Beneish", "flag_AltmanZ"]
# A missing measure never raises a flag (comparisons with NaN are False)
df["red_flags"] = df[flag_cols].sum(axis=1)
df["measures_available"] = df[["STA", "SNOA", "M_score", "Altman_Z"]].notna().sum(axis=1)
df["SELL"] = df["red_flags"] >= n_flags

df["buy_score"] = df[["GPA_pct", "BM_pct"]].mean(axis=1, skipna=False)
df["buy_score_pct"] = df["buy_score"].rank(pct=True) * 100
df["BUY"] = (df["buy_score_pct"] >= buy_pct) & (df["GPA_pct"] >= gpa_floor)
if excl_sell:
    df["BUY"] &= ~df["SELL"]

df["Signal"] = np.select([df["SELL"], df["BUY"]], ["SELL", "BUY"], "Neither")
sell = df[df["SELL"]].sort_values(["red_flags", "STA_pct"], ascending=False)
buy = df[df["BUY"]].sort_values("buy_score", ascending=False)

# ======================================================================================
# 4 · PAGE
# ======================================================================================
st.title("Consumer Discretionary: SELL and BUY screener")
st.caption("US-listed Consumer Discretionary companies (Yahoo Finance sector: Consumer Cyclical). "
           "SELL screens earnings quality and distress; BUY screens quality and value.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Companies screened", len(df))
c2.metric("SELL candidates", len(sell))
c3.metric("BUY candidates", len(buy))
c4.metric("Industries", df["industry"].nunique())

fmt = {"mcap_m": "{:,.0f}", "STA": "{:.3f}", "SNOA": "{:.2f}", "M_score": "{:.2f}",
       "PMAN": "{:.1%}", "Altman_Z": "{:.2f}", "GPA": "{:.3f}", "BM": "{:.3f}",
       "STA_pct": "{:.0f}", "SNOA_pct": "{:.0f}", "GPA_pct": "{:.0f}", "BM_pct": "{:.0f}",
       "buy_score": "{:.1f}"}
names = {"mcap_m": "Mkt cap ($M)", "M_score": "M-score", "Altman_Z": "Altman Z",
         "BM": "B/M", "STA_pct": "STA pctl", "SNOA_pct": "SNOA pctl", "GPA_pct": "GPA pctl",
         "BM_pct": "B/M pctl", "buy_score": "Combined score", "red_flags": "Red flags",
         "measures_available": "Measures available"}

tab_screen, tab_chart, tab_drill, tab_conc, tab_method = st.tabs(
    ["Screen", "Chart", "Company drill-down", "Conclusions", "Methodology"])

with tab_screen:
    st.subheader(f"SELL candidates ({len(sell)})")
    st.caption(f"At least {n_flags} of 4 red flags: STA ≥ {sta_pct}th pctl · SNOA ≥ {snoa_pct}th pctl · "
               f"M-score > {m_cut:.2f} · Altman Z < {z_cut:.2f}")
    sell_cols = ["ticker", "name", "industry", "mcap_m", "red_flags", "measures_available",
                 "STA", "STA_pct", "SNOA", "SNOA_pct", "M_score", "PMAN", "Altman_Z"] + flag_cols
    st.dataframe(sell[sell_cols].style.format({k: v for k, v in fmt.items() if k in sell_cols}, na_rep="—").format_index(lambda c: names.get(c, c), axis=1),
                 hide_index=True, width="stretch")
    st.download_button("Download SELL list", sell[sell_cols].to_csv(index=False), "sell.csv")

    st.subheader(f"BUY candidates ({len(buy)})")
    st.caption(f"Average of GPA and B/M percentile ranks in the top {100 - buy_pct}% · "
               f"GPA ≥ {gpa_floor}th pctl" + (" · not a SELL candidate" if excl_sell else ""))
    buy_cols = ["ticker", "name", "industry", "mcap_m", "GPA", "GPA_pct", "BM", "BM_pct", "buy_score"]
    st.dataframe(buy[buy_cols].style.format({k: v for k, v in fmt.items() if k in buy_cols}, na_rep="—").format_index(lambda c: names.get(c, c), axis=1),
                 hide_index=True, width="stretch")
    st.download_button("Download BUY list", buy[buy_cols].to_csv(index=False), "buy.csv")

with tab_chart:
    measures = {"GPA": "GPA", "B/M": "BM", "STA": "STA", "SNOA": "SNOA",
                "Beneish M-score": "M_score", "Altman Z": "Altman_Z"}
    a, b = st.columns(2)
    x_lab = a.selectbox("X axis", list(measures), index=2)
    y_lab = b.selectbox("Y axis", list(measures), index=0)
    plot = df.dropna(subset=[measures[x_lab], measures[y_lab]])
    # Clip extreme outliers to the 1st–99th percentile so the cloud is readable
    for col in {measures[x_lab], measures[y_lab]}:
        lo, hi = plot[col].quantile([0.01, 0.99])
        plot = plot[plot[col].between(lo, hi)]
    fig = px.scatter(plot, x=measures[x_lab], y=measures[y_lab], color="Signal",
                     color_discrete_map={"SELL": "#d62728", "BUY": "#2ca02c", "Neither": "#bbbbbb"},
                     hover_data=["ticker", "name", "industry"],
                     labels={measures[x_lab]: x_lab, measures[y_lab]: y_lab})
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{len(plot)} companies with both measures; extreme 1% tails hidden for readability.")

    by_ind = (df.groupby("industry")[["SELL", "BUY"]].sum()
                .query("SELL > 0 or BUY > 0").sort_values("SELL", ascending=False).reset_index())
    if len(by_ind):
        fig2 = px.bar(by_ind, x="industry", y=["SELL", "BUY"], barmode="group",
                      color_discrete_map={"SELL": "#d62728", "BUY": "#2ca02c"},
                      labels={"value": "Companies", "industry": "", "variable": ""})
        st.plotly_chart(fig2, width="stretch")

with tab_drill:
    pick = st.selectbox("Company", (df["ticker"] + " · " + df["name"]).sort_values())
    if pick:
        r = df[df["ticker"] == pick.split(" · ")[0]].iloc[0]
        st.markdown(f"### {r['name']} ({r['ticker']}) — **{r['Signal']}**")
        st.caption(f"{r['industry']} · market cap ${r['mcap_m']:,.0f}M · fiscal year end {r.get('fiscal_year_end', '—')}")

        def show(v, f):
            return "—" if pd.isna(v) else f.format(v)

        d1, d2 = st.columns(2)
        d1.markdown("**SELL measures**")
        d1.table(pd.DataFrame({
            "Value": [show(r["STA"], "{:.3f}"), show(r["SNOA"], "{:.2f}"),
                      show(r["M_score"], "{:.2f}"), show(r["PMAN"], "{:.1%}"), show(r["Altman_Z"], "{:.2f}")],
            "Percentile": [show(r["STA_pct"], "{:.0f}"), show(r["SNOA_pct"], "{:.0f}"), "", "", ""],
            "Red flag": ["🚩" if r["flag_STA"] else "", "🚩" if r["flag_SNOA"] else "",
                         "🚩" if r["flag_Beneish"] else "", "", "🚩" if r["flag_AltmanZ"] else ""],
        }, index=["STA", "SNOA", "Beneish M-score", "PMAN", "Altman Z"]))
        d2.markdown("**BUY measures**")
        d2.table(pd.DataFrame({
            "Value": [show(r["GPA"], "{:.3f}"), show(r["BM"], "{:.3f}"), show(r["buy_score"], "{:.1f}")],
            "Percentile": [show(r["GPA_pct"], "{:.0f}"), show(r["BM_pct"], "{:.0f}"),
                           show(r["buy_score_pct"], "{:.0f}")],
        }, index=["GPA", "B/M", "Combined score"]))
        with st.expander("Beneish M-score components"):
            comp = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]
            st.table(pd.DataFrame({"Value": [show(r[c], "{:.3f}") for c in comp]}, index=comp))

with tab_conc:
    st.info("Write these yourself after studying the filings. Go beyond the score: what in the "
            "business explains the numbers, and what would prove you wrong?")
    st.markdown("""
#### Top SELL candidates
**[TICKER] · [Company]** — *Which flags fired and why they matter here (e.g. accruals rising
faster than cash flow, inventory build-up, heavy lease and debt load). What could go wrong
for the stock, and what would change your mind?*

**[TICKER] · [Company]** — *...*

#### Top BUY candidates
**[TICKER] · [Company]** — *Why this company earns high gross profit on its assets and why
the market may be pricing it too cheaply. Main risk to the thesis.*

**[TICKER] · [Company]** — *...*
""")

with tab_method:
    st.markdown(f"""
**Universe.** Every US-listed company in Yahoo Finance's Consumer Cyclical (Consumer Discretionary)
sector on NYSE, NYSE American and Nasdaq, pulled with yfinance; latest two annual statements.
{len(df_all)} companies in the file, {len(df)} after the current filters. Duplicate share classes are
dropped. Excluded by design: {", ".join(EXCLUDE) if EXCLUDE else "none yet"}.

**Why these measures suit Consumer Discretionary.** Retailers, automakers, restaurants and
homebuilders carry large inventories, receivables and store leases, so accruals (STA, SNOA) and
Beneish's receivables, margin and asset-quality indexes are where earnings management shows up.
Spending is cyclical and many firms are highly levered, so Altman Z catches distress.
On the BUY side, gross profitability captures brand and pricing power, the main competitive
edge in this sector, and book-to-market keeps the screen from simply buying expensive winners.

**SELL measures (a firm is a SELL candidate with at least N red flags):**
- STA = (Net income − Operating cash flow) / Total assets. Higher = lower earnings quality.
- SNOA = (Operating assets − Operating liabilities) / Lagged total assets, where operating assets =
  total assets − cash & short-term investments and operating liabilities = total assets − total debt −
  minority interest − preferred stock − common equity. Total debt includes lease liabilities.
- Beneish M = −4.84 + 0.920 DSRI + 0.528 GMI + 0.404 AQI + 0.892 SGI + 0.115 DEPI − 0.172 SGAI
  + 4.679 TATA − 0.327 LVGI; PMAN = Φ(M).
- Altman Z = 1.2 WC/TA + 1.4 RE/TA + 3.3 EBIT/TA + 0.6 MVE/TL + 1.0 Sales/TA.

**BUY measures:** GPA = Gross profit / Total assets; B/M = Common equity / Market cap. Each is ranked
as a percentile within the screened universe; the BUY score is their average. The BUY side uses
different measures from the SELL side.

**Missing data.** Nothing is filled with zero, except minority interest and preferred stock, which
Yahoo leaves out when a company has none. A missing measure never raises a red flag, and a firm
without both GPA and B/M can't be a BUY. Negative book equity (common after large buybacks in
this sector) leaves B/M missing. B/M and Altman Z use market cap, so they're only calculated for
companies reporting in US dollars.
""")
    cov = df_all[["STA", "SNOA", "M_score", "Altman_Z", "GPA", "BM"]].notna().sum().rename("Companies with a value")
    st.table(cov.to_frame())
