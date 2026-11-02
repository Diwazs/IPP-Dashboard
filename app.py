"""
Balance Sheet to Beta - dashboard
Run:  streamlit run app.py
Same logic as the original script; the input() prompts become sidebar controls.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Balance Sheet to Beta", layout="wide")

# ---------------- settings (same as the script) ----------------
START, END = "2021-04-01", "2026-03-31"
CRORE, DAYS, TRADING_DAYS = 10_000_000, 365, 252
RISK_FREE = 0.0573            # 91-day T-bill
SAMPLES, MAX_WEIGHT, MIN_WEIGHT, SAME_UTILITY = 10_000, 0.30, 0.01, 0.001
TOP_N, MIN_HOLDINGS = 10, 5

PORTFOLIOS = [("Steady", 0.10, 0.15), ("Balanced", 0.14, 0.22), ("Adventurous", 0.18, 0.30)]
AMOUNT = 100_000

COMPANIES = {
    "INFY": ("Infosys", "IT"), "TCS": ("Tata Consultancy Services", "IT"),
    "HCLTECH": ("HCL Technologies", "IT"), "TECHM": ("Tech Mahindra", "IT"),
    "WIPRO": ("Wipro", "IT"), "PERSISTENT": ("Persistent Systems", "IT"),
    "COFORGE": ("Coforge", "IT"), "LTM": ("LTIMindtree", "IT"),
    "MPHASIS": ("Mphasis", "IT"), "OFSS": ("Oracle Financial Services", "IT"),
    "ITC": ("ITC", "FMCG"), "HINDUNILVR": ("Hindustan Unilever", "FMCG"),
    "NESTLEIND": ("Nestle India", "FMCG"), "BRITANNIA": ("Britannia Industries", "FMCG"),
    "TATACONSUM": ("Tata Consumer Products", "FMCG"), "DABUR": ("Dabur India", "FMCG"),
    "GODREJCP": ("Godrej Consumer Products", "FMCG"), "MARICO": ("Marico", "FMCG"),
    "COLPAL": ("Colgate-Palmolive India", "FMCG"), "UNITDSPR": ("United Spirits", "FMCG"),
}
INCOME_ITEMS = ["Total Revenue", "Net Income", "EBIT", "Interest Expense"]
BALANCE_ITEMS = ["Total Assets", "Stockholders Equity", "Total Debt", "Current Assets",
                 "Current Liabilities", "Inventory", "Accounts Receivable"]

RATIOS = {
    "net_margin": lambda f: f["Net Income"] / f["Total Revenue"],
    "roe": lambda f: f["Net Income"] / f["Stockholders Equity"],
    "roce": lambda f: f["EBIT"] / (f["Total Assets"] - f["Current Liabilities"]),
    "current_ratio": lambda f: f["Current Assets"] / f["Current Liabilities"],
    "quick_ratio": lambda f: (f["Current Assets"] - f["Inventory"].fillna(0)) / f["Current Liabilities"],
    "debt_to_equity": lambda f: f["Total Debt"] / f["Stockholders Equity"],
    "interest_coverage": lambda f: f["EBIT"] / f["Interest Expense"],
    "asset_turnover": lambda f: f["Total Revenue"] / f["Total Assets"],
    "inventory_days": lambda f: f["Inventory"] / f["Total Revenue"] * DAYS,
    "receivable_days": lambda f: f["Accounts Receivable"] / f["Total Revenue"] * DAYS,
}
LOWER_IS_BETTER = ["debt_to_equity", "inventory_days", "receivable_days"]
RATIO_NAMES = list(RATIOS)
ALL_RATIOS = RATIO_NAMES + ["revenue_cagr", "profit_cagr"]

# ---------------- data ----------------
def get_value(table, item, year):
    if item not in table.index or year not in table.columns:
        return None
    v = table.loc[item, year]
    return None if pd.isna(v) else v / CRORE


@st.cache_data(show_spinner=False)
def load_prices(tickers):
    symbols = [t + ".NS" for t in tickers] + ["^NSEI"]
    d = yf.download(symbols, start=START, end=END, progress=False, auto_adjust=True)["Close"]
    d = d.rename(columns={**{t + ".NS": t for t in tickers}, "^NSEI": "NIFTY50"})
    return d.dropna(axis=1, how="all")


@st.cache_data(show_spinner=False)
def load_fundamentals(items):
    rows = []
    for ticker, sector in items:
        try:
            s = yf.Ticker(ticker + ".NS")
            inc, bal = s.income_stmt, s.balance_sheet
        except Exception:
            continue
        if inc.empty:
            continue
        for y in inc.columns:
            r = {"ticker": ticker, "sector": sector, "year": y.year}
            for i in INCOME_ITEMS:
                r[i] = get_value(inc, i, y)
            for i in BALANCE_ITEMS:
                r[i] = get_value(bal, i, y)
            rows.append(r)
    return pd.DataFrame(rows).sort_values(["sector", "ticker", "year"])

# ---------------- analysis ----------------
def growth_rate(company, column):
    have = company[company[column].notna()]
    if len(have) < 2:
        return None
    first, last = have.iloc[0], have.iloc[-1]
    years = last["year"] - first["year"]
    if first[column] <= 0 or last[column] <= 0 or years <= 0:
        return None
    return (last[column] / first[column]) ** (1 / years) - 1


def build_ratios(fund):
    ratios = fund[["ticker", "sector", "year"]].copy()
    for name, fn in RATIOS.items():
        ratios[name] = fn(fund)
    ratios[RATIO_NAMES] = ratios[RATIO_NAMES].replace([np.inf, -np.inf], np.nan).astype(float)
    cr = ratios.groupby(["ticker", "sector"])[RATIO_NAMES].mean().reset_index()
    g = []
    for t in fund["ticker"].unique():
        c = fund[fund["ticker"] == t].sort_values("year")
        g.append({"ticker": t, "revenue_cagr": growth_rate(c, "Total Revenue"),
                  "profit_cagr": growth_rate(c, "Net Income")})
    cr = cr.merge(pd.DataFrame(g), on="ticker")
    cr[["revenue_cagr", "profit_cagr"]] = cr[["revenue_cagr", "profit_cagr"]].astype(float)
    return ratios, cr.sort_values(["sector", "ticker"])


def normalise(values, lower_is_better):
    lo, hi = values.min(), values.max()
    if hi == lo:
        return pd.Series(0.5, index=values.index)
    s = (values - lo) / (hi - lo)
    return 1 - s if lower_is_better else s


def red_flags(c):
    f = []
    if pd.notna(c["interest_coverage"]) and c["interest_coverage"] < 2: f.append("profit barely covers interest")
    if pd.notna(c["debt_to_equity"]) and c["debt_to_equity"] > 2: f.append("debt over twice equity")
    if pd.notna(c["roe"]) and c["roe"] < 0: f.append("losing shareholders money")
    if pd.notna(c["current_ratio"]) and c["current_ratio"] < 1: f.append("short-term dues exceed short-term assets")
    return f


def score_companies(cr):
    parts = []
    for sector in cr["sector"].unique():
        g = cr[cr["sector"] == sector].copy()
        usable = [n for n in ALL_RATIOS if g[n].notna().all()]
        g["score"] = sum(normalise(g[n], n in LOWER_IS_BETTER) for n in usable) / len(usable)
        g["ratios_used"] = len(usable)
        parts.append(g)
    s = pd.concat(parts).sort_values("score", ascending=False)
    s["rank"] = range(1, len(s) + 1)
    s["flags"] = [", ".join(red_flags(s.iloc[i])) for i in range(len(s))]
    return s


def run_portfolio(prices, selected, scores, A):
    rets = prices[selected + ["NIFTY50"]].pct_change(fill_method=None).dropna()
    yr = rets[selected].mean() * TRADING_DAYS
    cov = rets[selected].cov() * TRADING_DAYS
    n = len(selected)
    rng = np.random.default_rng(42)
    eq = np.ones(n) / n
    sw = scores.values / scores.values.sum()
    W = np.vstack([rng.dirichlet(np.ones(n) * s, size=SAMPLES) for s in [0.5, 1, 3, 8]] + [np.array([eq, sw])])
    W = W[W.max(axis=1) <= MAX_WEIGHT + 1e-9]
    R = W @ yr.values
    V = np.sum((W @ cov.values) * W, axis=1)
    conc = np.sum(W ** 2, axis=1)

    def pick(a):
        u = R - 0.5 * a * V
        tied = np.where(u >= u.max() - SAME_UTILITY)[0]
        return tied[conc[tied].argmin()]

    def tidy(w):
        w = w.copy(); w[w < MIN_WEIGHT] = 0
        return w / w.sum()

    def beta(w):
        p = np.cov((rets[selected] * w).sum(axis=1), rets["NIFTY50"])
        return p[0][1] / p[1][1]

    def worst(w):
        g = (1 + (rets[selected] * w).sum(axis=1)).cumprod()
        return ((g - g.cummax()) / g.cummax()).min()

    return dict(rets=rets, yr=yr, cov=cov, W=W, R=R, V=V, sd=np.sqrt(V), pick=pick, tidy=tidy,
                beta=beta, worst=worst, eq=eq, sw=sw, lowest=V.argmin(), n=n)

# ---------------- sidebar ----------------
st.sidebar.title("Balance Sheet to Beta")
st.sidebar.caption("NSE IT + FMCG screening and portfolio analytics")

st.sidebar.subheader("1. Risk aversion (A)")
mode = st.sidebar.radio("How to set A", ["Questionnaire", "Set A directly"])
if mode == "Questionnaire":
    st.sidebar.caption("What fixed deposit payout (Rs) would make you indifferent to each risky option?")
    implied = []
    for name, er, sd in PORTFOLIOS:
        worst, best = AMOUNT * (1 + er - sd), AMOUNT * (1 + er + sd)
        default = float(round(AMOUNT * (1 + er - 0.5 * 3 * sd ** 2), -2))
        ce = st.sidebar.number_input(f"{name}: avg Rs {AMOUNT*(1+er):,.0f}, range {worst:,.0f}-{best:,.0f}",
                                     min_value=float(worst) + 1, max_value=float(best) - 1,
                                     value=min(max(default, worst + 1), best - 1), step=500.0)
        implied.append(2 * (er - (ce - AMOUNT) / AMOUNT) / sd ** 2)
    A = float(np.mean(implied))
else:
    A = st.sidebar.slider("A", -2.0, 10.0, 3.0, 0.1)
profile = "risk averse" if A > 0 else "risk neutral" if A == 0 else "risk seeking"
st.sidebar.metric("Your A", f"{A:.2f}", profile)

st.sidebar.subheader("2. Investment")
investment = st.sidebar.number_input("Amount (Rs)", min_value=1000.0, value=100000.0, step=10000.0)

st.sidebar.subheader("3. Extra companies")
extra = st.sidebar.text_input("TICKER:SECTOR, comma separated", placeholder="e.g. LTTS:IT, EMAMILTD:FMCG")
companies = dict(COMPANIES)
for chunk in [c.strip() for c in extra.split(",") if c.strip()]:
    if ":" in chunk:
        t, s = [x.strip().upper() for x in chunk.split(":", 1)]
        if s in ("IT", "FMCG") and t not in companies:
            companies[t] = (t, s)

# ---------------- load + compute ----------------
st.title("Balance Sheet to Beta")
st.caption("Does screening on fundamentals quietly concentrate your portfolio into one sector?")

with st.spinner("Downloading prices and statements from Yahoo Finance..."):
    tickers = tuple(companies)
    prices = load_prices(tickers)
    fund = load_fundamentals(tuple((t, companies[t][1]) for t in tickers))

gone = [t for t in tickers if t not in prices.columns or t not in set(fund["ticker"])]
live = {t: v for t, v in companies.items() if t not in gone}
if gone:
    st.warning(f"Dropped (no data from Yahoo): {', '.join(gone)}")
fund = fund[fund["ticker"].isin(live)]
ratios, cr = build_ratios(fund)
scored = score_companies(cr)

clean = scored[scored["flags"] == ""]
healthy = clean if len(clean) >= MIN_HOLDINGS else scored
selected = list(healthy.head(TOP_N)["ticker"])
if len(selected) * MAX_WEIGHT < 1:
    st.error("Too few companies to respect the 30% cap."); st.stop()
mix = healthy.head(len(selected))["sector"].value_counts()

P = run_portfolio(prices, selected, healthy.set_index("ticker").loc[selected, "score"], A)
best = P["pick"](A)
yours = P["tidy"](P["W"][best])
held = [selected[i] for i in range(P["n"]) if yours[i] > 0]
corr = P["rets"][held].corr()
avg_corr = corr.values[np.triu_indices(len(held), 1)].mean() if len(held) > 1 else 0.0

def stats(w):
    e = float(w @ P["yr"].values); v = float(np.sum((w @ P["cov"].values) * w)); s = v ** 0.5
    return e, v, s

rows = []
for name, w in {"Yours": yours, "Equal weight": P["eq"], "Score weighted": P["sw"],
                "Lowest risk": P["tidy"](P["W"][P["lowest"]])}.items():
    e, v, s = stats(w)
    rows.append({"portfolio": name, "return_pct": round(e * 100, 1), "risk_pct": round(s * 100, 1),
                 "sharpe": round((e - RISK_FREE) / s, 2), "beta": round(P["beta"](w), 2),
                 "worst_fall_pct": round(P["worst"](w) * 100, 1), "utility": round(e - 0.5 * A * v, 4)})
nf = P["rets"]["NIFTY50"]; ne, ns = nf.mean() * TRADING_DAYS, nf.std() * TRADING_DAYS ** 0.5
ng = (1 + nf).cumprod()
rows.append({"portfolio": "Nifty 50", "return_pct": round(ne * 100, 1), "risk_pct": round(ns * 100, 1),
             "sharpe": round((ne - RISK_FREE) / ns, 2), "beta": 1.0,
             "worst_fall_pct": round(((ng - ng.cummax()) / ng.cummax()).min() * 100, 1),
             "utility": round(ne - 0.5 * A * ns ** 2, 4)})
comp = pd.DataFrame(rows).set_index("portfolio")

# ---------------- headline ----------------
you = comp.loc["Yours"]
k = st.columns(5)
k[0].metric("Companies analysed", len(live))
k[1].metric("Expected return", f"{you.return_pct}%")
k[2].metric("Risk (SD)", f"{you.risk_pct}%")
k[3].metric("Beta vs Nifty", you.beta)
k[4].metric("Avg correlation", f"{avg_corr:.2f}")

tabs = st.tabs(["How it works", "Ratios", "Ranking", "Your portfolio", "Frontier", "Diversification", "If A changed"])

with tabs[0]:
    st.markdown("""
**Pipeline:** risk questionnaire gives **A** -> download prices and statements -> 10 ratios + 2 growth rates
-> score each company **within its sector** -> drop red-flag companies -> top 10 -> test ~40,000 splits of money
(max 30% per company) -> keep the one with the highest utility **U = E(r) - 0.5 x A x variance**.
""")
    c1, c2 = st.columns(2)
    c1.markdown("**Ratios used**\n- Profitability: net margin, ROE, ROCE\n- Liquidity: current, quick\n"
                "- Leverage: debt/equity, interest coverage\n- Efficiency: asset turnover, inventory days, receivable days\n"
                "- Growth: revenue CAGR, profit CAGR")
    c2.markdown("**Red flags (absolute checks)**\n- Interest coverage < 2\n- Debt/equity > 2\n- ROE < 0\n- Current ratio < 1\n\n"
                "Relative scores only say who is best of a weak sector; flags catch that.")

with tabs[1]:
    sel = st.multiselect("Companies", list(live), default=selected[:5])
    fig = px.line(ratios[ratios["ticker"].isin(sel)], x="year", y="roe", color="ticker", markers=True,
                  title="Return on equity over the years")
    fig.update_yaxes(tickformat=".0%"); st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cr.set_index("ticker").round(3), use_container_width=True)

with tabs[2]:
    fig = px.bar(scored.sort_values("score"), x="score", y="ticker", color="sector", orientation="h",
                 title="Fundamental score, within sector (0 = weakest, 1 = strongest)", height=600)
    st.plotly_chart(fig, use_container_width=True)
    fl = scored[scored["flags"] != ""]
    if fl.empty:
        st.success("No red flags: every company clears the basic health checks.")
    else:
        st.warning("Red flags"); st.dataframe(fl[["rank", "ticker", "sector", "flags"]].set_index("rank"))
    st.write("**Sector mix of the top picks**")
    st.bar_chart(mix)
    if mix.max() >= len(selected) * 0.7:
        st.info(f"{mix.max()} of {len(selected)} picks come from one sector: quality screening concentrated the portfolio.")

with tabs[3]:
    alloc = pd.DataFrame({"ticker": selected, "weight_pct": (yours * 100).round(1),
                          "amount_rs": (yours * investment).round()})
    alloc["company"] = alloc["ticker"].map(lambda t: live[t][0])
    alloc = alloc[alloc["weight_pct"] > 0].sort_values("weight_pct", ascending=False)
    c1, c2 = st.columns([1, 1])
    c1.dataframe(alloc.set_index("ticker"), use_container_width=True)
    c2.plotly_chart(px.pie(alloc, names="ticker", values="weight_pct", hole=0.45), use_container_width=True)
    st.subheader("How it compares")
    st.dataframe(comp, use_container_width=True)
    st.caption("Utility is scored at YOUR A, so yours should come out highest among the portfolios tested.")

with tabs[4]:
    order = np.argsort(P["sd"]); fr, fret, bs = [], [], -999.0
    for i in order:
        if P["R"][i] > bs:
            bs = P["R"][i]; fr.append(P["sd"][i] * 100); fret.append(P["R"][i] * 100)
    sharpe = (P["R"] - RISK_FREE) / P["sd"]; tg = sharpe.argmax()
    e, v, s = stats(yours)
    fig = go.Figure()
    fig.add_scatter(x=P["sd"] * 100, y=P["R"] * 100, mode="markers", name="Portfolios tested",
                    marker=dict(size=3, color="#d0d0d0"))
    fig.add_scatter(x=np.sqrt(np.diag(P["cov"].values)) * 100, y=P["yr"].values * 100, mode="markers+text",
                    text=selected, textposition="top center", name="Single companies",
                    marker=dict(symbol="square", size=9, color="grey"))
    fig.add_scatter(x=fr, y=fret, mode="lines", name="Efficient frontier", line=dict(width=3, color="#0072B2"))
    x2 = np.array([0, P["sd"][tg] * 1.25])
    fig.add_scatter(x=x2 * 100, y=(RISK_FREE + sharpe[tg] * x2) * 100, mode="lines", name="Capital market line",
                    line=dict(dash="dash", color="#009E73"))
    fig.add_scatter(x=[s * 100], y=[e * 100], mode="markers", name="Yours", marker=dict(size=16, color="#D55E00"))
    fig.add_scatter(x=[P["sd"][tg] * 100], y=[P["R"][tg] * 100], mode="markers", name="Best Sharpe",
                    marker=dict(size=14, color="#009E73"))
    fig.update_layout(title="Markowitz efficient frontier", xaxis_title="Risk, yearly SD (%)",
                      yaxis_title="Expected return a year (%)", height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Every single company sits below the frontier: the case for holding a portfolio, not one share.")

with tabs[5]:
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="Blues", zmin=0, zmax=1,
                              title=f"Do the holdings move together? Average = {avg_corr:.2f}"),
                    use_container_width=True)
    st.write("Above 0.6: far less diversification than the holding count suggests. 0.4-0.6: moderate. Below 0.4: genuinely spread.")

with tabs[6]:
    show = [-2, 0, 2, 4, 6, 10, 20]; tbl = {}
    for a in show:
        w = P["tidy"](P["W"][P["pick"](a)])
        tbl[f"A={a}"] = list((w * 100).round()) + [round(float(w @ P["yr"].values) * 100, 1),
                                                   round(stats(w)[2] * 100, 1)]
    st.dataframe(pd.DataFrame(tbl, index=selected + ["RETURN %", "RISK %"]), use_container_width=True)
    st.caption("A of 1 to 4 covers most real investors; higher values only make the trend visible.")

st.divider()
st.caption("Data: Yahoo Finance via yfinance (5 financial years to Mar 2026). Expected returns are historical averages, "
           "not forecasts. Educational project, not investment advice.")
