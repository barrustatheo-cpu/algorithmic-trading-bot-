"""
Trading Bot Dashboard - Flask + Plotly
Author: Theo
"""

from flask import Flask, render_template_string
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import json
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ============================================================
# CONFIG
# ============================================================
TICKER        = "QQQ"
START         = "2022-01-01"
END           = "2024-12-31"
INTERVAL      = "1d"
INITIAL_CAP   = 10_000
FEES          = 0.001
SLIPPAGE      = 0.0005
TRAIN_RATIO   = 0.33
SCORE_MIN     = 4
SLOPE_WIN     = 3
ATR_STOP      = 1.2
ATR_TP        = 3.0
ATR_TRAIL     = 1.0
ATR_TRAIL_ACT = 0.8
PULLBACK_MAX  = 0.015


# ============================================================
# 1. DATA
# ============================================================
def load_data():
    df = yf.download(TICKER, start=START, end=END,
                     interval=INTERVAL, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


# ============================================================
# 2. INDICATEURS
# ============================================================
def build_indicators(df):
    for w in [10, 20, 50, 200]:
        df[f"MA{w}"] = df["Close"].rolling(w).mean()

    df["MA20_slope"] = df["MA20"] - df["MA20"].shift(SLOPE_WIN)
    df["MA50_slope"] = df["MA50"] - df["MA50"].shift(SLOPE_WIN)

    delta      = df["Close"].diff()
    gain       = delta.clip(lower=0).rolling(14).mean()
    loss       = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"]  = 100 - (100 / (1 + gain / loss))

    ema12           = df["Close"].ewm(span=12).mean()
    ema26           = df["Close"].ewm(span=26).mean()
    df["MACD"]      = ema12 - ema26
    df["MACD_Sig"]  = df["MACD"].ewm(span=9).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Sig"]

    bb_mid       = df["Close"].rolling(20).mean()
    bb_std       = df["Close"].rolling(20).std()
    df["BB_pos"] = (df["Close"] - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-9)

    hl        = df["High"] - df["Low"]
    hc        = (df["High"] - df["Close"].shift()).abs()
    lc        = (df["Low"]  - df["Close"].shift()).abs()
    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    df["Vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1)
    df["Mom_5"]     = df["Close"].pct_change(5)
    df["Dist_MA20"] = (df["Close"] - df["MA20"]) / df["MA20"]

    df.dropna(inplace=True)
    return df


# ============================================================
# 3. GATE
# ============================================================
def entry_gate(row):
    return (float(row["Close"])      > float(row["MA200"]) and
            float(row["MA50_slope"]) > 0 and
            float(row["MA20_slope"]) > 0 and
            float(row["Mom_5"])      > 0 and
            float(row["RSI"])        > 48)


# ============================================================
# 4. SCORE
# ============================================================
def entry_score(row):
    score = 0
    if float(row["Close"]) > float(row["MA200"]):                                    score += 1
    if float(row["MA10"])  > float(row["MA20"]) and \
       float(row["MA20"])  > float(row["MA50"]):                                     score += 1
    if 48 < float(row["RSI"]) < 68:                                                  score += 1
    if float(row["MACD_Hist"]) > 0 and float(row["MACD"]) > float(row["MACD_Sig"]): score += 1
    if 0.35 < float(row["BB_pos"]) < 0.80:                                           score += 1
    if float(row["Vol_ratio"]) > 1.0:                                                score += 1
    return score


# ============================================================
# 5. PULLBACK
# ============================================================
def is_pullback(row):
    dist = float(row["Dist_MA20"])
    return 0.0 <= dist <= PULLBACK_MAX


# ============================================================
# 6. BACKTEST
# ============================================================
def backtest(df):
    split   = int(len(df) * TRAIN_RATIO)
    df_test = df.iloc[split:].copy().reset_index(drop=False)

    balance     = float(INITIAL_CAP)
    shares      = 0.0
    entry_price = 0.0
    peak_price  = 0.0
    stop_price  = 0.0
    tp_price    = 0.0
    trail_act   = 0.0
    in_trade    = False
    signal_active = False
    patience    = 0
    MAX_PATIENCE = 5

    history   = []
    buy_pts   = []
    sell_pts  = []
    trades    = []

    for i in range(len(df_test)):
        row      = df_test.iloc[i]
        p        = float(row["Close"])
        atr      = float(row["ATR"])
        gain_pct = (p - entry_price) / entry_price if in_trade else 0.0

        # ── EXITS ─────────────────────────────────────────────
        if in_trade:
            peak_price  = max(peak_price, p)
            trailing_on = p >= trail_act
            floor       = (peak_price - ATR_TRAIL * atr
                           if trailing_on else stop_price)

            if p >= tp_price:
                proceeds = shares * p * (1 - SLIPPAGE) * (1 - FEES)
                pnl      = proceeds - shares * entry_price
                trades.append({
                    "pnl":        pnl,
                    "return_pct": gain_pct * 100,
                    "type":       "take_profit"
                })
                balance    = proceeds
                shares     = 0.0
                in_trade   = False
                signal_active = False
                sell_pts.append(i)

            elif p <= floor:
                proceeds  = shares * p * (1 - SLIPPAGE) * (1 - FEES)
                pnl       = proceeds - shares * entry_price
                exit_type = "trailing_stop" if trailing_on else "stop_loss"
                trades.append({
                    "pnl":        pnl,
                    "return_pct": gain_pct * 100,
                    "type":       exit_type
                })
                balance    = proceeds
                shares     = 0.0
                in_trade   = False
                signal_active = False
                sell_pts.append(i)

        # ── ENTRÉE ────────────────────────────────────────────
        if not in_trade and balance > 100:
            gate = entry_gate(row)
            sc   = entry_score(row)

            if not signal_active:
                if gate and sc >= SCORE_MIN:
                    signal_active = True
                    patience      = 0

            if signal_active:
                patience += 1
                if is_pullback(row):
                    exec_p      = p * (1 + SLIPPAGE)
                    shares      = (balance * (1 - FEES)) / exec_p
                    entry_price = exec_p
                    peak_price  = exec_p
                    stop_price  = exec_p - ATR_STOP     * atr
                    tp_price    = exec_p + ATR_TP        * atr
                    trail_act   = exec_p + ATR_TRAIL_ACT * atr
                    balance     = 0.0
                    in_trade    = True
                    signal_active = False
                    buy_pts.append(i)
                elif patience >= MAX_PATIENCE:
                    signal_active = False

        history.append(balance + shares * p if in_trade else balance)

    # Fermer position finale
    if in_trade and shares > 0:
        p        = float(df_test.iloc[-1]["Close"])
        gain_pct = (p - entry_price) / entry_price
        proceeds = shares * p * (1 - FEES)
        trades.append({
            "pnl":        proceeds - shares * entry_price,
            "return_pct": gain_pct * 100,
            "type":       "end_close"
        })
        history[-1] = proceeds

    return df_test, history, buy_pts, sell_pts, trades


# ============================================================
# 7. METRICS
# ============================================================
def compute_metrics(history, trades):
    h   = pd.Series(history, dtype=float)
    ret = h.pct_change().dropna()
    ann = 252

    final   = float(h.iloc[-1])
    gain    = (final - INITIAL_CAP) / INITIAL_CAP * 100
    sharpe  = float((ret.mean() / ret.std()) * np.sqrt(ann)) if ret.std() > 0 else 0
    neg     = ret[ret < 0]
    sortino = float((ret.mean() / neg.std()) * np.sqrt(ann)) if len(neg) > 0 else 0
    dd      = float((h / h.cummax() - 1).min() * 100)
    calmar  = gain / abs(dd) if dd != 0 else 0

    pnl_list = [t["pnl"]        for t in trades]
    ret_list = [t["return_pct"] for t in trades]
    n        = len(pnl_list)
    wins     = [x for x in pnl_list if x > 0]
    losses   = [x for x in pnl_list if x < 0]
    win_rate = len(wins) / n * 100 if n else 0
    pf       = abs(sum(wins) / sum(losses)) if losses else 0
    avg_win  = np.mean([r for r in ret_list if r > 0]) if wins   else 0
    avg_loss = np.mean([r for r in ret_list if r < 0]) if losses else 0
    rr       = abs(avg_win / avg_loss)                 if avg_loss != 0 else 0

    return {
        "final":    final,
        "gain":     gain,
        "sharpe":   sharpe,
        "sortino":  sortino,
        "calmar":   calmar,
        "dd":       dd,
        "n":        n,
        "win_rate": win_rate,
        "pf":       pf,
        "avg_win":  avg_win,
        "avg_loss": avg_loss,
        "rr":       rr,
    }


# ============================================================
# 8. CHARTS
# ============================================================
def build_charts(df_test, history, buy_pts, sell_pts):
    prices = df_test["Close"].values.tolist()
    idx    = list(range(len(df_test)))

    # Essaie d'utiliser les dates, sinon les indices
    try:
        dates = df_test["Date"].astype(str).tolist()
    except Exception:
        try:
            dates = df_test.index.astype(str).tolist()
        except Exception:
            dates = idx

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Price + Signals", "Equity Curve", "MACD", "RSI"),
        row_heights=[0.40, 0.25, 0.18, 0.17]
    )

    # ── Prix ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=prices,
        name="Price",
        line=dict(color="#e0e0e0", width=1.2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=df_test["MA20"].values.tolist(),
        name="MA20", line=dict(color="#A09FE8", width=1)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=df_test["MA50"].values.tolist(),
        name="MA50", line=dict(color="#EF9F27", width=1)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=df_test["MA200"].values.tolist(),
        name="MA200", line=dict(color="#E24B4A", width=1, dash="dot")
    ), row=1, col=1)

    if buy_pts:
        fig.add_trace(go.Scatter(
            x=[dates[i] for i in buy_pts],
            y=[prices[i] * 0.997 for i in buy_pts],
            mode="markers", name="BUY",
            marker=dict(color="#1D9E75", size=12, symbol="triangle-up")
        ), row=1, col=1)

    if sell_pts:
        fig.add_trace(go.Scatter(
            x=[dates[i] for i in sell_pts],
            y=[prices[i] * 1.003 for i in sell_pts],
            mode="markers", name="SELL",
            marker=dict(color="#E24B4A", size=12, symbol="triangle-down")
        ), row=1, col=1)

    # ── Equity ─────────────────────────────────────────────────
    h = [float(v) for v in history]
    fig.add_trace(go.Scatter(
        x=dates, y=h,
        name="Equity",
        line=dict(color="#3B8BD4", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(59,139,212,0.07)"
    ), row=2, col=1)

    fig.add_hline(
        y=INITIAL_CAP,
        line_dash="dash", line_color="#555",
        row=2, col=1
    )

    # ── MACD ───────────────────────────────────────────────────
    macd      = df_test["MACD"].values.tolist()
    macd_sig  = df_test["MACD_Sig"].values.tolist()
    macd_hist = df_test["MACD_Hist"].values.tolist()

    fig.add_trace(go.Scatter(
        x=dates, y=macd,
        name="MACD", line=dict(color="#3B8BD4", width=1)
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=macd_sig,
        name="Signal", line=dict(color="#EF9F27", width=1)
    ), row=3, col=1)

    fig.add_trace(go.Bar(
        x=dates, y=macd_hist,
        name="Histogram",
        marker_color=["#1D9E75" if v >= 0 else "#E24B4A"
                      for v in macd_hist],
        opacity=0.7
    ), row=3, col=1)

    # ── RSI ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=df_test["RSI"].values.tolist(),
        name="RSI", line=dict(color="#A09FE8", width=1)
    ), row=4, col=1)

    fig.add_hline(y=70, line_dash="dash",
                  line_color="#E24B4A", opacity=0.5, row=4, col=1)
    fig.add_hline(y=48, line_dash="dash",
                  line_color="#1D9E75", opacity=0.5, row=4, col=1)
    fig.add_hline(y=30, line_dash="dash",
                  line_color="#1D9E75", opacity=0.5, row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#0d0d0d",
        font=dict(family="monospace", color="#ccc", size=11),
        height=900,
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=60, r=20, t=60, b=40),
        hovermode="x unified",
    )
    fig.update_yaxes(gridcolor="#1a1a1a", zerolinecolor="#333")
    fig.update_xaxes(gridcolor="#1a1a1a", rangeslider_visible=False)

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


# ============================================================
# 9. HTML
# ============================================================
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Bot Dashboard — QQQ</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#0d0d0d; color:#ccc; font-family:monospace; }

  .header {
    background:#111;
    border-bottom:1px solid #1e1e1e;
    padding:20px 32px;
    display:flex;
    align-items:center;
    justify-content:space-between;
  }
  .header h1 { color:#fff; font-size:16px; letter-spacing:3px; }
  .tag {
    background:#1D9E75;
    color:#fff;
    padding:4px 14px;
    border-radius:4px;
    font-size:11px;
    letter-spacing:1px;
  }

  .metrics {
    display:grid;
    grid-template-columns:repeat(auto-fit, minmax(150px, 1fr));
    gap:10px;
    padding:24px 32px;
  }
  .card {
    background:#111;
    border:1px solid #1e1e1e;
    border-radius:8px;
    padding:16px 12px;
    text-align:center;
  }
  .card .lbl {
    font-size:9px;
    color:#555;
    letter-spacing:1.5px;
    text-transform:uppercase;
    margin-bottom:8px;
  }
  .card .val       { font-size:22px; font-weight:bold; color:#fff; }
  .card .val.green { color:#1D9E75; }
  .card .val.red   { color:#E24B4A; }
  .card .val.blue  { color:#3B8BD4; }

  .chart-wrap {
    padding:0 32px 32px;
  }
  .chart-label {
    font-size:10px;
    color:#444;
    letter-spacing:2px;
    text-transform:uppercase;
    padding:8px 0 12px;
  }
  #chart {
    border:1px solid #1a1a1a;
    border-radius:8px;
    overflow:hidden;
  }

  .footer {
    text-align:center;
    padding:18px;
    font-size:10px;
    color:#333;
    border-top:1px solid #1a1a1a;
  }
</style>
</head>
<body>

<div class="header">
  <h1>⬡ ALGORITHMIC TRADING BOT — {{ ticker }}</h1>
  <div class="tag">BACKTEST {{ start }} → {{ end }}</div>
</div>

<div class="metrics">

  <div class="card">
    <div class="lbl">Total Return</div>
    <div class="val {{ 'green' if gain > 0 else 'red' }}">
      {{ ('+' if gain > 0 else '') + '{:.2f}'.format(gain) + '%' }}
    </div>
  </div>

  <div class="card">
    <div class="lbl">Final Balance</div>
    <div class="val blue">${{ '{:,.2f}'.format(final) }}</div>
  </div>

  <div class="card">
    <div class="lbl">Sharpe Ratio</div>
    <div class="val {{ 'green' if sharpe > 1 else 'blue' }}">
      {{ '{:.3f}'.format(sharpe) }}
    </div>
  </div>

  <div class="card">
    <div class="lbl">Sortino Ratio</div>
    <div class="val blue">{{ '{:.3f}'.format(sortino) }}</div>
  </div>

  <div class="card">
    <div class="lbl">Calmar Ratio</div>
    <div class="val blue">{{ '{:.3f}'.format(calmar) }}</div>
  </div>

  <div class="card">
    <div class="lbl">Max Drawdown</div>
    <div class="val red">{{ '{:.2f}'.format(dd) }}%</div>
  </div>

  <div class="card">
    <div class="lbl">Win Rate</div>
    <div class="val {{ 'green' if win_rate >= 50 else 'red' }}">
      {{ '{:.1f}'.format(win_rate) }}%
    </div>
  </div>

  <div class="card">
    <div class="lbl">Profit Factor</div>
    <div class="val {{ 'green' if pf >= 1.5 else 'blue' }}">
      {{ '{:.2f}'.format(pf) }}
    </div>
  </div>

  <div class="card">
    <div class="lbl">RR Ratio</div>
    <div class="val blue">{{ '{:.2f}'.format(rr) }}x</div>
  </div>

  <div class="card">
    <div class="lbl">Total Trades</div>
    <div class="val">{{ n }}</div>
  </div>

  <div class="card">
    <div class="lbl">Avg Win</div>
    <div class="val green">+{{ '{:.2f}'.format(avg_win) }}%</div>
  </div>

  <div class="card">
    <div class="lbl">Avg Loss</div>
    <div class="val red">{{ '{:.2f}'.format(avg_loss) }}%</div>
  </div>

</div>

<div class="chart-wrap">
  <div class="chart-label">Backtest Analysis — QQQ Daily</div>
  <div id="chart"></div>
</div>

<div class="footer">
  Algorithmic Trading Bot v18 — Pullback Entry + ATR Dynamic Stops
  &nbsp;|&nbsp; Built by Theo Barrusta
</div>

<script>
  var fig = {{ chart_json | safe }};
  Plotly.newPlot('chart', fig.data, fig.layout, {responsive: true});
</script>
</body>
</html>
"""


# ============================================================
# 10. FLASK ROUTE
# ============================================================
@app.route("/")
def index():
    print("[INFO] Loading data...")
    df = load_data()
    df = build_indicators(df)
    df_test, history, buys, sells, trades = backtest(df)
    m  = compute_metrics(history, trades)
    chart = build_charts(df_test, history, buys, sells)
    print("[INFO] Dashboard ready!")

    return render_template_string(
        HTML,
        ticker   = TICKER,
        start    = START,
        end      = END,
        final    = m["final"],
        gain     = m["gain"],
        sharpe   = m["sharpe"],
        sortino  = m["sortino"],
        calmar   = m["calmar"],
        dd       = m["dd"],
        n        = m["n"],
        win_rate = m["win_rate"],
        pf       = m["pf"],
        avg_win  = m["avg_win"],
        avg_loss = m["avg_loss"],
        rr       = m["rr"],
        chart_json = chart
    )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "━"*48)
    print("  TRADING BOT DASHBOARD")
    print("  Ouvre : http://localhost:5000")
    print("━"*48 + "\n")
    app.run(debug=False, port=5000)