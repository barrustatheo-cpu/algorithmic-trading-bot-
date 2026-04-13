"""
Multi-Asset Backtest — Algorithmic Trading Bot v18
Author: Theo
Tests the strategy across multiple tickers to prove robustness.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

plt.style.use("dark_background")
plt.rcParams.update({
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
})

# ============================================================
# CONFIGURATION
# ============================================================
TICKERS = ["QQQ", "NVDA", "MSFT", "AAPL", "SPY"]
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
def load_data(ticker):
    df = yf.download(ticker, start=START, end=END,
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
    if float(row["Close"]) > float(row["MA200"]):                                     score += 1
    if float(row["MA10"])  > float(row["MA20"]) and \
       float(row["MA20"])  > float(row["MA50"]):                                      score += 1
    if 48 < float(row["RSI"]) < 68:                                                   score += 1
    if float(row["MACD_Hist"]) > 0 and float(row["MACD"]) > float(row["MACD_Sig"]): score += 1
    if 0.35 < float(row["BB_pos"]) < 0.80:                                            score += 1
    if float(row["Vol_ratio"]) > 1.0:                                                 score += 1
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

    history  = []
    buy_pts  = []
    sell_pts = []
    trades   = []

    for i in range(len(df_test)):
        row      = df_test.iloc[i]
        p        = float(row["Close"])
        atr      = float(row["ATR"])
        gain_pct = (p - entry_price) / entry_price if in_trade else 0.0

        if in_trade:
            peak_price  = max(peak_price, p)
            trailing_on = p >= trail_act
            floor       = (peak_price - ATR_TRAIL * atr
                           if trailing_on else stop_price)

            if p >= tp_price:
                proceeds = shares * p * (1 - SLIPPAGE) * (1 - FEES)
                trades.append({"pnl": proceeds - shares*entry_price,
                                "return_pct": gain_pct*100,
                                "type": "take_profit"})
                balance, shares, in_trade = proceeds, 0.0, False
                signal_active = False
                sell_pts.append(i)

            elif p <= floor:
                proceeds  = shares * p * (1 - SLIPPAGE) * (1 - FEES)
                exit_type = "trailing_stop" if trailing_on else "stop_loss"
                trades.append({"pnl": proceeds - shares*entry_price,
                                "return_pct": gain_pct*100,
                                "type": exit_type})
                balance, shares, in_trade = proceeds, 0.0, False
                signal_active = False
                sell_pts.append(i)

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

    if in_trade and shares > 0:
        p        = float(df_test.iloc[-1]["Close"])
        gain_pct = (p - entry_price) / entry_price
        proceeds = shares * p * (1 - FEES)
        trades.append({"pnl": proceeds - shares*entry_price,
                        "return_pct": gain_pct*100,
                        "type": "end_close"})
        history[-1] = proceeds

    return df_test, history, buy_pts, sell_pts, trades


# ============================================================
# 7. METRICS
# ============================================================
def compute_metrics(history, trades, ticker):
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

    stops = sum(1 for t in trades if t["type"] == "stop_loss")

    return {
        "Ticker":       ticker,
        "Return":       f"{gain:+.2f}%",
        "Final ($)":    f"${final:,.0f}",
        "Sharpe":       f"{sharpe:.2f}",
        "Sortino":      f"{sortino:.2f}",
        "Drawdown":     f"{dd:.2f}%",
        "Win Rate":     f"{win_rate:.1f}%",
        "RR":           f"{rr:.2f}x",
        "Trades":       n,
        "Stop losses":  stops,
        # Raw values for sorting/coloring
        "_gain":        gain,
        "_sharpe":      sharpe,
        "_history":     history,
    }


# ============================================================
# 8. VISUALISATION MULTI-ACTIFS
# ============================================================
def plot_multi(results):
    n     = len(results)
    fig   = plt.figure(figsize=(18, 14), facecolor="#0d0d0d")
    gs    = gridspec.GridSpec(2, 1, hspace=0.5,
                              height_ratios=[1, 1.5])

    # ── Tableau comparatif ────────────────────────────────────
    ax_table = fig.add_subplot(gs[0])
    ax_table.axis("off")

    cols = ["Ticker", "Return", "Final ($)", "Sharpe",
            "Sortino", "Drawdown", "Win Rate", "RR", "Trades", "Stop losses"]
    rows = [[r[c] for c in cols] for r in results]

    table = ax_table.table(
        cellText=rows,
        colLabels=cols,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # Couleurs header
    for j in range(len(cols)):
        table[(0, j)].set_facecolor("#1a1a1a")
        table[(0, j)].set_text_props(color="#aaa", fontweight="bold")

    # Couleurs lignes selon performance
    for i, r in enumerate(results):
        color = "#0d2b1a" if r["_gain"] > 0 else "#2b0d0d"
        for j in range(len(cols)):
            table[(i+1, j)].set_facecolor(color)
            text_color = "#1D9E75" if r["_gain"] > 0 else "#E24B4A"
            if j == 1:  # colonne Return
                table[(i+1, j)].set_text_props(color=text_color, fontweight="bold")
            else:
                table[(i+1, j)].set_text_props(color="#ccc")

    ax_table.set_title(
        f"Multi-Asset Backtest  |  {START} → {END}  |  "
        f"Strategy: Pullback Entry + ATR Stops",
        color="white", fontsize=11, pad=20
    )

    # ── Equity curves ─────────────────────────────────────────
    ax_eq = fig.add_subplot(gs[1])
    colors = ["#1D9E75", "#3B8BD4", "#EF9F27", "#A09FE8", "#E24B4A"]

    for i, r in enumerate(results):
        h = pd.Series(r["_history"], dtype=float)
        ax_eq.plot(h.values, color=colors[i], lw=1.5,
                   label=f"{r['Ticker']} ({r['Return']})")

    ax_eq.axhline(INITIAL_CAP, color="#444", lw=0.8,
                  linestyle="--", label="Initial capital")
    ax_eq.fill_between(range(len(results[0]["_history"])),
                       INITIAL_CAP,
                       pd.Series(results[0]["_history"]).values,
                       alpha=0.05, color="#1D9E75")

    ax_eq.set_title("Equity Curves — All Tickers", color="white")
    ax_eq.legend(loc="upper left", fontsize=9, framealpha=0.2)
    ax_eq.grid(alpha=0.1)
    ax_eq.set_ylabel("Portfolio Value ($)")

    plt.suptitle(
        "Algorithmic Trading Bot v18 — Multi-Asset Robustness Test",
        color="white", fontsize=13, y=0.98
    )

    plt.savefig("multi_asset_results.png", dpi=150,
                bbox_inches="tight", facecolor="#0d0d0d")
    plt.show()
    print("[PLOT]  Saved → multi_asset_results.png")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "━"*52)
    print("  MULTI-ASSET BACKTEST — v18 Strategy")
    print(f"  Tickers : {', '.join(TICKERS)}")
    print(f"  Period  : {START} → {END}")
    print("━"*52 + "\n")

    results = []

    for ticker in TICKERS:
        print(f"[{ticker}]  Loading & running backtest...")
        try:
            df = load_data(ticker)
            df = build_indicators(df)
            df_test, history, buys, sells, trades = backtest(df)
            m  = compute_metrics(history, trades, ticker)
            results.append(m)
            print(f"[{ticker}]  Return: {m['Return']}  "
                  f"Sharpe: {m['Sharpe']}  "
                  f"Win rate: {m['Win Rate']}  "
                  f"Trades: {m['Trades']}")
        except Exception as e:
            print(f"[{ticker}]  ERROR: {e}")

    print("\n" + "━"*52)
    print("  SUMMARY")
    print("━"*52)
    for r in results:
        gain_str = r["Return"]
        print(f"  {r['Ticker']:<6}  {gain_str:<10}  "
              f"Sharpe {r['Sharpe']:<7}  "
              f"Win {r['Win Rate']:<8}  "
              f"{r['Trades']} trades")
    print("━"*52)

    plot_multi(results)

    # Push reminder
    print("\n[GIT]  N'oublie pas de pusher :")
    print("  git add multi_asset_results.png backtest_multi.py")
    print("  git commit -m 'Add multi-asset backtest'")
    print("  git push origin main")