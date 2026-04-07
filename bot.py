"""
Algorithmic Trading Bot v18 - Pullback Entry
Author: Theo
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
TICKER      = "QQQ"
START       = "2022-01-01"
END         = "2024-12-31"
INTERVAL    = "1d"
INITIAL_CAP = 10_000
FEES        = 0.001
SLIPPAGE    = 0.0005
TRAIN_RATIO = 0.33
SCORE_MIN   = 4
SLOPE_WIN   = 3

ATR_STOP      = 1.2
ATR_TP        = 3.0
ATR_TRAIL     = 1.0
ATR_TRAIL_ACT = 0.8

# Pullback : on entre si prix est proche de la MA20
# Max distance autorisée entre prix et MA20 = 1.5%
PULLBACK_MAX = 0.015


# ============================================================
# 1. DATA
# ============================================================
def load_data(ticker, start, end, interval):
    df = yf.download(ticker, start=start, end=end,
                     interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    print(f"[DATA]  {len(df)} candles — {ticker} ({start} → {end})")
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
    df["BB_pos"] = (
        (df["Close"] - (bb_mid - 2*bb_std)) /
        (4 * bb_std + 1e-9)
    )

    hl        = df["High"] - df["Low"]
    hc        = (df["High"] - df["Close"].shift()).abs()
    lc        = (df["Low"]  - df["Close"].shift()).abs()
    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    df["Vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1)
    df["Mom_5"]     = df["Close"].pct_change(5)
    df["Mom_10"]    = df["Close"].pct_change(10)

    # Distance prix / MA20 (normalisée)
    df["Dist_MA20"] = (df["Close"] - df["MA20"]) / df["MA20"]

    df.dropna(inplace=True)
    return df


# ============================================================
# 3. GATE — conditions obligatoires
# ============================================================
def entry_gate(row):
    bull_lt = row["Close"] > row["MA200"]
    ma50_up = row["MA50_slope"] > 0
    ma20_up = row["MA20_slope"] > 0
    mom_pos = row["Mom_5"] > 0
    rsi_ok  = row["RSI"] > 48
    return bull_lt and ma50_up and ma20_up and mom_pos and rsi_ok


# ============================================================
# 4. SCORE — 6 conditions
# ============================================================
def entry_score(row):
    score = 0
    if row["Close"] > row["MA200"]:                              score += 1
    if row["MA10"] > row["MA20"] and row["MA20"] > row["MA50"]:  score += 1
    if 48 < row["RSI"] < 68:                                     score += 1
    if row["MACD_Hist"] > 0 and row["MACD"] > row["MACD_Sig"]:  score += 1
    if 0.35 < row["BB_pos"] < 0.80:                              score += 1
    if row["Vol_ratio"] > 1.0:                                   score += 1
    return score


# ============================================================
# 5. PULLBACK CHECK
# ============================================================
def is_pullback(row):
    """
    On entre seulement si le prix est proche de la MA20.
    Distance max = PULLBACK_MAX (1.5%).
    Ça évite d'acheter en haut d'un move — on attend le retour.
    """
    dist = row["Dist_MA20"]
    # Prix entre MA20 et +1.5% au-dessus de MA20
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

    # Système de patience : on valide le signal puis on attend le pullback
    signal_active = False   # signal validé, on attend le pullback
    signal_score  = 0
    patience      = 0       # jours d'attente depuis le signal
    MAX_PATIENCE  = 5       # max 5 jours d'attente pour le pullback

    history        = []
    buy_pts        = []
    sell_pts       = []
    trades         = []
    scores_all     = []
    gate_blocked   = 0
    pullback_waited= 0
    atr_log        = []

    for i in range(len(df_test)):
        row      = df_test.iloc[i]
        p        = float(row["Close"])
        atr      = float(row["ATR"])
        gain_pct = (p - entry_price) / entry_price if in_trade else 0.0

        atr_log.append(atr)

        # ── EXITS ─────────────────────────────────────────────
        if in_trade:
            peak_price  = max(peak_price, p)
            trailing_on = p >= trail_act
            floor       = (peak_price - ATR_TRAIL * atr
                           if trailing_on else stop_price)

            if p >= tp_price:
                proceeds = shares * p * (1 - SLIPPAGE) * (1 - FEES)
                pnl      = proceeds - shares * entry_price
                trades.append({"pnl": pnl,
                                "return_pct": gain_pct * 100,
                                "type": "take_profit"})
                balance, shares, in_trade = proceeds, 0.0, False
                signal_active = False
                sell_pts.append(i)

            elif p <= floor:
                proceeds  = shares * p * (1 - SLIPPAGE) * (1 - FEES)
                pnl       = proceeds - shares * entry_price
                exit_type = "trailing_stop" if trailing_on else "stop_loss"
                trades.append({"pnl": pnl,
                                "return_pct": gain_pct * 100,
                                "type": exit_type})
                balance, shares, in_trade = proceeds, 0.0, False
                signal_active = False
                sell_pts.append(i)

        # ── LOGIQUE D'ENTRÉE ──────────────────────────────────
        if not in_trade and balance > 100:
            gate = entry_gate(row)
            sc   = entry_score(row)
            scores_all.append(sc)

            # Étape 1 : valider le signal (gate + score)
            if not signal_active:
                if gate and sc >= SCORE_MIN:
                    signal_active = True
                    signal_score  = sc
                    patience      = 0
                elif not gate:
                    gate_blocked += 1

            # Étape 2 : si signal actif, attendre le pullback
            if signal_active and not in_trade:
                patience += 1

                if is_pullback(row):
                    # Pullback atteint → on entre
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

                    sl_pct = (exec_p - stop_price) / exec_p * 100
                    tp_pct = (tp_price - exec_p)   / exec_p * 100
                    print(f"  [BUY] #{len(buy_pts):02d}  "
                          f"price=${exec_p:.2f}  "
                          f"ATR={atr/exec_p*100:.2f}%  "
                          f"SL=-{sl_pct:.2f}%  TP=+{tp_pct:.2f}%  "
                          f"(pullback j+{patience})")

                elif patience >= MAX_PATIENCE:
                    # Trop long → on annule le signal
                    signal_active = False
                    pullback_waited += 1
        else:
            scores_all.append(0)

        history.append(balance + shares * p if in_trade else balance)

    # Fermer position finale
    if in_trade and shares > 0:
        p        = float(df_test.iloc[-1]["Close"])
        gain_pct = (p - entry_price) / entry_price
        proceeds = shares * p * (1 - FEES)
        pnl      = proceeds - shares * entry_price
        trades.append({"pnl": pnl,
                        "return_pct": gain_pct * 100,
                        "type": "end_close"})
        history[-1] = proceeds

    print(f"\n[GATE]     Entrées bloquées       : {gate_blocked}")
    print(f"[PULLBACK] Signaux expirés (5j)   : {pullback_waited}")
    print(f"[TEST]     Trades déclenchés      : {len(buy_pts)}")
    return df_test, history, buy_pts, sell_pts, trades, scores_all, atr_log


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

    by_type = {}
    for t in trades:
        by_type[t["type"]] = by_type.get(t["type"], 0) + 1

    m = {
        "Final balance":   f"${final:,.2f}",
        "Total return":    f"{gain:+.2f}%",
        "Sharpe ratio":    f"{sharpe:.3f}",
        "Sortino ratio":   f"{sortino:.3f}",
        "Calmar ratio":    f"{calmar:.3f}",
        "Max drawdown":    f"{dd:.2f}%",
        "Total trades":    n,
        "Win rate":        f"{win_rate:.1f}%",
        "Profit factor":   f"{pf:.2f}",
        "Avg win/trade":   f"{avg_win:+.2f}%",
        "Avg loss/trade":  f"{avg_loss:+.2f}%",
        "RR ratio":        f"{rr:.2f}x",
        "Exit breakdown":  str(by_type),
    }

    print("\n" + "━"*48)
    print("  PERFORMANCE REPORT")
    print("━"*48)
    for k, v in m.items():
        print(f"  {k:<24} {v}")
    print("━"*48)
    return m


# ============================================================
# 8. VISUALISATION
# ============================================================
def plot_results(df_test, history, buy_pts, sell_pts,
                 metrics, trades, scores_all, atr_log):

    fig = plt.figure(figsize=(16, 13), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(4, 1, hspace=0.5,
                            height_ratios=[3, 2, 1.2, 1.2])

    prices = df_test["Close"].values
    ma200  = df_test["MA200"].values
    ma20   = df_test["MA20"].values
    h      = pd.Series(history, dtype=float)
    idx    = list(range(len(df_test)))

    # ── Price ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    bull = prices > ma200
    ax1.fill_between(idx, prices, prices.min(),
                     where=bull,  alpha=0.06, color="#1D9E75")
    ax1.fill_between(idx, prices, prices.min(),
                     where=~bull, alpha=0.06, color="#E24B4A")
    ax1.plot(prices,                  color="#e0e0e0", lw=1,   label="Price")
    ax1.plot(df_test["MA10"].values,  color="#3B8BD4", lw=0.9, label="MA10")
    ax1.plot(ma20,                    color="#A09FE8", lw=0.9, label="MA20 (pullback)")
    ax1.plot(df_test["MA50"].values,  color="#EF9F27", lw=0.9, label="MA50")
    ax1.plot(ma200,                   color="#E24B4A", lw=0.8,
             linestyle="--",          label="MA200")
    for i in buy_pts:
        ax1.scatter(i, prices[i]*0.997, color="#1D9E75",
                    s=80, zorder=5, marker="^")
    for i in sell_pts:
        ax1.scatter(i, prices[i]*1.003, color="#E24B4A",
                    s=80, zorder=5, marker="v")
    ax1.set_title(
        f"{TICKER}  2023→2024  |  ▲ BUY sur pullback MA20  ▼ SELL  |  "
        f"SL×{ATR_STOP}ATR  TP×{ATR_TP}ATR  max attente {5}j"
    )
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.15)
    ax1.grid(alpha=0.1)

    # ── Equity ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(h.values, color="#3B8BD4", lw=1.5, label="Equity curve")
    ax2.axhline(INITIAL_CAP, color="#555", lw=0.8,
                linestyle="--", label="Initial capital")
    ax2.fill_between(idx, INITIAL_CAP, h.values,
                     where=(h.values >= INITIAL_CAP),
                     alpha=0.15, color="#1D9E75")
    ax2.fill_between(idx, INITIAL_CAP, h.values,
                     where=(h.values <  INITIAL_CAP),
                     alpha=0.15, color="#E24B4A")
    ax2.set_title(
        f"Return {metrics['Total return']}  |  "
        f"Sharpe {metrics['Sharpe ratio']}  |  "
        f"Win rate {metrics['Win rate']}  |  "
        f"RR {metrics['RR ratio']}  |  "
        f"Drawdown {metrics['Max drawdown']}"
    )
    ax2.legend(fontsize=8, framealpha=0.15)
    ax2.grid(alpha=0.1)

    # ── Distance prix / MA20 (pullback zone) ──────────────────
    ax3 = fig.add_subplot(gs[2])
    dist_ma20 = df_test["Dist_MA20"].values * 100
    pb_colors = ["#1D9E75" if 0 <= d <= PULLBACK_MAX*100
                 else "#444444" for d in dist_ma20]
    ax3.bar(idx, dist_ma20, color=pb_colors, width=1, alpha=0.85)
    ax3.axhline(0,                  color="#888", lw=0.8, linestyle="--")
    ax3.axhline(PULLBACK_MAX * 100, color="#1D9E75", lw=0.9,
                linestyle="--", label=f"Pullback max +{PULLBACK_MAX*100:.1f}%")
    ax3.set_title(
        "Distance prix / MA20  |  "
        "vert = zone pullback valide (entrée autorisée)"
    )
    ax3.legend(fontsize=8, framealpha=0.15)
    ax3.grid(alpha=0.1)

    # ── RSI ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(df_test["RSI"].values, color="#A09FE8", lw=0.9, label="RSI (14)")
    ax4.axhline(70, color="#E24B4A", lw=0.7, linestyle="--", label="70")
    ax4.axhline(48, color="#1D9E75", lw=0.7, linestyle="--", label="48 (gate)")
    ax4.fill_between(idx, 70, 100, alpha=0.07, color="#E24B4A")
    ax4.fill_between(idx, 0,  48,  alpha=0.07, color="#E24B4A")
    ax4.set_ylim(0, 100)
    ax4.set_title("RSI (14)")
    ax4.legend(fontsize=8, framealpha=0.15, ncol=3)
    ax4.grid(alpha=0.1)

    plt.savefig("bot_results.png", dpi=150,
                bbox_inches="tight", facecolor="#0d0d0d")
    plt.show()
    print("[PLOT]  Saved → bot_results.png")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "━"*48)
    print(f"  ML TRADING BOT v18 — {TICKER}")
    print("━"*48 + "\n")

    df                                               = load_data(TICKER, START, END, INTERVAL)
    df                                               = build_indicators(df)
    df_test, history, buys, sells, trades, sc, atr_l = backtest(df)
    metrics                                          = compute_metrics(history, trades)
    plot_results(df_test, history, buys, sells,
                 metrics, trades, sc, atr_l)