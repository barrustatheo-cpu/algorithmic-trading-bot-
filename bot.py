"""
Algorithmic Trading Bot v19 - Kelly Criterion + VIX Regime Filter
Author: Theo
Fixes: partial position sizing, VIX date lookup, history tracking
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

# Kelly Criterion
KELLY_FRACTION = 0.5   # demi-Kelly = plus conservateur
MAX_POSITION   = 0.95  # max 95% du capital
MIN_POSITION   = 0.10  # min 10% du capital

# VIX
VIX_HIGH    = 25   # position réduite
VIX_EXTREME = 35   # pas de trade


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


def load_vix(start, end):
    vix = yf.download("^VIX", start=start, end=end,
                      interval="1d", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = vix[["Close"]].rename(columns={"Close": "VIX"})
    vix.dropna(inplace=True)
    # Normalise l'index en date pure pour éviter les erreurs de timezone
    vix.index = pd.to_datetime(vix.index).normalize()
    print(f"[VIX]   {len(vix)} candles chargées")
    return vix


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
    return 0.0 <= float(row["Dist_MA20"]) <= PULLBACK_MAX


# ============================================================
# 6. KELLY CRITERION
# ============================================================
def kelly_position(trades_history):
    """
    Calcule la fraction optimale du capital à investir.
    Kelly = W - (1-W)/R
    W = win rate sur les 20 derniers trades
    R = ratio avg_gain / avg_loss
    Retourne une fraction entre MIN_POSITION et MAX_POSITION.
    """
    if len(trades_history) < 5:
        return 0.50  # pas assez d'historique → 50% par défaut

    recent   = trades_history[-20:]
    wins     = [t for t in recent if t["pnl"] > 0]
    losses   = [t for t in recent if t["pnl"] < 0]

    if not wins or not losses:
        return 0.50

    W        = len(wins) / len(recent)
    avg_win  = np.mean([t["pnl"] for t in wins])
    avg_loss = abs(np.mean([t["pnl"] for t in losses]))
    R        = avg_win / avg_loss if avg_loss > 0 else 1.0

    kelly = W - (1 - W) / R
    kelly = kelly * KELLY_FRACTION
    kelly = max(MIN_POSITION, min(MAX_POSITION, kelly))
    return kelly


# ============================================================
# 7. VIX REGIME
# ============================================================
def get_vix_value(vix_df, date):
    """
    Récupère le VIX pour une date donnée.
    Utilise asof() pour trouver la valeur la plus récente disponible.
    Retourne 20.0 (neutre) si pas de données.
    """
    try:
        date_norm = pd.Timestamp(date).normalize()
        idx       = vix_df.index.asof(date_norm)
        if pd.isna(idx):
            return 20.0
        return float(vix_df.loc[idx, "VIX"])
    except Exception:
        return 20.0


def get_vix_multiplier(vix_value):
    """
    VIX < 25  → position 100%
    VIX 25-35 → position 50-100% (interpolation)
    VIX > 35  → position 0% (pas de trade)
    """
    if vix_value > VIX_EXTREME:
        return 0.0
    elif vix_value > VIX_HIGH:
        return 0.5 + 0.5 * (VIX_EXTREME - vix_value) / (VIX_EXTREME - VIX_HIGH)
    else:
        return 1.0


# ============================================================
# 8. BACKTEST
# ============================================================
def backtest(df, vix_df):
    split   = int(len(df) * TRAIN_RATIO)
    df_test = df.iloc[split:].copy().reset_index(drop=False)

    # FIX : on sépare cash (non-investi) et shares (investi)
    cash        = float(INITIAL_CAP)  # capital non-investi
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

    history      = []
    buy_pts      = []
    sell_pts     = []
    trades       = []
    vix_log      = []
    position_log = []  # (idx, position_pct) pour chaque trade entré

    for i in range(len(df_test)):
        row      = df_test.iloc[i]
        p        = float(row["Close"])
        atr      = float(row["ATR"])
        gain_pct = (p - entry_price) / entry_price if in_trade else 0.0

        # VIX du jour
        try:
            date = row["Date"]
        except Exception:
            date = df_test.index[i]
        vix_val = get_vix_value(vix_df, date)
        vix_log.append(vix_val)

        # ── EXITS ─────────────────────────────────────────────
        if in_trade:
            peak_price  = max(peak_price, p)
            trailing_on = p >= trail_act
            floor       = (peak_price - ATR_TRAIL * atr
                           if trailing_on else stop_price)

            # Take profit
            if p >= tp_price:
                proceeds  = shares * p * (1 - SLIPPAGE) * (1 - FEES)
                pnl       = proceeds - shares * entry_price
                trades.append({"pnl": pnl,
                                "return_pct": gain_pct * 100,
                                "type": "take_profit"})
                # FIX : ajoute les proceeds au cash restant
                cash    += proceeds
                shares   = 0.0
                in_trade = False
                signal_active = False
                sell_pts.append(i)

            # Trailing stop ou stop dur
            elif p <= floor:
                proceeds  = shares * p * (1 - SLIPPAGE) * (1 - FEES)
                pnl       = proceeds - shares * entry_price
                exit_type = "trailing_stop" if trailing_on else "stop_loss"
                trades.append({"pnl": pnl,
                                "return_pct": gain_pct * 100,
                                "type": exit_type})
                # FIX : ajoute les proceeds au cash restant
                cash    += proceeds
                shares   = 0.0
                in_trade = False
                signal_active = False
                sell_pts.append(i)

        # ── ENTRÉE ────────────────────────────────────────────
        if not in_trade and cash > 100:
            gate = entry_gate(row)
            sc   = entry_score(row)

            if not signal_active:
                if gate and sc >= SCORE_MIN:
                    signal_active = True
                    patience      = 0

            if signal_active:
                patience += 1
                if is_pullback(row):
                    vix_mult = get_vix_multiplier(vix_val)

                    if vix_mult == 0.0:
                        # VIX extrême → on annule le signal
                        signal_active = False
                        print(f"  [VIX]  Trade bloqué — VIX={vix_val:.1f}")
                    else:
                        # Kelly position sizing
                        kelly    = kelly_position(trades)
                        position = kelly * vix_mult
                        position = max(MIN_POSITION, min(MAX_POSITION, position))

                        exec_p       = p * (1 + SLIPPAGE)
                        capital_used = cash * position
                        shares       = (capital_used * (1 - FEES)) / exec_p
                        entry_price  = exec_p
                        peak_price   = exec_p
                        stop_price   = exec_p - ATR_STOP     * atr
                        tp_price     = exec_p + ATR_TP        * atr
                        trail_act    = exec_p + ATR_TRAIL_ACT * atr

                        # FIX : soustrait seulement le capital investi
                        cash        -= capital_used
                        in_trade     = True
                        signal_active = False
                        buy_pts.append(i)
                        position_log.append((i, position * 100))

                        print(f"  [BUY] #{len(buy_pts):02d}  "
                              f"price=${exec_p:.2f}  "
                              f"VIX={vix_val:.1f}  "
                              f"Kelly={kelly*100:.0f}%  "
                              f"Pos={position*100:.0f}%  "
                              f"Investi=${capital_used:,.0f}  "
                              f"Cash=${cash:,.0f}")

                elif patience >= MAX_PATIENCE:
                    signal_active = False

        # Portfolio value = cash + valeur des shares
        history.append(cash + shares * p)

    # Fermer position finale
    if in_trade and shares > 0:
        p        = float(df_test.iloc[-1]["Close"])
        gain_pct = (p - entry_price) / entry_price
        proceeds = shares * p * (1 - FEES)
        pnl      = proceeds - shares * entry_price
        trades.append({"pnl": pnl,
                        "return_pct": gain_pct * 100,
                        "type": "end_close"})
        # FIX : valeur finale = cash + proceeds
        history[-1] = cash + proceeds

    if position_log:
        avg_pos = np.mean([p for _, p in position_log])
        print(f"\n[KELLY]  Position moyenne par trade : {avg_pos:.1f}%")
    print(f"[VIX]    VIX moyen : {np.mean(vix_log):.1f}")

    return df_test, history, buy_pts, sell_pts, trades, vix_log, position_log


# ============================================================
# 9. METRICS
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
# 10. VISUALISATION
# ============================================================
def plot_results(df_test, history, buy_pts, sell_pts,
                 metrics, trades, vix_log, position_log):

    fig = plt.figure(figsize=(16, 15), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(5, 1, hspace=0.5,
                            height_ratios=[3, 2, 1.2, 1.2, 1.2])

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
    ax1.plot(ma20,                    color="#A09FE8", lw=0.9, label="MA20")
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
        f"{TICKER}  {START}→{END}  |  ▲ BUY  ▼ SELL  |  "
        f"Kelly Criterion + VIX Filter"
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

    # ── VIX ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    vix_colors = ["#E24B4A" if v > VIX_EXTREME else
                  "#EF9F27" if v > VIX_HIGH    else
                  "#1D9E75" for v in vix_log]
    ax3.bar(idx[:len(vix_log)], vix_log,
            color=vix_colors, width=1, alpha=0.85)
    ax3.axhline(VIX_HIGH,    color="#EF9F27", lw=0.9,
                linestyle="--", label=f"VIX {VIX_HIGH} (pos réduite)")
    ax3.axhline(VIX_EXTREME, color="#E24B4A", lw=0.9,
                linestyle="--", label=f"VIX {VIX_EXTREME} (bloqué)")
    ax3.set_title(
        "VIX  |  vert = normal  |  orange = position réduite  |  rouge = trade bloqué"
    )
    ax3.legend(fontsize=8, framealpha=0.15, ncol=2)
    ax3.grid(alpha=0.1)

    # ── Kelly Position Size ───────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    if position_log:
        pos_x = [pl[0] for pl in position_log]
        pos_y = [pl[1] for pl in position_log]
        ax4.bar(pos_x, pos_y, color="#A09FE8",
                width=3, alpha=0.85, label="Position size %")
        ax4.axhline(50, color="#555", lw=0.8,
                    linestyle="--", label="50% (default)")
    ax4.set_ylim(0, 100)
    ax4.set_title("Kelly Position Sizing  (% du capital investi par trade)")
    ax4.legend(fontsize=8, framealpha=0.15)
    ax4.grid(alpha=0.1)

    # ── RSI ───────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[4])
    ax5.plot(df_test["RSI"].values, color="#A09FE8", lw=0.9, label="RSI (14)")
    ax5.axhline(70, color="#E24B4A", lw=0.7, linestyle="--", label="70")
    ax5.axhline(48, color="#1D9E75", lw=0.7, linestyle="--", label="48 (gate)")
    ax5.fill_between(idx, 70, 100, alpha=0.07, color="#E24B4A")
    ax5.fill_between(idx, 0,  48,  alpha=0.07, color="#E24B4A")
    ax5.set_ylim(0, 100)
    ax5.set_title("RSI (14)")
    ax5.legend(fontsize=8, framealpha=0.15, ncol=3)
    ax5.grid(alpha=0.1)

    plt.savefig("bot_results.png", dpi=150,
                bbox_inches="tight", facecolor="#0d0d0d")
    plt.show()
    print("[PLOT]  Saved → bot_results.png")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "━"*48)
    print(f"  ML TRADING BOT v19 — {TICKER}")
    print(f"  Kelly Criterion + VIX Regime Filter")
    print("━"*48 + "\n")

    df     = load_data(TICKER, START, END, INTERVAL)
    vix_df = load_vix(START, END)
    df     = build_indicators(df)

    df_test, history, buys, sells, trades, vix_log, pos_log = backtest(df, vix_df)
    metrics = compute_metrics(history, trades)
    plot_results(df_test, history, buys, sells,
                 metrics, trades, vix_log, pos_log)