"""
Parameter Optimizer — Trading Bot v19
Author: Theo

Teste automatiquement des centaines de combinaisons de paramètres
et trouve la configuration optimale pour maximiser le Sharpe ratio.

Usage : python optimize.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import warnings
warnings.filterwarnings("ignore")

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

# Grille de paramètres à tester
PARAM_GRID = {
    "atr_stop":      [0.8, 1.0, 1.2, 1.5, 2.0],
    "atr_tp":        [2.0, 2.5, 3.0, 3.5, 4.0],
    "atr_trail":     [0.8, 1.0, 1.2],
    "atr_trail_act": [0.5, 0.8, 1.0],
    "score_min":     [3, 4, 5],
    "pullback_max":  [0.01, 0.015, 0.02],
}


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
        df["MA" + str(w)] = df["Close"].rolling(w).mean()

    df["MA20_slope"] = df["MA20"] - df["MA20"].shift(3)
    df["MA50_slope"] = df["MA50"] - df["MA50"].shift(3)

    delta     = df["Close"].diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

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
# 3. GATE + SCORE
# ============================================================
def entry_gate(row):
    return (float(row["Close"])      > float(row["MA200"]) and
            float(row["MA50_slope"]) > 0 and
            float(row["MA20_slope"]) > 0 and
            float(row["Mom_5"])      > 0 and
            float(row["RSI"])        > 48)


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
# 4. BACKTEST RAPIDE (sans logs ni graphiques)
# ============================================================
def backtest_fast(df, params):
    split   = int(len(df) * TRAIN_RATIO)
    df_test = df.iloc[split:].copy().reset_index(drop=False)

    atr_stop      = params["atr_stop"]
    atr_tp        = params["atr_tp"]
    atr_trail     = params["atr_trail"]
    atr_trail_act = params["atr_trail_act"]
    score_min     = params["score_min"]
    pullback_max  = params["pullback_max"]

    cash        = float(INITIAL_CAP)
    shares      = 0.0
    entry_price = 0.0
    peak_price  = 0.0
    stop_price  = 0.0
    tp_price    = 0.0
    trail_act   = 0.0
    in_trade    = False
    sig_active  = False
    patience    = 0

    history = []
    trades  = []

    for i in range(len(df_test)):
        row      = df_test.iloc[i]
        p        = float(row["Close"])
        atr      = float(row["ATR"])
        gain_pct = (p - entry_price) / entry_price if in_trade else 0.0

        # Exits
        if in_trade:
            peak_price  = max(peak_price, p)
            trailing_on = p >= trail_act
            floor       = (peak_price - atr_trail * atr
                           if trailing_on else stop_price)

            if p >= tp_price:
                proceeds = shares * p * (1 - SLIPPAGE) * (1 - FEES)
                trades.append({"pnl": proceeds - shares*entry_price,
                                "ret": gain_pct*100, "type": "tp"})
                cash    += proceeds
                shares   = 0.0
                in_trade = False
                sig_active = False

            elif p <= floor:
                proceeds = shares * p * (1 - SLIPPAGE) * (1 - FEES)
                t        = "trail" if trailing_on else "sl"
                trades.append({"pnl": proceeds - shares*entry_price,
                                "ret": gain_pct*100, "type": t})
                cash    += proceeds
                shares   = 0.0
                in_trade = False
                sig_active = False

        # Entrée
        if not in_trade and cash > 100:
            gate = entry_gate(row)
            sc   = entry_score(row)

            if not sig_active and gate and sc >= score_min:
                sig_active = True
                patience   = 0

            if sig_active:
                patience += 1
                dist = float(row["Dist_MA20"])
                if 0.0 <= dist <= pullback_max:
                    exec_p      = p * (1 + SLIPPAGE)
                    shares      = (cash * (1 - FEES)) / exec_p
                    entry_price = exec_p
                    peak_price  = exec_p
                    stop_price  = exec_p - atr_stop     * atr
                    tp_price    = exec_p + atr_tp        * atr
                    trail_act   = exec_p + atr_trail_act * atr
                    cash        = 0.0
                    in_trade    = True
                    sig_active  = False
                elif patience >= 5:
                    sig_active = False

        history.append(cash + shares * p)

    # Fermer position finale
    if in_trade and shares > 0:
        p        = float(df_test.iloc[-1]["Close"])
        gain_pct = (p - entry_price) / entry_price
        proceeds = shares * p * (1 - FEES)
        trades.append({"pnl": proceeds - shares*entry_price,
                        "ret": gain_pct*100, "type": "close"})
        history[-1] = cash + proceeds

    return history, trades


# ============================================================
# 5. METRICS
# ============================================================
def metrics(history, trades):
    if len(trades) < 2:
        return None

    h   = pd.Series(history, dtype=float)
    ret = h.pct_change().dropna()

    final   = float(h.iloc[-1])
    gain    = (final - INITIAL_CAP) / INITIAL_CAP * 100
    sharpe  = float((ret.mean() / ret.std()) * np.sqrt(252)) if ret.std() > 0 else 0
    dd      = float((h / h.cummax() - 1).min() * 100)
    calmar  = gain / abs(dd) if dd != 0 else 0

    pnl_list = [t["pnl"] for t in trades]
    n        = len(pnl_list)
    wins     = [x for x in pnl_list if x > 0]
    losses   = [x for x in pnl_list if x < 0]
    win_rate = len(wins) / n * 100 if n else 0
    pf       = abs(sum(wins) / sum(losses)) if losses else 0

    ret_list = [t["ret"] for t in trades]
    avg_win  = np.mean([r for r in ret_list if r > 0]) if wins   else 0
    avg_loss = np.mean([r for r in ret_list if r < 0]) if losses else 0
    rr       = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    return {
        "gain":     gain,
        "sharpe":   sharpe,
        "dd":       dd,
        "calmar":   calmar,
        "n":        n,
        "win_rate": win_rate,
        "pf":       pf,
        "rr":       rr,
    }


# ============================================================
# 6. OPTIMISATION
# ============================================================
def optimize(df):
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))

    total  = len(combos)
    print("\n[OPT]  " + str(total) + " combinaisons a tester...")
    print("[OPT]  Cela peut prendre 2-3 minutes...\n")

    results = []

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        # Filtre : TP doit etre plus grand que SL
        if params["atr_tp"] <= params["atr_stop"]:
            continue

        history, trades = backtest_fast(df, params)
        m               = metrics(history, trades)

        if m is None:
            continue

        # On optimise sur le Sharpe ratio
        if m["sharpe"] > 0 and m["n"] >= 5:
            results.append({**params, **m})

        # Progression
        if (idx + 1) % 200 == 0:
            pct = round((idx + 1) / total * 100, 1)
            print("  " + str(pct) + "% — " + str(len(results)) + " configs positives trouvees...")

    return results


# ============================================================
# 7. AFFICHAGE RÉSULTATS
# ============================================================
def print_results(results):
    if not results:
        print("\n[!] Aucune configuration positive trouvee.")
        return

    # Trier par Sharpe ratio
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    top10 = results[:10]

    sep = "=" * 90

    print("\n" + sep)
    print("  TOP 10 CONFIGURATIONS  (triees par Sharpe ratio)")
    print(sep)

    header = (
        "  Rk  " +
        "SL".rjust(5) +
        "TP".rjust(5) +
        "Tr".rjust(5) +
        "TrA".rjust(5) +
        "Sc".rjust(4) +
        "PB".rjust(6) +
        "Return".rjust(9) +
        "Sharpe".rjust(8) +
        "Drawdown".rjust(10) +
        "WinR".rjust(7) +
        "RR".rjust(6) +
        "Trades".rjust(8)
    )
    print(header)
    print("  " + "-" * 86)

    for i, r in enumerate(top10):
        line = (
            "  " + str(i+1).ljust(4) +
            str(r["atr_stop"]).rjust(5) +
            str(r["atr_tp"]).rjust(5) +
            str(r["atr_trail"]).rjust(5) +
            str(r["atr_trail_act"]).rjust(5) +
            str(r["score_min"]).rjust(4) +
            str(r["pullback_max"]).rjust(6) +
            (str(round(r["gain"], 2)) + "%").rjust(9) +
            str(round(r["sharpe"], 3)).rjust(8) +
            (str(round(r["dd"], 2)) + "%").rjust(10) +
            (str(round(r["win_rate"], 1)) + "%").rjust(7) +
            (str(round(r["rr"], 2)) + "x").rjust(6) +
            str(r["n"]).rjust(8)
        )
        print(line)

    print(sep)

    # Meilleure config
    best = top10[0]
    print("\n  MEILLEURE CONFIG (Sharpe " + str(round(best["sharpe"], 3)) + ") :")
    print("")
    print("  ATR_STOP      = " + str(best["atr_stop"]))
    print("  ATR_TP        = " + str(best["atr_tp"]))
    print("  ATR_TRAIL     = " + str(best["atr_trail"]))
    print("  ATR_TRAIL_ACT = " + str(best["atr_trail_act"]))
    print("  SCORE_MIN     = " + str(best["score_min"]))
    print("  PULLBACK_MAX  = " + str(best["pullback_max"]))
    print("")
    print("  Return   : " + str(round(best["gain"], 2)) + "%")
    print("  Sharpe   : " + str(round(best["sharpe"], 3)))
    print("  Drawdown : " + str(round(best["dd"], 2)) + "%")
    print("  Win rate : " + str(round(best["win_rate"], 1)) + "%")
    print("  RR ratio : " + str(round(best["rr"], 2)) + "x")
    print("  Trades   : " + str(best["n"]))
    print("")
    print("  Copie ces valeurs dans bot.py pour optimiser les performances !")
    print(sep + "\n")

    return best


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 52)
    print("  PARAMETER OPTIMIZER — " + TICKER)
    print("  " + START + " -> " + END)
    print("=" * 52)

    print("\n[DATA]  Chargement des donnees...")
    df = load_data()
    df = build_indicators(df)
    print("[DATA]  " + str(len(df)) + " candles chargees")

    results = optimize(df)
    best    = print_results(results)