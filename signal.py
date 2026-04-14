"""
Real-Time Signal Generator - Trading Bot v19
Author: Theo

Usage:
  python signal.py              -> signal QQQ
  python signal.py NVDA         -> signal NVDA
  python signal.py QQQ NVDA MSFT -> plusieurs tickers
"""

import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Active les couleurs ANSI sur Windows
os.system("color")

# ============================================================
# CONFIGURATION
# ============================================================
SCORE_MIN    = 4
SLOPE_WIN    = 3
PULLBACK_MAX = 0.015
VIX_HIGH     = 25
VIX_EXTREME  = 35

# ============================================================
# COULEURS (compatibles Windows CMD)
# ============================================================
def green(t):  return "\033[92m" + str(t) + "\033[0m"
def yellow(t): return "\033[93m" + str(t) + "\033[0m"
def red(t):    return "\033[91m" + str(t) + "\033[0m"
def gray(t):   return "\033[90m" + str(t) + "\033[0m"
def cyan(t):   return "\033[96m" + str(t) + "\033[0m"
def white(t):  return "\033[97m" + str(t) + "\033[0m"
def bold(t):   return "\033[1m"  + str(t) + "\033[0m"
def dim(t):    return "\033[2m"  + str(t) + "\033[0m"

def sig_color(sig, t):
    if sig == "BUY":            return green(t)
    if sig == "WAIT PULLBACK":  return yellow(t)
    if sig == "DANGER":         return red(t)
    return gray(t)


# ============================================================
# 1. DATA
# ============================================================
def load_ticker(ticker):
    df = yf.download(ticker, period="200d", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


def load_vix():
    try:
        v = yf.download("^VIX", period="5d", interval="1d", progress=False)
        if isinstance(v.columns, pd.MultiIndex):
            v.columns = v.columns.get_level_values(0)
        return float(v["Close"].iloc[-1])
    except Exception:
        return 20.0


# ============================================================
# 2. INDICATEURS
# ============================================================
def build_indicators(df):
    for w in [10, 20, 50, 200]:
        df["MA" + str(w)] = df["Close"].rolling(w).mean()

    df["MA20_slope"] = df["MA20"] - df["MA20"].shift(SLOPE_WIN)
    df["MA50_slope"] = df["MA50"] - df["MA50"].shift(SLOPE_WIN)

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
    df["BB_pos"] = (df["Close"] - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)

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
def check_gate(row):
    checks = []

    g1 = float(row["Close"]) > float(row["MA200"])
    checks.append(("Prix > MA200 (bull market)", g1))

    g2 = float(row["MA50_slope"]) > 0
    checks.append(("MA50 en hausse", g2))

    g3 = float(row["MA20_slope"]) > 0
    checks.append(("MA20 en hausse", g3))

    g4 = float(row["Mom_5"]) > 0
    checks.append(("Momentum 5j positif", g4))

    g5 = float(row["RSI"]) > 48
    checks.append(("RSI > 48", g5))

    gate_ok = g1 and g2 and g3 and g4 and g5
    return gate_ok, checks


# ============================================================
# 4. SCORE
# ============================================================
def check_score(row):
    score  = 0
    checks = []

    rsi_val = round(float(row["RSI"]), 1)
    bb_val  = round(float(row["BB_pos"]), 2)
    vol_val = round(float(row["Vol_ratio"]), 2)

    c1 = float(row["Close"]) > float(row["MA200"])
    checks.append(("Prix > MA200", c1))
    if c1: score += 1

    c2 = (float(row["MA10"]) > float(row["MA20"]) and
          float(row["MA20"]) > float(row["MA50"]))
    checks.append(("MA10 > MA20 > MA50", c2))
    if c2: score += 1

    c3 = 48 < rsi_val < 68
    checks.append(("RSI " + str(rsi_val) + " entre 48 et 68", c3))
    if c3: score += 1

    c4 = (float(row["MACD_Hist"]) > 0 and
          float(row["MACD"]) > float(row["MACD_Sig"]))
    checks.append(("MACD haussier", c4))
    if c4: score += 1

    c5 = 0.35 < bb_val < 0.80
    checks.append(("BB_pos " + str(bb_val) + " entre 0.35 et 0.80", c5))
    if c5: score += 1

    c6 = vol_val > 1.0
    checks.append(("Volume " + str(vol_val) + "x la moyenne", c6))
    if c6: score += 1

    return score, checks


# ============================================================
# 5. PULLBACK
# ============================================================
def check_pullback(row):
    dist = float(row["Dist_MA20"])
    ok   = 0.0 <= dist <= PULLBACK_MAX
    return ok, dist


# ============================================================
# 6. VIX MULTIPLIER
# ============================================================
def vix_mult(vix):
    if vix > VIX_EXTREME:
        return 0.0
    elif vix > VIX_HIGH:
        return 0.5 + 0.5 * (VIX_EXTREME - vix) / (VIX_EXTREME - VIX_HIGH)
    return 1.0


# ============================================================
# 7. ANALYSE COMPLÈTE
# ============================================================
def analyze(ticker, vix_val):
    df  = load_ticker(ticker)
    df  = build_indicators(df)
    row = df.iloc[-1]

    price   = float(row["Close"])
    atr     = float(row["ATR"])

    gate_ok,  gate_checks  = check_gate(row)
    score,    score_checks = check_score(row)
    pb_ok,    dist_ma20    = check_pullback(row)
    mult                   = vix_mult(vix_val)

    stop  = price - 1.2 * atr
    tp    = price + 3.0 * atr
    sl_p  = (price - stop) / price * 100
    tp_p  = (tp - price)   / price * 100
    rr    = tp_p / sl_p if sl_p > 0 else 0

    # Signal
    if mult == 0.0:
        signal = "DANGER"
        reason = "VIX " + str(round(vix_val, 1)) + " > " + str(VIX_EXTREME) + " — marche en panique"
    elif not gate_ok:
        signal = "HOLD"
        reason = "Gate echoue — conditions insuffisantes"
    elif score < SCORE_MIN:
        signal = "HOLD"
        reason = "Score " + str(score) + "/6 < minimum " + str(SCORE_MIN)
    elif not pb_ok:
        signal = "WAIT PULLBACK"
        d_str  = str(round(dist_ma20 * 100, 2))
        if dist_ma20 > PULLBACK_MAX:
            reason = "Prix trop haut vs MA20 (+" + d_str + "%) — attendre retour"
        else:
            reason = "Prix sous MA20 (" + d_str + "%) — attendre"
    else:
        signal = "BUY"
        reason = "Tous les criteres valides — score " + str(score) + "/6"

    if VIX_HIGH < vix_val <= VIX_EXTREME:
        m_str  = str(round(mult * 100))
        reason = reason + " — position reduite a " + m_str + "% (VIX eleve)"

    return {
        "ticker":       ticker,
        "price":        price,
        "signal":       signal,
        "reason":       reason,
        "gate_ok":      gate_ok,
        "gate_checks":  gate_checks,
        "score":        score,
        "score_checks": score_checks,
        "pb_ok":        pb_ok,
        "dist_ma20":    dist_ma20,
        "vix":          vix_val,
        "mult":         mult,
        "atr":          atr,
        "atr_pct":      atr / price * 100,
        "stop":         stop,
        "tp":           tp,
        "sl_pct":       sl_p,
        "tp_pct":       tp_p,
        "rr":           rr,
        "rsi":          float(row["RSI"]),
        "ma20":         float(row["MA20"]),
        "ma50":         float(row["MA50"]),
        "ma200":        float(row["MA200"]),
    }


# ============================================================
# 8. AFFICHAGE
# ============================================================
SEP = "=" * 52

def print_signal(r):
    sig = r["signal"]

    print("")
    print(cyan(SEP))
    print("  " + bold(r["ticker"]) + "  —  " + sig_color(sig, sig))
    print("  " + dim(r["reason"]))
    print(cyan(SEP))

    # Prix et indicateurs
    price_str   = "$" + str(round(r["price"], 2))
    rsi_str     = str(round(r["rsi"], 1))
    atr_str     = str(round(r["atr"], 2)) + "  (" + str(round(r["atr_pct"], 2)) + "%)"
    ma20_str    = str(round(r["ma20"], 2)) + "  (dist: " + str(round(r["dist_ma20"]*100, 2)) + "%)"
    vix_str     = str(round(r["vix"], 1)) + "  (mult: " + str(round(r["mult"]*100)) + "%)"

    print("  " + dim("Prix         ") + "  " + white(price_str))
    print("  " + dim("RSI (14)     ") + "  " + white(rsi_str))
    print("  " + dim("ATR          ") + "  " + white(atr_str))
    print("  " + dim("MA20         ") + "  " + white(ma20_str))
    print("  " + dim("VIX          ") + "  " + white(vix_str))

    # Niveaux SL / TP
    if sig in ("BUY", "WAIT PULLBACK"):
        print("")
        print("  " + white("Niveaux si entree maintenant :"))
        stop_str = "$" + str(round(r["stop"], 2)) + "  (-" + str(round(r["sl_pct"], 2)) + "%)"
        tp_str   = "$" + str(round(r["tp"],   2)) + "  (+" + str(round(r["tp_pct"], 2)) + "%)"
        rr_str   = str(round(r["rr"], 2)) + "x"
        print("  " + dim("Stop loss    ") + "  " + red(stop_str))
        print("  " + dim("Take profit  ") + "  " + green(tp_str))
        print("  " + dim("RR theorique ") + "  " + white(rr_str))

    # Gate
    print("")
    print("  " + white("GATE (5 conditions obligatoires) :"))
    for name, ok in r["gate_checks"]:
        icon = green("OK") if ok else red("NO")
        line = white(name) if ok else dim(name)
        print("  [" + icon + "]  " + line)

    # Score
    print("")
    score_title = "SCORE : " + str(r["score"]) + "/6  (min " + str(SCORE_MIN) + ")"
    print("  " + white(score_title))
    for name, ok in r["score_checks"]:
        icon = green("OK") if ok else red("NO")
        line = white(name) if ok else dim(name)
        print("  [" + icon + "]  " + line)

    # Pullback
    print("")
    if r["gate_ok"] and r["score"] >= SCORE_MIN:
        if r["pb_ok"]:
            print("  " + green("PULLBACK ACTIF — entree valide maintenant"))
        else:
            d = str(round(r["dist_ma20"] * 100, 2))
            print("  " + yellow("ATTENDRE PULLBACK — dist MA20 : " + d + "%"))
    elif not r["gate_ok"]:
        print("  " + red("GATE ECHOUE — pas de trade possible"))
    else:
        print("  " + gray("SCORE INSUFFISANT — pas de trade possible"))

    print(cyan(SEP))
    print("")


def print_summary(results):
    if len(results) <= 1:
        return

    print("")
    print(cyan(SEP))
    print("  " + bold("RESUME — " + str(len(results)) + " TICKERS"))
    print(cyan(SEP))

    header = (
        "  " +
        "Ticker".ljust(8) +
        "Prix".rjust(10) +
        "  " +
        "Signal".ljust(16) +
        "Score".rjust(7) +
        "RSI".rjust(7) +
        "VIX".rjust(7)
    )
    print(white(header))
    print("  " + "-" * 50)

    for r in results:
        sig  = r["signal"]
        line = (
            "  " +
            r["ticker"].ljust(8) +
            ("$" + str(round(r["price"], 2))).rjust(10) +
            "  " +
            sig.ljust(16) +
            (str(r["score"]) + "/6").rjust(7) +
            str(round(r["rsi"], 1)).rjust(7) +
            str(round(r["vix"], 1)).rjust(7)
        )
        print(sig_color(sig, line))

    print(cyan(SEP))
    print("")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["QQQ"]
    tickers = [t.upper() for t in tickers]

    now = datetime.now().strftime("%A %d %B %Y  %H:%M")

    print("")
    print(cyan(SEP))
    print("  " + bold("TRADING BOT — SIGNAL DU JOUR"))
    print("  " + dim(now))
    print(cyan(SEP))
    print("  Chargement des donnees...")

    vix_val = load_vix()

    results = []
    for ticker in tickers:
        try:
            r = analyze(ticker, vix_val)
            results.append(r)
            print_signal(r)
        except Exception as e:
            print(red("  [ERREUR] " + ticker + " : " + str(e)))

    print_summary(results)