"""
AI Trading Assistant - Gemini via requests (compatible Python 3.13)
Author: Theo
"""

import requests
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION BOT
# ============================================================
SCORE_MIN    = 4
SLOPE_WIN    = 3
PULLBACK_MAX = 0.01
VIX_HIGH     = 25
VIX_EXTREME  = 35
ATR_STOP     = 1.5
ATR_TP       = 3.0

POPULAR_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "QQQ",  "SPY",  "AMD",  "INTC",  "NFLX", "JPM",  "BAC",
    "V",    "MA",   "PYPL", "CRM",   "ADBE", "ORCL", "QCOM",
    "MU",   "AVGO", "TXN",  "NOW",   "UBER", "SHOP", "COIN"
]

SYSTEM_PROMPT = """Tu es un assistant expert en trading algorithmique.
Tu analyses les marches en temps reel avec un bot v20 base sur MA, RSI, MACD,
Bollinger, ATR, VIX et Kelly Criterion.
Parle directement et simplement comme un ami trader experimente.
Explique le pourquoi de chaque signal. Mentionne toujours les risques.
Tu n'es pas un conseiller financier agree."""

IGNORE_WORDS = {
    "BUY","HOLD","SELL","THE","AND","FOR","SUR","ET","LE","LA","LES",
    "DE","DU","AU","AUX","EST","MON","TON","SON","QUE","QUI","PAR",
    "DANS","AVEC","COMMENT","POURQUOI","QUAND","QUOI","QUEL","UNE",
    "UN","IL","ELLE","ON","NOUS","VOUS","ILS","ELLES","MOI","TOI",
    "LUI","EUX","CAR","MAIS","OU","DONC","OR","NI","AI","IA","RSI",
    "ATR","VIX","MA","MACD","BOT","SIGNAL","ACTION","ACTIONS"
}

SCAN_KEYWORDS = [
    "scanne","scan","cherche","trouve","meilleures actions",
    "top actions","analyse tout","quelles actions","marche en ce moment",
    "meilleure","meilleures"
]


# ============================================================
# 1. INDICATEURS
# ============================================================
def load_and_analyze(ticker):
    try:
        df = yf.download(ticker, period="200d", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        if len(df) < 50:
            return None

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
        df["BB_pos"] = (df["Close"] - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-9)

        hl        = df["High"] - df["Low"]
        hc        = (df["High"] - df["Close"].shift()).abs()
        lc        = (df["Low"]  - df["Close"].shift()).abs()
        df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

        df["Vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1)
        df["Mom_5"]     = df["Close"].pct_change(5)
        df["Mom_20"]    = df["Close"].pct_change(20)
        df["Dist_MA20"] = (df["Close"] - df["MA20"]) / df["MA20"]

        df.dropna(inplace=True)
        return df
    except Exception:
        return None


def get_vix():
    try:
        v = yf.download("^VIX", period="5d", interval="1d", progress=False)
        if isinstance(v.columns, pd.MultiIndex):
            v.columns = v.columns.get_level_values(0)
        return float(v["Close"].iloc[-1])
    except Exception:
        return 20.0


def compute_signal(ticker):
    df = load_and_analyze(ticker)
    if df is None:
        return None

    row   = df.iloc[-1]
    price = float(row["Close"])
    atr   = float(row["ATR"])
    vix   = get_vix()

    gate_checks = {
        "Prix > MA200":        float(row["Close"]) > float(row["MA200"]),
        "MA50 en hausse":      float(row["MA50_slope"]) > 0,
        "MA20 en hausse":      float(row["MA20_slope"]) > 0,
        "Momentum 5j positif": float(row["Mom_5"]) > 0,
        "RSI > 48":            float(row["RSI"]) > 48,
    }
    gate_ok = all(gate_checks.values())

    score_checks = {
        "Prix > MA200":       float(row["Close"]) > float(row["MA200"]),
        "MA10>MA20>MA50":     float(row["MA10"]) > float(row["MA20"]) > float(row["MA50"]),
        "RSI sain (48-68)":   48 < float(row["RSI"]) < 68,
        "MACD haussier":      float(row["MACD_Hist"]) > 0 and float(row["MACD"]) > float(row["MACD_Sig"]),
        "BB neutre":          0.35 < float(row["BB_pos"]) < 0.80,
        "Volume > moyenne":   float(row["Vol_ratio"]) > 1.0,
    }
    score = sum(score_checks.values())

    dist_ma20   = float(row["Dist_MA20"])
    pullback_ok = 0.0 <= dist_ma20 <= PULLBACK_MAX

    if vix > VIX_EXTREME:
        signal = "DANGER"
    elif not gate_ok:
        signal = "HOLD"
    elif score < SCORE_MIN:
        signal = "HOLD"
    elif not pullback_ok:
        signal = "WAIT PULLBACK"
    else:
        signal = "BUY"

    return {
        "ticker":       ticker,
        "price":        round(price, 2),
        "signal":       signal,
        "gate_ok":      gate_ok,
        "gate_checks":  gate_checks,
        "score":        score,
        "score_checks": score_checks,
        "pullback_ok":  pullback_ok,
        "dist_ma20":    round(dist_ma20 * 100, 2),
        "vix":          round(vix, 1),
        "rsi":          round(float(row["RSI"]), 1),
        "atr":          round(atr, 2),
        "stop":         round(price - ATR_STOP * atr, 2),
        "tp":           round(price + ATR_TP   * atr, 2),
        "sl_pct":       round(ATR_STOP * atr / price * 100, 2),
        "tp_pct":       round(ATR_TP   * atr / price * 100, 2),
        "ma20":         round(float(row["MA20"]), 2),
        "ma50":         round(float(row["MA50"]), 2),
        "ma200":        round(float(row["MA200"]), 2),
        "perf_1w":      round(float(row["Mom_5"]) * 100, 2),
        "perf_1m":      round(float(row["Mom_20"]) * 100, 2),
    }


def scan_multiple(tickers):
    results = []
    for i, t in enumerate(tickers):
        print("  Scan " + str(i+1) + "/" + str(len(tickers)) + " — " + t + "...    ", end="\r")
        r = compute_signal(t)
        if r:
            results.append(r)
    print(" " * 50, end="\r")
    return results


# ============================================================
# 2. CONTEXTE MARCHE
# ============================================================
def build_context(data):
    if isinstance(data, list):
        buys  = [r for r in data if r["signal"] == "BUY"]
        waits = [r for r in data if r["signal"] == "WAIT PULLBACK"]
        holds = [r for r in data if r["signal"] == "HOLD"]
        lines = [
            "SCAN MARCHE — " + str(len(data)) + " actions",
            "BUY (" + str(len(buys)) + ") : " + ", ".join([r["ticker"] for r in buys]),
            "WAIT (" + str(len(waits)) + ") : " + ", ".join([r["ticker"] for r in waits]),
            "HOLD (" + str(len(holds)) + ") : " + ", ".join([r["ticker"] for r in holds]),
            "",
            "DETAILS BUY :"
        ]
        for r in buys:
            lines.append(
                "  " + r["ticker"] + " $" + str(r["price"]) +
                " RSI=" + str(r["rsi"]) +
                " Score=" + str(r["score"]) + "/6" +
                " SL=-" + str(r["sl_pct"]) + "%" +
                " TP=+" + str(r["tp_pct"]) + "%" +
                " 1W=" + str(r["perf_1w"]) + "%"
            )
        return "\n".join(lines)
    else:
        r  = data
        rr = round(r["tp_pct"] / r["sl_pct"], 2) if r["sl_pct"] > 0 else 0
        gate_str  = "\n".join(["  " + k + ": " + ("OK" if v else "NON")
                               for k, v in r["gate_checks"].items()])
        score_str = "\n".join(["  " + k + ": " + ("OK" if v else "NON")
                               for k, v in r["score_checks"].items()])
        return (
            "ANALYSE " + r["ticker"] + "\n"
            "Prix: $" + str(r["price"]) + "\n"
            "Signal: " + r["signal"] + "\n"
            "Score: " + str(r["score"]) + "/6\n"
            "RSI: " + str(r["rsi"]) + "\n"
            "VIX: " + str(r["vix"]) + "\n"
            "Dist MA20: " + str(r["dist_ma20"]) + "%\n\n"
            "GATE:\n" + gate_str + "\n\n"
            "SCORE:\n" + score_str + "\n\n"
            "Stop loss: $" + str(r["stop"]) + " (-" + str(r["sl_pct"]) + "%)\n"
            "Take profit: $" + str(r["tp"]) + " (+" + str(r["tp_pct"]) + "%)\n"
            "RR: " + str(rr) + "x\n"
            "MA20: $" + str(r["ma20"]) + "\n"
            "MA50: $" + str(r["ma50"]) + "\n"
            "MA200: $" + str(r["ma200"]) + "\n"
            "Perf 1W: " + str(r["perf_1w"]) + "%\n"
            "Perf 1M: " + str(r["perf_1m"]) + "%\n"
        )


# ============================================================
# 3. PARSING
# ============================================================
def parse_message(msg):
    msg_lower = msg.lower()
    for kw in SCAN_KEYWORDS:
        if kw in msg_lower:
            return "SCAN", None

    words   = msg.upper().split()
    tickers = []
    for w in words:
        clean = "".join(c for c in w if c.isalpha())
        if 1 <= len(clean) <= 5 and clean not in IGNORE_WORDS:
            tickers.append(clean)

    return "TICKERS", (tickers if tickers else None)


# ============================================================
# 4. APPEL GEMINI VIA REQUESTS
# ============================================================
def ask_gemini(api_key, history, user_message, market_context=None):
    if market_context:
        full_message = "[DONNEES MARCHE]\n" + market_context + "\n\n[QUESTION]\n" + user_message
    else:
        full_message = user_message

    # Construit les messages pour Gemini
    contents = []

    # Ajoute l'historique
    for msg in history:
        contents.append({
            "role": msg["role"],
            "parts": [{"text": msg["content"]}]
        })

    # Ajoute le message actuel
    contents.append({
        "role": "user",
        "parts": [{"text": full_message}]
    })

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + api_key
    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": 1024,
            "temperature": 0.7
        }
    }

    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=30
    )

    if response.status_code != 200:
        raise Exception("Erreur API " + str(response.status_code) + " : " + response.text[:200])

    result = response.json()
    reply  = result["candidates"][0]["content"]["parts"][0]["text"]

    # Ajoute à l'historique
    history.append({"role": "user",  "content": user_message})
    history.append({"role": "model", "content": reply})

    return reply


# ============================================================
# 5. MAIN
# ============================================================
def main():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("\n[ERREUR] GEMINI_API_KEY non definie.")
        print("Lance : set GEMINI_API_KEY=ta_cle_ici\n")
        return

    history = []

    print("\n" + "="*55)
    print("  ASSISTANT IA DE TRADING — Powered by Gemini")
    print("  Bot v20 | Kelly + VIX | Optimise")
    print("="*55)
    print("\nExemples :")
    print("  > Analyse NVDA")
    print("  > Signal sur AAPL et MSFT ?")
    print("  > Scanne les meilleures actions")
    print("  > Explique le RSI")
    print("\nTape 'exit' pour quitter.\n")

    while True:
        try:
            user_input = input("Toi : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nA bientot !")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "quitter", "bye"):
            print("\nAssistant : A bientot ! Bon trading !")
            break

        intent, tickers = parse_message(user_input)
        context = None

        if intent == "SCAN":
            print("\nAssistant : Je scanne " + str(len(POPULAR_TICKERS)) +
                  " actions, ca prend 30 secondes...\n")
            results = scan_multiple(POPULAR_TICKERS)
            context = build_context(results)

        elif tickers:
            analyzed = []
            for t in tickers[:5]:
                print("  Analyse " + t + "...", end="\r")
                r = compute_signal(t)
                if r:
                    analyzed.append(r)
            print(" " * 40, end="\r")

            if len(analyzed) == 1:
                context = build_context(analyzed[0])
            elif len(analyzed) > 1:
                context = build_context(analyzed)

        try:
            print("Assistant : ", end="", flush=True)
            reply = ask_gemini(api_key, history, user_input, context)
            print(reply + "\n")
        except Exception as e:
            err = str(e)
            if "API_KEY" in err.upper() or "400" in err:
                print("[ERREUR] Cle API invalide.\n")
            elif "429" in err:
                print("[ERREUR] Quota depasse, reessaie dans 1 minute.\n")
            else:
                print("[ERREUR] " + err + "\n")


if __name__ == "__main__":
    main()