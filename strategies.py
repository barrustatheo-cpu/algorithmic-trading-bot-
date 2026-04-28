"""
Legendary Investors Strategies v2 - Human Language
Author: Theo

Les 7 legendes de la finance avec donnees fondamentales reelles
et rapports en langage naturel.
"""

import pandas as pd
import numpy as np
import yfinance as yf


# ============================================================
# DONNEES FONDAMENTALES REELLES
# ============================================================
def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        return {
            "pe_ratio":       info.get("trailingPE",         None),
            "forward_pe":     info.get("forwardPE",          None),
            "peg_ratio":      info.get("pegRatio",           None),
            "pb_ratio":       info.get("priceToBook",        None),
            "ps_ratio":       info.get("priceToSalesTrailing12Months", None),
            "eps_growth":     info.get("earningsGrowth",     None),
            "revenue_growth": info.get("revenueGrowth",      None),
            "profit_margin":  info.get("profitMargins",      None),
            "roe":            info.get("returnOnEquity",     None),
            "debt_equity":    info.get("debtToEquity",       None),
            "current_ratio":  info.get("currentRatio",       None),
            "beta":           info.get("beta",               None),
            "market_cap":     info.get("marketCap",          None),
            "sector":         info.get("sector",             "N/A"),
            "name":           info.get("longName",           ticker),
            "52w_high":       info.get("fiftyTwoWeekHigh",   None),
            "52w_low":        info.get("fiftyTwoWeekLow",    None),
            "analyst_target": info.get("targetMeanPrice",    None),
            "dividend_yield": info.get("dividendYield",      None),
        }
    except Exception:
        return {}


# ============================================================
# 1. WARREN BUFFETT — Value & Quality
# ============================================================
def buffett_score(df, fundamentals):
    score  = 0
    checks = {}
    row    = df.iloc[-1]

    above_ma200 = (df["Close"] > df["MA200"]).tail(20).sum()
    c1 = above_ma200 >= 15
    checks["Tendance LT stable (15/20j > MA200)"] = c1
    if c1: score += 1

    atr_pct = float(row["ATR"]) / float(row["Close"]) * 100
    c2 = atr_pct < 2.5
    checks["Volatilite maitrisee (ATR < 2.5%)"] = c2
    if c2: score += 1

    pe = fundamentals.get("pe_ratio")
    if pe and pe > 0:
        c3 = pe < 35
        checks["P/E raisonnable (< 35) : " + str(round(pe, 1))] = c3
        if c3: score += 1
    else:
        c3 = float(row["Close"]) > float(row["MA200"])
        checks["Prix > MA200 (proxy valeur)"] = c3
        if c3: score += 1

    roe = fundamentals.get("roe")
    if roe and roe > 0:
        c4 = roe > 0.15
        checks["ROE fort > 15% : " + str(round(roe*100, 1)) + "%"] = c4
        if c4: score += 1
    else:
        mom_20 = float(df["Close"].pct_change(20).iloc[-1])
        c4 = mom_20 > 0.02
        checks["Momentum LT > 2% (proxy ROE)"] = c4
        if c4: score += 1

    margin = fundamentals.get("profit_margin")
    if margin and margin > 0:
        c5 = margin > 0.10
        checks["Marge nette > 10% : " + str(round(margin*100, 1)) + "%"] = c5
        if c5: score += 1
    else:
        high_52 = df["High"].tail(252).max()
        low_52  = df["Low"].tail(252).min()
        pos = (float(row["Close"]) - low_52) / (high_52 - low_52 + 1e-9)
        c5 = pos < 0.95
        checks["Pas en surachat extreme (52W)"] = c5
        if c5: score += 1

    return score, checks, "Warren Buffett"


# ============================================================
# 2. WILLIAM O'NEIL — CAN SLIM
# ============================================================
def oneil_score(df, fundamentals):
    score  = 0
    checks = {}
    row    = df.iloc[-1]

    eps_growth = fundamentals.get("eps_growth")
    if eps_growth is not None:
        c1 = eps_growth > 0.20
        checks["C - Croissance EPS > 20% : " + str(round(eps_growth*100, 1)) + "%"] = c1
        if c1: score += 1
    else:
        mom_5 = float(df["Close"].pct_change(5).iloc[-1])
        c1 = mom_5 > 0.02
        checks["C - Momentum 5j > 2% (proxy EPS)"] = c1
        if c1: score += 1

    rev_growth = fundamentals.get("revenue_growth")
    if rev_growth is not None:
        c2 = rev_growth > 0.15
        checks["A - Croissance revenus > 15% : " + str(round(rev_growth*100, 1)) + "%"] = c2
        if c2: score += 1
    else:
        mom_50 = float(df["Close"].pct_change(50).iloc[-1])
        c2 = mom_50 > 0.10
        checks["A - Trend 50j > 10% (proxy revenus)"] = c2
        if c2: score += 1

    high_50 = df["High"].tail(50).max()
    low_50  = df["Low"].tail(50).min()
    pos_50  = (float(row["Close"]) - low_50) / (high_50 - low_50 + 1e-9)
    c3 = pos_50 > 0.75
    checks["N - Dans top 25% du range 50j"] = c3
    if c3: score += 1

    c4 = float(row["Vol_ratio"]) > 1.2
    checks["S - Volume 1.2x la moyenne"] = c4
    if c4: score += 1

    c5 = 55 < float(row["RSI"]) < 80
    checks["L - RSI leader (55-80) : " + str(round(float(row["RSI"]), 1))] = c5
    if c5: score += 1

    c6 = float(row["MA50_slope"]) > 0 and float(row["Close"]) > float(row["MA200"])
    checks["M - Marche haussier (MA50 up + > MA200)"] = c6
    if c6: score += 1

    avg_3  = df["Volume"].tail(3).mean()
    avg_20 = df["Volume"].tail(20).mean()
    c7 = avg_3 > avg_20 * 1.1
    checks["I - Volume institutionnel croissant"] = c7
    if c7: score += 1

    return score, checks, "William O'Neil"


# ============================================================
# 3. PAUL TUDOR JONES — Trend Following
# ============================================================
def tudor_jones_score(df, fundamentals):
    score  = 0
    checks = {}
    row    = df.iloc[-1]

    c1 = float(row["Close"]) > float(row["MA200"])
    checks["Tendance principale haussiere (> MA200)"] = c1
    if c1: score += 1

    ma50_ma200 = (float(row["MA50"]) - float(row["MA200"])) / float(row["MA200"])
    c2 = ma50_ma200 > 0.02
    checks["Acceleration : MA50 > MA200 + 2%"] = c2
    if c2: score += 1

    c3 = float(row["MA10"]) > float(row["MA20"])
    checks["Confirmation CT : MA10 > MA20"] = c3
    if c3: score += 1

    c4 = float(row["RSI"]) > 50
    checks["RSI > 50 (force haussiere)"] = c4
    if c4: score += 1

    c5 = float(row["MACD"]) > 0 and float(row["MACD_Hist"]) > 0
    checks["MACD positif et histogramme > 0"] = c5
    if c5: score += 1

    beta = fundamentals.get("beta")
    if beta:
        c6 = 0.8 < beta < 2.5
        checks["Beta optimal (0.8-2.5) : " + str(round(beta, 2))] = c6
        if c6: score += 1
    else:
        atr_pct = float(row["ATR"]) / float(row["Close"]) * 100
        c6 = atr_pct > 0.5
        checks["Volatilite suffisante pour trader"] = c6
        if c6: score += 1

    return score, checks, "Paul Tudor Jones"


# ============================================================
# 4. GEORGE SOROS — Reflexivite
# ============================================================
def soros_score(df, fundamentals):
    score  = 0
    checks = {}
    row    = df.iloc[-1]

    mom_5     = float(df["Close"].pct_change(5).iloc[-1])
    mom_5_avg = float(df["Close"].pct_change(5).tail(20).mean())
    c1 = mom_5 > mom_5_avg * 1.3 and mom_5 > 0
    checks["Acceleration momentum (1.3x moyenne)"] = c1
    if c1: score += 1

    vol_3  = float(df["Volume"].tail(3).mean())
    vol_20 = float(df["Volume"].tail(20).mean())
    c2 = vol_3 > vol_20 * 1.1
    checks["Volume croissant (+10%)"] = c2
    if c2: score += 1

    c3 = float(row["BB_pos"]) > 0.5
    checks["Prix dans moitie superieure Bollinger"] = c3
    if c3: score += 1

    c4 = float(row["RSI"]) < 78
    checks["RSI < 78 (pas encore retournement)"] = c4
    if c4: score += 1

    peg = fundamentals.get("peg_ratio")
    if peg and peg > 0:
        c5 = peg < 2.0
        checks["PEG < 2.0 : " + str(round(peg, 2))] = c5
        if c5: score += 1
    else:
        mom_20 = float(df["Close"].pct_change(20).iloc[-1])
        c5 = mom_20 > 0
        checks["Momentum 20j positif (macro favorable)"] = c5
        if c5: score += 1

    return score, checks, "George Soros"


# ============================================================
# 5. JESSE LIVERMORE — Pivotal Points
# ============================================================
def livermore_score(df, fundamentals):
    score  = 0
    checks = {}
    row    = df.iloc[-1]

    high_20   = float(df["High"].tail(21).iloc[:-1].max())
    close_now = float(row["Close"])
    c1 = close_now >= high_20 * 0.98
    checks["Cassure ou proche resistance 20j"] = c1
    if c1: score += 1

    c2 = float(row["Vol_ratio"]) > 1.25
    checks["Volume fort sur le mouvement (>1.25x)"] = c2
    if c2: score += 1

    change_2d = abs(float(df["Close"].pct_change(2).iloc[-1]))
    c3 = change_2d < 0.06
    checks["Mouvement ordonne (<6% en 2 jours)"] = c3
    if c3: score += 1

    c4 = float(row["Close"]) > float(row["MA50"])
    checks["Prix > MA50 (tendance favorable)"] = c4
    if c4: score += 1

    high_52 = fundamentals.get("52w_high")
    if high_52:
        dist_52w = (high_52 - close_now) / high_52
        c5 = dist_52w < 0.10
        checks["Dans 10% du plus haut 52W"] = c5
        if c5: score += 1
    else:
        high_52_calc = df["High"].tail(252).max()
        dist = (high_52_calc - close_now) / high_52_calc
        c5 = dist < 0.15
        checks["Dans 15% du plus haut 52W (calcule)"] = c5
        if c5: score += 1

    return score, checks, "Jesse Livermore"


# ============================================================
# 6. RAY DALIO — Risk Parity
# ============================================================
def dalio_score(df, fundamentals):
    score  = 0
    checks = {}
    row    = df.iloc[-1]

    atr_now = float(row["ATR"])
    atr_avg = float(df["ATR"].tail(50).mean())
    c1 = atr_now < atr_avg * 1.15
    checks["Volatilite dans la norme (ATR < moy+15%)"] = c1
    if c1: score += 1

    c2 = (float(row["MA10"]) > float(row["MA20"]) and
          float(row["MA20"]) > float(row["MA50"]))
    checks["Alignement parfait MA10>MA20>MA50"] = c2
    if c2: score += 1

    mom_5 = abs(float(df["Close"].pct_change(5).iloc[-1]))
    c3 = 0.005 < mom_5 < 0.10
    checks["Momentum equilibre (0.5-10% sur 5j)"] = c3
    if c3: score += 1

    c4 = 42 < float(row["RSI"]) < 68
    checks["RSI equilibre (42-68) : " + str(round(float(row["RSI"]), 1))] = c4
    if c4: score += 1

    de = fundamentals.get("debt_equity")
    if de is not None:
        c5 = de < 150
        checks["Dette/Equity acceptable (<150) : " + str(round(de, 1))] = c5
        if c5: score += 1
    else:
        c5 = float(row["Close"]) > float(row["MA200"])
        checks["Regime haussier confirme"] = c5
        if c5: score += 1

    return score, checks, "Ray Dalio"


# ============================================================
# 7. STAN DRUCKENMILLER — Asymetrie
# ============================================================
def druckenmiller_score(df, fundamentals, atr_stop=1.5, atr_tp=3.0):
    score  = 0
    checks = {}
    row    = df.iloc[-1]

    rr = atr_tp / atr_stop
    c1 = rr >= 2.0
    checks["RR >= 2x : " + str(round(rr, 1)) + "x"] = c1
    if c1: score += 1

    dist_ma20 = float(row["Dist_MA20"])
    c2 = -0.005 <= dist_ma20 <= 0.015
    checks["Pullback sur support MA20"] = c2
    if c2: score += 1

    trend = (float(row["MA50"]) - float(row["MA200"])) / float(row["MA200"])
    c3 = trend > 0.01
    checks["Tendance forte MA50/MA200 > 1%"] = c3
    if c3: score += 1

    mom_10 = float(df["Close"].pct_change(10).iloc[-1])
    c4 = mom_10 > 0.005
    checks["Momentum 10j positif > 0.5%"] = c4
    if c4: score += 1

    target = fundamentals.get("analyst_target")
    price  = float(row["Close"])
    if target and target > 0:
        upside = (target - price) / price
        c5 = upside > 0.05
        checks["Upside analyste > 5% : +" + str(round(upside*100, 1)) + "%"] = c5
        if c5: score += 1
    else:
        bb_pos = float(row["BB_pos"])
        c5 = bb_pos < 0.75
        checks["Pas en zone de surachat Bollinger"] = c5
        if c5: score += 1

    return score, checks, "Stan Druckenmiller"


# ============================================================
# SCORE COMPOSITE
# ============================================================
def composite_score(df, atr_stop=1.5, atr_tp=3.0, ticker=None):
    fundamentals = get_fundamentals(ticker) if ticker else {}

    strategies = [
        buffett_score(df, fundamentals),
        oneil_score(df, fundamentals),
        tudor_jones_score(df, fundamentals),
        soros_score(df, fundamentals),
        livermore_score(df, fundamentals),
        dalio_score(df, fundamentals),
        druckenmiller_score(df, fundamentals, atr_stop, atr_tp),
    ]

    max_scores  = [5, 7, 6, 5, 5, 5, 5]
    total_score = 0
    total_max   = 0
    details     = {}

    for i, (score, checks, name) in enumerate(strategies):
        total_score += score
        total_max   += max_scores[i]
        details[name] = {
            "score":  score,
            "max":    max_scores[i],
            "pct":    round(score / max_scores[i] * 100),
            "checks": checks,
        }

    normalized = round(total_score / total_max * 100)
    return normalized, total_score, total_max, details


# ============================================================
# RAPPORT EN LANGAGE HUMAIN
# ============================================================
def human_report(normalized, total_score, total_max, details, ticker=""):
    lines = []
    name  = ticker.upper() if ticker else "cette action"

    lines.append("")
    lines.append("=" * 60)
    lines.append("  ANALYSE — " + name)
    lines.append("=" * 60)

    if normalized >= 80:
        intro = ("Franchement, " + name + " coche presque toutes les cases. "
                 "Les 7 legendes seraient d'accord — c'est un setup exceptionnel.")
    elif normalized >= 65:
        intro = (name + " presente un setup solide. La majorite des grands "
                 "investisseurs seraient a l'aise pour entrer sur ce niveau.")
    elif normalized >= 50:
        intro = (name + " est mitige. Il y a des arguments pour et contre. "
                 "Certaines legendes diraient oui, d'autres attendraient.")
    elif normalized >= 35:
        intro = ("Honnêtement, " + name + " n'est pas au top en ce moment. "
                 "La plupart des legendes attendraient avant d'entrer.")
    else:
        intro = (name + " ne passe pas les filtres. Aucune legende serieuse "
                 "n'entrerait sur ce niveau. Mieux vaut rester en cash.")

    lines.append("")
    lines.append("  " + intro)

    opinions = {
        "Warren Buffett": {
            "high": "Buffett serait interesse. L'entreprise montre les caracteristiques qu'il aime : qualite, stabilite et prix raisonnable.",
            "mid":  "Buffett hesiterait. Certains criteres de qualite sont la mais pas tous. Il attendrait probablement.",
            "low":  "Buffett passerait son chemin. Ce n'est pas le type d'entreprise de qualite qu'il recherche."
        },
        "William O'Neil": {
            "high": "O'Neil adorerait ce setup. Fort momentum, volume croissant, nouveaux hauts — exactement son style CAN SLIM.",
            "mid":  "O'Neil verrait du potentiel mais manque de conviction. Il voudrait voir plus de force avant d'agir.",
            "low":  "O'Neil dirait clairement non. Pas assez de momentum ni de croissance pour son systeme CAN SLIM."
        },
        "Paul Tudor Jones": {
            "high": "Tudor Jones suivrait la tendance sans hesiter. La direction est claire et il ne va jamais contre une tendance aussi propre.",
            "mid":  "Tudor Jones attendrait une confirmation. La tendance existe mais manque de clarte.",
            "low":  "Tudor Jones ne toucherait pas a ca. Pas de tendance claire, trop risque."
        },
        "George Soros": {
            "high": "Soros sentirait le momentum s'accelerer. Il prendrait une position significative en anticipant la continuation.",
            "mid":  "Soros serait prudent. Le momentum existe mais pas assez fort pour miser gros.",
            "low":  "Soros attendrait un signal plus clair avant d'agir."
        },
        "Jesse Livermore": {
            "high": "Livermore verrait une cassure propre avec du volume. C'est exactement le type de pivot qu'il chassait toute sa carriere.",
            "mid":  "Livermore chercherait un meilleur point d'entree. La cassure n'est pas encore assez nette.",
            "low":  "Livermore ne verrait pas de pivot clair. Trop de resistance au-dessus."
        },
        "Ray Dalio": {
            "high": "Dalio validerait le risk/reward. Le risque est maitrise et la recompense potentielle est attrayante.",
            "mid":  "Dalio reduirait la taille de position. Trop d'incertitude pour miser plein.",
            "low":  "Dalio resterait en cash. Le profil risque/recompense n'est pas assez favorable."
        },
        "Stan Druckenmiller": {
            "high": "Druckenmiller miserait gros. L'asymetrie est parfaite — peu a perdre, beaucoup a gagner.",
            "mid":  "Druckenmiller prendrait une petite position pour tater le terrain.",
            "low":  "Druckenmiller attendrait. Pas assez d'asymetrie pour justifier un trade."
        }
    }

    lines.append("")
    lines.append("  CE QUE DISENT LES LEGENDES :")
    lines.append("")

    for name_inv, d in details.items():
        pct = d["pct"]
        if pct >= 65:
            opinion = opinions.get(name_inv, {}).get("high", "")
            icon    = "[OK]"
        elif pct >= 40:
            opinion = opinions.get(name_inv, {}).get("mid", "")
            icon    = "[~~]"
        else:
            opinion = opinions.get(name_inv, {}).get("low", "")
            icon    = "[NO]"

        score_str = str(d["score"]) + "/" + str(d["max"])
        lines.append("  " + icon + " " + name_inv +
                     " (" + str(pct) + "% — " + score_str + ")")
        lines.append("      " + opinion)
        lines.append("")

    strong = [n for n, d in details.items() if d["pct"] >= 65]
    weak   = [n for n, d in details.items() if d["pct"] < 40]

    if strong:
        lines.append("  POUR : " + ", ".join(strong))
        lines.append("")
    if weak:
        lines.append("  CONTRE : " + ", ".join(weak))
        lines.append("")

    lines.append("=" * 60)
    lines.append("  SCORE COMPOSITE : " + str(normalized) + "/100")

    if normalized >= 75:
        verdict = "SETUP EXCEPTIONNEL — Entre si le gate est valide"
        conseil = "Position : pleine (Kelly max)"
    elif normalized >= 60:
        verdict = "BON SETUP — Conditions favorables"
        conseil = "Position : normale"
    elif normalized >= 45:
        verdict = "SETUP CORRECT — Attendre confirmation"
        conseil = "Position : reduite"
    elif normalized >= 30:
        verdict = "SETUP FAIBLE — Patience recommandee"
        conseil = "Position : minimale ou cash"
    else:
        verdict = "EVITER — Rester en cash"
        conseil = "Position : 0%"

    lines.append("  VERDICT : " + verdict)
    lines.append("  CONSEIL : " + conseil)
    lines.append("=" * 60)
    lines.append("")

    return "\n".join(lines)


def print_composite(normalized, total_score, total_max, details, ticker=""):
    print(human_report(normalized, total_score, total_max, details, ticker))