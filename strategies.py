"""
Legendary Investors Strategies — Trading Bot v21
Author: Theo

Integre les techniques des 7 meilleurs investisseurs de l'histoire :
  1. Warren Buffett  — Value + Quality
  2. William O'Neil  — CAN SLIM momentum
  3. Paul Tudor Jones — Trend following strict
  4. George Soros    — Momentum + retournement
  5. Jesse Livermore — Pivotal points + volume
  6. Ray Dalio       — Risk parity
  7. Stan Druckenmiller — Asymetrie risque/reward
"""

import pandas as pd
import numpy as np


# ============================================================
# 1. WARREN BUFFETT — Value & Quality
# ============================================================
def buffett_score(df):
    """
    Buffett achete des entreprises de qualite a prix raisonnable.
    Sur les ETF/actions on approxime avec :
    - Tendance long terme stable (prix > MA200 depuis longtemps)
    - Faible volatilite (ATR bas = business stable)
    - Momentum positif a long terme (Mom_20 > 0)
    - Prix pas suracheté (loin du top des 52 semaines)
    Score : 0 a 4
    """
    score  = 0
    checks = {}

    row = df.iloc[-1]

    # 1. Tendance long terme : prix > MA200 depuis au moins 20 jours
    above_ma200 = df["Close"] > df["MA200"]
    consistency = above_ma200.tail(20).sum()
    c1 = consistency >= 15  # 15/20 jours au-dessus MA200
    checks["Tendance LT stable (15/20j > MA200)"] = c1
    if c1: score += 1

    # 2. Faible volatilite relative (ATR < 2% du prix)
    atr_pct = float(row["ATR"]) / float(row["Close"]) * 100
    c2 = atr_pct < 2.0
    checks["Volatilite faible (ATR < 2%)"] = c2
    if c2: score += 1

    # 3. Momentum long terme positif (20 jours)
    mom_20 = float(df["Close"].pct_change(20).iloc[-1])
    c3 = mom_20 > 0.02  # +2% sur 20 jours
    checks["Momentum LT positif (+2% sur 20j)"] = c3
    if c3: score += 1

    # 4. Prix pas en zone de surachat extreme (pas dans top 5% du range 52W)
    high_52w = df["High"].tail(252).max()
    low_52w  = df["Low"].tail(252).min()
    range_52w = high_52w - low_52w
    pos_52w   = (float(row["Close"]) - low_52w) / (range_52w + 1e-9)
    c4 = pos_52w < 0.95  # pas dans les 5% du haut
    checks["Pas en surachat extremes (52W)"] = c4
    if c4: score += 1

    return score, checks, "Buffett (Value/Quality)"


# ============================================================
# 2. WILLIAM O'NEIL — CAN SLIM
# ============================================================
def oneil_score(df):
    """
    O'Neil cherche des actions avec fort momentum et volume croissant.
    CAN SLIM adapte aux ETF/actions :
    C — Current earnings (approxime par momentum recent)
    A — Annual growth (trend long terme)
    N — New highs (prix proche du plus haut)
    S — Supply/demand (volume croissant)
    L — Leader (RSI fort)
    I — Institutional (vol_ratio eleve)
    M — Market direction (MA50 hausse)
    Score : 0 a 7
    """
    score  = 0
    checks = {}
    row    = df.iloc[-1]

    # C — Momentum recent fort (>3% sur 5 jours)
    mom_5 = float(df["Close"].pct_change(5).iloc[-1])
    c1 = mom_5 > 0.03
    checks["C - Momentum 5j > 3%"] = c1
    if c1: score += 1

    # A — Tendance annuelle positive
    mom_50 = float(df["Close"].pct_change(50).iloc[-1])
    c2 = mom_50 > 0.10  # +10% sur 50 jours
    checks["A - Trend 50j > 10%"] = c2
    if c2: score += 1

    # N — Prix proche des nouveaux hauts (top 20% du range 50 jours)
    high_50 = df["High"].tail(50).max()
    low_50  = df["Low"].tail(50).min()
    pos_50  = (float(row["Close"]) - low_50) / (high_50 - low_50 + 1e-9)
    c3 = pos_50 > 0.80
    checks["N - Prix dans top 20% (50j)"] = c3
    if c3: score += 1

    # S — Volume en hausse (Vol_ratio > 1.2)
    c4 = float(row["Vol_ratio"]) > 1.2
    checks["S - Volume > 1.2x moyenne"] = c4
    if c4: score += 1

    # L — RSI fort (55-75 = zone leader)
    c5 = 55 < float(row["RSI"]) < 75
    checks["L - RSI leader (55-75)"] = c5
    if c5: score += 1

    # I — Demande institutionnelle (vol > moyenne 3 derniers jours)
    avg_vol_3 = df["Volume"].tail(3).mean()
    avg_vol_20 = df["Volume"].tail(20).mean()
    c6 = avg_vol_3 > avg_vol_20 * 1.1
    checks["I - Vol 3j > Vol 20j (+10%)"] = c6
    if c6: score += 1

    # M — Direction marche (MA50 en hausse)
    c7 = float(row["MA50_slope"]) > 0
    checks["M - MA50 en hausse"] = c7
    if c7: score += 1

    return score, checks, "O'Neil (CAN SLIM)"


# ============================================================
# 3. PAUL TUDOR JONES — Trend Following
# ============================================================
def tudor_jones_score(df):
    """
    Tudor Jones suit les tendances avec une discipline de fer.
    Principes cles :
    - Ne jamais aller contre la tendance principale
    - Pyramider sur les gagnants
    - Couper les pertes rapidement
    Score adapte : 0 a 5
    """
    score  = 0
    checks = {}
    row    = df.iloc[-1]

    # 1. Tendance principale haussiere (prix > MA200)
    c1 = float(row["Close"]) > float(row["MA200"])
    checks["Tendance principale haussiere"] = c1
    if c1: score += 1

    # 2. Acceleration de la tendance (MA50 > MA200 et ecart croissant)
    ma50_ma200_gap = (float(row["MA50"]) - float(row["MA200"])) / float(row["MA200"])
    c2 = ma50_ma200_gap > 0.02  # MA50 au moins 2% au-dessus MA200
    checks["MA50 > MA200 + 2% (acceleration)"] = c2
    if c2: score += 1

    # 3. Momentum court terme confirme (MA10 > MA20)
    c3 = float(row["MA10"]) > float(row["MA20"])
    checks["MA10 > MA20 (momentum CT)"] = c3
    if c3: score += 1

    # 4. Pas de divergence RSI (RSI > 50 = tendance haussiere confirmee)
    c4 = float(row["RSI"]) > 50
    checks["RSI > 50 (pas de divergence)"] = c4
    if c4: score += 1

    # 5. MACD confirme la tendance
    c5 = float(row["MACD"]) > 0 and float(row["MACD_Hist"]) > 0
    checks["MACD positif et histogramme > 0"] = c5
    if c5: score += 1

    return score, checks, "Tudor Jones (Trend)"


# ============================================================
# 4. GEORGE SOROS — Reflexivite & Momentum
# ============================================================
def soros_score(df):
    """
    Soros identifie les moments ou le marche accelere dans un sens
    et capitalise dessus avant le retournement.
    Score : 0 a 4
    """
    score  = 0
    checks = {}
    row    = df.iloc[-1]

    # 1. Acceleration du momentum (mom_5 > mom recente moyenne)
    mom_5     = float(df["Close"].pct_change(5).iloc[-1])
    mom_5_avg = float(df["Close"].pct_change(5).tail(20).mean())
    c1 = mom_5 > mom_5_avg * 1.5 and mom_5 > 0
    checks["Acceleration momentum (1.5x moyenne)"] = c1
    if c1: score += 1

    # 2. Volume croissant (confirme le mouvement)
    vol_now  = float(df["Volume"].tail(3).mean())
    vol_past = float(df["Volume"].tail(20).mean())
    c2 = vol_now > vol_past * 1.15
    checks["Volume croissant (+15%)"] = c2
    if c2: score += 1

    # 3. Bollinger Bands : prix dans la moitie superieure
    c3 = float(row["BB_pos"]) > 0.5
    checks["Prix dans moitie sup Bollinger"] = c3
    if c3: score += 1

    # 4. Pas encore en zone de renversement (RSI < 75)
    c4 = float(row["RSI"]) < 75
    checks["RSI < 75 (pas encore retournement)"] = c4
    if c4: score += 1

    return score, checks, "Soros (Momentum)"


# ============================================================
# 5. JESSE LIVERMORE — Pivotal Points
# ============================================================
def livermore_score(df):
    """
    Livermore entrait sur les cassures de niveaux pivots avec volume.
    Logique :
    - Identifier les resistance/support cles
    - Entrer sur cassure avec volume fort
    - Confirmer avec momentum
    Score : 0 a 4
    """
    score  = 0
    checks = {}
    row    = df.iloc[-1]

    # 1. Cassure au-dessus de la resistance (plus haut 20 jours)
    high_20  = float(df["High"].tail(21).iloc[:-1].max())
    close_now = float(row["Close"])
    c1 = close_now > high_20 * 0.99  # proche ou au-dessus du plus haut 20j
    checks["Cassure resistance 20j"] = c1
    if c1: score += 1

    # 2. Volume fort sur la cassure
    c2 = float(row["Vol_ratio"]) > 1.3
    checks["Volume fort sur cassure (>1.3x)"] = c2
    if c2: score += 1

    # 3. Pas de faux breakout (prix stable les 2 derniers jours)
    change_2d = abs(float(df["Close"].pct_change(2).iloc[-1]))
    c3 = change_2d < 0.05  # pas de mouvement trop violent
    checks["Pas de faux breakout (<5% en 2j)"] = c3
    if c3: score += 1

    # 4. Tendance principale favorable
    c4 = float(row["Close"]) > float(row["MA50"])
    checks["Prix > MA50 (tendance favorable)"] = c4
    if c4: score += 1

    return score, checks, "Livermore (Pivots)"


# ============================================================
# 6. RAY DALIO — Risk Parity
# ============================================================
def dalio_score(df):
    """
    Dalio equilibre le risque — il entre quand le risque est faible
    et la recompense potentielle est elevee.
    Score : 0 a 4
    """
    score  = 0
    checks = {}
    row    = df.iloc[-1]

    # 1. Faible risque : volatilite basse (ATR < moyenne)
    atr_now = float(row["ATR"])
    atr_avg = float(df["ATR"].tail(50).mean())
    c1 = atr_now < atr_avg * 1.1
    checks["Volatilite normale (ATR < moy+10%)"] = c1
    if c1: score += 1

    # 2. Tendance claire (MA10 > MA20 > MA50)
    c2 = (float(row["MA10"]) > float(row["MA20"]) and
          float(row["MA20"]) > float(row["MA50"]))
    checks["Tendance claire MA10>MA20>MA50"] = c2
    if c2: score += 1

    # 3. Momentum equilibre (ni trop fort ni trop faible)
    mom_5 = abs(float(df["Close"].pct_change(5).iloc[-1]))
    c3 = 0.01 < mom_5 < 0.08  # entre 1% et 8%
    checks["Momentum equilibre (1-8% sur 5j)"] = c3
    if c3: score += 1

    # 4. RSI en zone equilibree (45-65)
    c4 = 45 < float(row["RSI"]) < 65
    checks["RSI equilibre (45-65)"] = c4
    if c4: score += 1

    return score, checks, "Dalio (Risk Parity)"


# ============================================================
# 7. STAN DRUCKENMILLER — Asymetrie R/R
# ============================================================
def druckenmiller_score(df, atr_stop=1.5, atr_tp=3.0):
    """
    Druckenmiller cherche des setups avec une asymetrie maximale :
    petit risque, grand gain potentiel.
    Il concentre ses bets sur les meilleures opportunites.
    Score : 0 a 4
    """
    score  = 0
    checks = {}
    row    = df.iloc[-1]

    price = float(row["Close"])
    atr   = float(row["ATR"])

    # 1. Ratio R/R favorable (TP/SL >= 2x)
    rr = atr_tp / atr_stop
    c1 = rr >= 2.0
    checks["RR >= 2x (TP/SL asymetrique)"] = c1
    if c1: score += 1

    # 2. Prix proche du support (bon point d'entree)
    dist_ma20 = float(row["Dist_MA20"])
    c2 = 0.0 <= dist_ma20 <= 0.015  # pullback sur MA20
    checks["Pullback sur support MA20"] = c2
    if c2: score += 1

    # 3. Tendance forte (MA50 bien au-dessus MA200)
    trend_strength = (float(row["MA50"]) - float(row["MA200"])) / float(row["MA200"])
    c3 = trend_strength > 0.01
    checks["Tendance forte MA50/MA200 > 1%"] = c3
    if c3: score += 1

    # 4. Momentum positif confirme
    mom_10 = float(df["Close"].pct_change(10).iloc[-1])
    c4 = mom_10 > 0.01
    checks["Momentum 10j positif > 1%"] = c4
    if c4: score += 1

    return score, checks, "Druckenmiller (Asymetrie)"


# ============================================================
# SCORE COMPOSITE — Toutes strategies combinées
# ============================================================
def composite_score(df, atr_stop=1.5, atr_tp=3.0):
    """
    Calcule le score composite de toutes les strategies.
    Retourne un score normalise entre 0 et 100.
    """
    strategies = [
        buffett_score(df),
        oneil_score(df),
        tudor_jones_score(df),
        soros_score(df),
        livermore_score(df),
        dalio_score(df),
        druckenmiller_score(df, atr_stop, atr_tp),
    ]

    total_score = 0
    total_max   = 0
    details     = {}

    max_scores = [4, 7, 5, 4, 4, 4, 4]  # max par strategie

    for i, (score, checks, name) in enumerate(strategies):
        total_score += score
        total_max   += max_scores[i]
        details[name] = {
            "score":    score,
            "max":      max_scores[i],
            "pct":      round(score / max_scores[i] * 100),
            "checks":   checks,
        }

    normalized = round(total_score / total_max * 100)

    return normalized, total_score, total_max, details


def print_composite(normalized, total_score, total_max, details):
    """Affiche le rapport detaille de toutes les strategies."""
    print("\n" + "="*55)
    print("  LEGENDARY INVESTORS COMPOSITE SCORE")
    print("="*55)

    bar_len = 30
    for name, d in details.items():
        filled  = int(d["pct"] / 100 * bar_len)
        bar     = "█" * filled + "░" * (bar_len - filled)
        score_str = str(d["score"]) + "/" + str(d["max"])
        print("\n  " + name)
        print("  [" + bar + "]  " + str(d["pct"]) + "%  (" + score_str + ")")
        for check_name, ok in d["checks"].items():
            icon = "  ✓" if ok else "  ✗"
            print(icon + "  " + check_name)

    print("\n" + "="*55)
    total_str = str(total_score) + "/" + str(total_max)
    print("  SCORE COMPOSITE : " + str(normalized) + "/100  (" + total_str + ")")

    if normalized >= 75:
        verdict = "TRES FORT — Les legendes sont unanimes"
    elif normalized >= 60:
        verdict = "FORT — Majorite des legendes d'accord"
    elif normalized >= 45:
        verdict = "NEUTRE — Avis partages"
    elif normalized >= 30:
        verdict = "FAIBLE — Peu de legendes d'accord"
    else:
        verdict = "TRES FAIBLE — Eviter"

    print("  VERDICT       : " + verdict)
    print("="*55 + "\n")