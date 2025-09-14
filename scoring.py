# -*- coding: utf-8 -*-
import yaml, pandas as pd
from typing import Dict, Tuple, List

def compute_expected_benefits(client_row: pd.Series,
                              tx: pd.DataFrame,
                              tr: pd.DataFrame,
                              cfg: dict) -> Tuple[Dict[str, float], Dict[str, dict]]:
    rates = cfg["rates"]
    cats = cfg["categories"]

    benefits: Dict[str, float] = {}
    facts: Dict[str, dict] = {}

    # amounts in KZT (упрощение — считаем, что currency=KZT; иначе можно добавить конверсию)
    tx = tx.copy()
    if "currency" in tx.columns:
        # тут можно вставить конверсию, если будет таблица курсов в cfg
        pass
    tx["amount_kzt"] = tx["amount"].clip(lower=0)

    total_spend = tx["amount_kzt"].sum()
    cat_spend = tx.groupby("category")["amount_kzt"].sum().to_dict()

    # travel card
    travel_spend = sum(cat_spend.get(c, 0) for c in cats["travel"])
    travel_cashback = rates["travel_cashback"] * travel_spend
    benefits["Карта для путешествий"] = travel_cashback
    facts["Карта для путешествий"] = {"travel_spend": travel_spend}

    # premium card
    avg_bal = float(client_row.get("avg_monthly_balance_KZT", 0))
    base = rates["premium"]["base_default"]
    if 1_000_000 <= avg_bal < 6_000_000:
        base = rates["premium"]["base_mid"]
    elif avg_bal >= 6_000_000:
        base = rates["premium"]["base_high"]
    boosted_spend = sum(cat_spend.get(c, 0) for c in cats["premium_boosted"])
    boosted_cb = rates["premium"]["boosted_categories_rate"] * boosted_spend
    other_spend = max(total_spend - boosted_spend, 0)
    other_cb = base * other_spend
    premium_cb = min(boosted_cb + other_cb, rates["premium"]["cashback_cap_month"])
    benefits["Премиальная карта"] = premium_cb
    facts["Премиальная карта"] = {"avg_balance": avg_bal, "boosted_spend": boosted_spend, "total_spend": total_spend, "rate": base}

    # credit card
    items = sorted(cat_spend.items(), key=lambda x: x[1], reverse=True)
    top3 = items[:3]
    top3_spend = sum(v for _, v in top3)
    top3_cats = {c for c,_ in top3}
    online_extra = sum(cat_spend.get(c, 0) for c in set(cats["online"]) - top3_cats)
    cc_cb = rates["credit_card"]["fav_rate"] * (top3_spend + online_extra)
    benefits["Кредитная карта"] = cc_cb
    facts["Кредитная карта"] = {"top3": [c for c,_ in top3], "top3_spend": top3_spend}

    # FX
    fx_ops = 0
    fx_vol = 0.0
    if tr is not None and not tr.empty:
        fx_ops = tr["type"].isin(["fx_buy","fx_sell","deposit_fx_topup_out","deposit_fx_withdraw_in"]).sum()
        fx_vol = tr.loc[tr["type"].isin(["fx_buy","fx_sell"]), "amount"].sum()
    fx_saving = rates["fx_saving_rate"] * fx_vol
    benefits["Обмен валют"] = fx_saving
    facts["Обмен валют"] = {"fx_volume_est": fx_vol, "fx_ops": int(fx_ops)}

    # cash loan (0 по умолчанию без сигнала)
    benefits["Кредит наличными"] = 0.0
    facts["Кредит наличными"] = {}

    # deposits (на 3 месяца)
    dep_multi = rates["deposits"]["multi"] * avg_bal / 12.0 * 3
    dep_save = rates["deposits"]["save"] * avg_bal / 12.0 * 3
    dep_acc  = rates["deposits"]["accum"] * avg_bal / 12.0 * 3
    benefits["Депозит Мультивалютный"] = dep_multi
    benefits["Депозит Сберегательный"] = dep_save
    benefits["Депозит Накопительный"]  = dep_acc
    facts["Депозит Мультивалютный"] = {"avg_balance": avg_bal}
    facts["Депозит Сберегательный"] = {"avg_balance": avg_bal}
    facts["Депозит Накопительный"]  = {"avg_balance": avg_bal}

    # invest / gold — сигнал присутствия
    invest_signal = 1 if (tr is not None and not tr.empty and tr["type"].isin(["invest_in","invest_out"]).any()) else 0
    gold_signal   = 1 if (tr is not None and not tr.empty and tr["type"].isin(["gold_buy_out","gold_sell_in"]).any()) else 0
    benefits["Инвестиции"] = 1000 * invest_signal
    benefits["Золотые слитки"] = 1000 * gold_signal
    facts["Инвестиции"] = {"signal": invest_signal}
    facts["Золотые слитки"] = {"signal": gold_signal}

    return benefits, facts

def rank_products(benefits: Dict[str, float]):
    return sorted(benefits.items(), key=lambda x: x[1], reverse=True)
