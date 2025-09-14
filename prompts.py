# -*- coding: utf-8 -*-
from datetime import datetime
import pandas as pd

RU_MONTHS_GEN = {
    1: "январе", 2: "феврале", 3: "марте", 4: "апреле", 5: "мае", 6: "июне",
    7: "июле", 8: "августе", 9: "сентябре", 10: "октябре", 11: "ноябре", 12: "декабре"
}

def format_kzt(amount: float) -> str:
    amount = 0 if pd.isna(amount) else float(amount)
    s = f"{amount:,.0f}".replace(",", "X").replace(".", ",").replace("X", " ")
    return f"{s} ₸"

def month_of_last_full_period(dates: pd.Series) -> int:
    if dates is None or dates.empty:
        dt = datetime.now()
        m = dt.month - 1 or 12
        return m
    months = pd.to_datetime(dates, errors="coerce").dt.month.dropna().astype(int)
    if months.empty:
        dt = datetime.now()
        m = dt.month - 1 or 12
        return m
    return int(months.mode().iloc[0])

SYSTEM_PROMPT = """Вы — редактор банка. Пишите короткие пуш-уведомления (180–220 символов), на «вы», без капса, максимум 1 «!».
Без воды и давления; одна мысль и один CTA из списка: «Открыть», «Настроить», «Посмотреть», «Оформить сейчас», «Оформить карту», «Открыть вклад», «Открыть счёт».
Форматируйте валюту так: 27 400 ₸. Допустим 0–1 эмодзи по смыслу.
"""

def build_user_prompt(name: str,
                      status: str,
                      age: int,
                      city: str,
                      avg_balance: float,
                      behavior: dict,
                      product: str,
                      expected_benefit: float,
                      cta: str,
                      ref_month: int) -> str:
    month_name = RU_MONTHS_GEN.get(ref_month, "последнем месяце")
    benefit = format_kzt(expected_benefit) if expected_benefit and expected_benefit > 0 else "—"
    top_categories = ", ".join(behavior.get("top_categories", [])[:3]) if behavior else ""
    taxi_count = behavior.get("taxi_count", 0)
    travel_sum = behavior.get("travel_sum", 0.0)
    travel_sum_fmt = format_kzt(travel_sum) if travel_sum else "0 ₸"
    ab = format_kzt(avg_balance)
    return f"""Дано:
- Клиент: {name}, статус: {status}, возраст: {age}, город: {city}, средний остаток: {ab}
- Поведение за 3 мес: топ категории — {top_categories}; такси: {taxi_count}; траты на поездки: {travel_sum_fmt}
- Выбранный продукт: {product}
- Ожидаемая выгода: {benefit}
- Месяц: {month_name}
- CTA: {cta}

Задача:
Сгенерируйте 1 пуш-сообщение: персональный контекст → польза продукта → CTA. 180–220 символов, обращение на «вы», без капса.
"""
