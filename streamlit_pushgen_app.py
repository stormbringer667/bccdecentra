#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
import streamlit as st

#  Нормализация статуса
STATUS_MAP = {
    "зп": "Зарплатный клиент",
    "зарплатный клиент": "Зарплатный клиент",
    "salary": "Зарплатный клиент",
    "ст": "Студент",
    "студент": "Студент",
    "прем": "Премиальный клиент",
    "премиальный клиент": "Премиальный клиент",
    "premium": "Премиальный клиент",
    "std": "Стандартный клиент",
    "стандартный клиент": "Стандартный клиент",
}
def normalize_status(s: str) -> str:
    if s is None:
        return "Стандартный клиент"
    key = str(s).strip().lower()
    return STATUS_MAP.get(key, s if s in STATUS_MAP.values() else "Стандартный клиент")

def first_name(full_name: str) -> str:
    return full_name.split()[0] if isinstance(full_name, str) and full_name else "Клиент"

# Оформление страницы 
st.set_page_config(page_title="CSV → Признаки + LLM-заглушка", layout="wide")
st.title("CSV  + LLM-заглушка)")
st.caption("Загрузите CSV с полями: client_code, name, status, city, date, category, amount, currency.")

#   Sidebar
with st.sidebar:
    st.header("Импорт CSV")
    sep = st.text_input("Разделитель CSV", value=",")
    thousands = st.text_input("Разделитель тысяч (если есть)", value="")
    decimal = st.text_input("Десятичный разделитель", value=".")
    st.markdown("---")
    st.header("LLM ")
    use_llm_stub = st.checkbox("Включить LLM", value=False)
    default_llm_template = (
        "{first_name}, персональная рекомендация. Ваш средний расход ~{m_total_spend_fmt}/мес. "
        "Основные категории: {top_categories}. [LLM_STUB]"
    )
    llm_template = st.text_area("Шаблон сообщения ", value=default_llm_template, height=100)
    st.caption("Доступные поля: {first_name}, {status_norm}, {city}, {m_total_spend_fmt}, {top_categories}")
    st.markdown("---")
    features_filename = st.text_input("Имя файла (фичи)", value="features.csv")
    messages_filename = st.text_input("Имя файла (LLM)", value="messages_stub.csv")

# ------ Загрузка CSV 
required_cols = ["client_code", "name", "status", "city", "date", "category", "amount", "currency"]
st.subheader("Загрузите исходный CSV")
up = st.file_uploader("dataset.csv", type=["csv"])

if up is not None:
    # Чтение и валидация
    df = pd.read_csv(up, sep=sep or ",", thousands=(thousands or None), decimal=(decimal or "."))
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"В CSV отсутствуют колонки: {missing}")
        st.stop()

    # Приведение типов
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["client_code", "date", "category", "amount"])

    st.caption("Первые строки входного CSV")
    st.dataframe(df.head(20), use_container_width=True)

    
    df["ym"] = df["date"].dt.to_period("M")

    months_per_client = df.groupby("client_code")["ym"].nunique().rename("months").to_frame()

    spend_by = df.groupby(["client_code", "category"]).agg(spend_kzt=("amount", "sum")).reset_index()
    pivot_spend = spend_by.pivot(index="client_code", columns="category", values="spend_kzt").fillna(0.0)

    total_spend = df.groupby("client_code").agg(total_spend_kzt=("amount", "sum"))

    profiles = (
        df.sort_values(["client_code", "date"])
          .drop_duplicates("client_code")
          .loc[:, ["client_code", "name", "status", "city"]]
          .set_index("client_code")
          .copy()
    )
    profiles["status_norm"] = profiles["status"].apply(normalize_status)

    feat = profiles.join([pivot_spend, total_spend, months_per_client], how="left").fillna(0.0)

    if "months" in feat.columns:
        cat_cols_raw = [c for c in feat.columns if c not in ["name", "status", "status_norm", "city", "total_spend_kzt", "months"]]
        for c in cat_cols_raw:
            feat[f"m_{c}"] = feat[c] / feat["months"].where(feat["months"] > 0, 1)
        feat["m_total_spend_kzt"] = feat["total_spend_kzt"] / feat["months"].where(feat["months"] > 0, 1)
    else:
        cat_cols_raw = []

    base_cols = ["name", "status", "status_norm", "city", "months", "total_spend_kzt", "m_total_spend_kzt"]
    m_cols = sorted([c for c in feat.columns if c.startswith("m_")])
    cat_cols = sorted([c for c in feat.columns if c not in base_cols + m_cols and c not in ["name", "status", "status_norm", "city", "months", "total_spend_kzt"]])
    ordered = [c for c in base_cols if c in feat.columns] + cat_cols + m_cols

    out_df = feat[ordered].reset_index()  # вернуть client_code
    st.success("Фичи готовы. Ниже предпросмотр.")
    st.dataframe(out_df.head(60), use_container_width=True)

    # ---------- 6) Выгрузка фичей ----------
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Скачать фичи (CSV)",
        data=csv_bytes,
        file_name=features_filename or "features.csv",
        mime="text/csv",
    )

    # 7) LLM-заглушка
    def top_categories_from_row(row) -> str:
        
        m_pairs = []
        for col in row.index:
            if col.startswith("m_") and isinstance(row[col], (int, float)):
                cat_name = col[2:]  # убрать 'm_'
                m_pairs.append((cat_name, float(row[col])))
        m_pairs.sort(key=lambda x: x[1], reverse=True)
        top = [name for name, val in m_pairs if val > 0][:3]
        return ", ".join(top) if top else "—"

    def make_llm_stub(row, template: str) -> str:
        
        ctx = {
            "first_name": first_name(row.get("name", "Клиент")),
            "status_norm": row.get("status_norm", "Стандартный клиент"),
            "city": row.get("city", ""),
            "m_total_spend_fmt": f"{float(row.get('m_total_spend_kzt', 0.0)):.0f} ₸",
            "top_categories": top_categories_from_row(row),
        }
        try:
            return template.format(**ctx).strip()
        except Exception:
            # На случай если пользователь добавил неизвестные плейсхолдеры
            return (
                f"{ctx['first_name']}, персональная рекомендация. "
                f"Средний расход: {ctx['m_total_spend_fmt']}. "
                f"Категории: {ctx['top_categories']}. [LLM_STUB]"
            )

    if use_llm_stub:
        st.markdown("### Генерация сообщений ")
        # Берём по клиенту первую запись из out_df (он уже уникален на client_code)
        msgs = out_df[["client_code", "name", "status_norm", "city", "m_ыtotal_spend_kzt"] + [c for c in out_df.columns if c.startswith("m_")]].copy()
        msgs["llm_stub"] = msgs.apply(lambda r: make_llm_stub(r, llm_template), axis=1)

        st.dataframe(msgs[["client_code", "llm_stub"]].head(60), use_container_width=True)

        msgs_csv = msgs[["client_code", "llm_stub"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Скачать LLM (CSV)",
            data=msgs_csv,
            file_name=messages_filename or "messages_stub.csv",
            mime="text/csv",
        )

else:
    st.info("Загрузите CSV с транзакциями.")
