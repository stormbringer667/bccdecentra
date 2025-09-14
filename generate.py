# -*- coding: utf-8 -*-
import argparse, os, json, yaml
import pandas as pd
from tqdm import tqdm
from data_loader import load_clients, load_client_tables
from scoring import compute_expected_benefits, rank_products
from prompts import SYSTEM_PROMPT, build_user_prompt, format_kzt, month_of_last_full_period
from validator import validate_push, autocorrect
from ollama_client import generate_with_guardrails

OUT_DIR = "out"
INTER_DIR = "intermediate"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(INTER_DIR, exist_ok=True)

def build_behavior(tx: pd.DataFrame) -> dict:
    if tx is None or tx.empty:
        return {"top_categories": [], "taxi_count": 0, "travel_sum": 0.0}
    cat_spend = tx.groupby("category")["amount"].sum().sort_values(ascending=False).to_dict()
    top3 = list(cat_spend)[:3]
    taxi_count = int((tx["category"] == "Такси").sum()) if "category" in tx.columns else 0
    travel_sum = float(tx.loc[tx["category"].isin(["Путешествия","Такси","Отели"]), "amount"].sum())
    return {"top_categories": top3, "taxi_count": taxi_count, "travel_sum": travel_sum}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients_csv", default="data/clients.csv")
    ap.add_argument("--model", default="none", help="Ollama model name (e.g., 'mistral:7b'). Use 'none' for template generator.")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    clients = load_clients(args.clients_csv)
    tables = load_client_tables()

    rows, sft_lines, benefits_dump = [], [], []

    for _, c in tqdm(clients.iterrows(), total=len(clients)):
        cid = int(c["client_code"])
        tx = tables.get(cid, {}).get("tx", pd.DataFrame(columns=["date","category","amount","currency","client_code"]))
        tr = tables.get(cid, {}).get("tr", pd.DataFrame(columns=["date","type","direction","amount","currency","client_code"]))

        benefits, facts = compute_expected_benefits(c, tx, tr, cfg)
        ranked = rank_products(benefits)
        ref_month = month_of_last_full_period(tx["date"] if "date" in tx.columns else pd.Series([], dtype=str))

        benefits_dump.append({"client_code": cid, "ranked": ranked})

        best_product = ranked[0][0] if ranked else "Инвестиции"
        expected_benefit = benefits.get(best_product, 0.0)

        # CTA
        if best_product == "Карта для путешествий":
            cta = cfg["cta"]["travel"]
        elif best_product == "Премиальная карта":
            cta = cfg["cta"]["premium"]
        elif best_product == "Кредитная карта":
            cta = cfg["cta"]["credit"]
        elif best_product == "Обмен валют":
            cta = cfg["cta"]["fx"]
        elif best_product in {"Депозит Мультивалютный","Депозит Сберегательный","Депозит Накопительный"}:
            cta = cfg["cta"]["deposit"]
        elif best_product == "Инвестиции":
            cta = cfg["cta"]["invest"]
        elif best_product == "Золотые слитки":
            cta = cfg["cta"]["gold"]
        else:
            cta = "Посмотреть"

        behavior = build_behavior(tx)
        user_prompt = build_user_prompt(
            name=c.get("name", "Клиент"),
            status=c.get("status", ""),
            age=int(c.get("age", 0)) if pd.notna(c.get("age", None)) else 0,
            city=c.get("city", ""),
            avg_balance=float(c.get("avg_monthly_balance_KZT", 0)),
            behavior=behavior,
            product=best_product,
            expected_benefit=expected_benefit,
            cta=cta,
            ref_month=ref_month
        )

                # --- Генерация ---
        push = ""
        if args.model != "none":
            push = generate_with_guardrails(args.model, SYSTEM_PROMPT, user_prompt)

        # Фолбэк на шаблон, если из модели ничего не пришло
        if not push:
            from prompts import format_kzt, RU_MONTHS_GEN
            benefit_txt = format_kzt(expected_benefit) if expected_benefit and expected_benefit > 0 else ""
            name = c.get("name","Клиент")
            month_name = RU_MONTHS_GEN.get(ref_month, "последнем месяце")
            if best_product == "Карта для путешествий":
                push = f"{name}, в {month_name} вы часто ездите и пользуетесь такси. С картой для путешествий вернётся до {benefit_txt}. Открыть карту."
            elif best_product == "Премиальная карта":
                push = f"{name}, у вас стабильно высокий остаток и траты в ресторанах. Премиальная карта даст повышенный кешбэк и бесплатные снятия. Оформить сейчас."
            elif best_product == "Кредитная карта":
                cats = ', '.join(behavior.get('top_categories', [])[:3]) or "любимых категориях"
                push = f"{name}, ваши топ-категории — {cats}. Кредитная карта даёт до 10% кешбэка и на онлайн-сервисы. Оформить карту."
            elif best_product == "Обмен валют":
                push = f"{name}, вы часто платите в валюте. В приложении выгодный обмен без комиссии и авто-покупка по целевому курсу. Настроить обмен."
            elif best_product in {"Депозит Мультивалютный","Депозит Сберегательный","Депозит Накопительный"}:
                push = f"{name}, у вас остаются свободные средства. Разместите их на вкладе — удобно копить и получать вознаграждение. Открыть вклад."
            elif best_product == "Инвестиции":
                push = f"{name}, попробуйте инвестиции с низким порогом входа и без комиссий на старт. Открыть счёт."
            else:
                push = f"{name}, для диверсификации можно добавить золотые слитки 999,9 пробы. Посмотреть варианты."

        # Валидация и автокоррекция
        chk = validate_push(push)
        if not chk["ok"]:
            push = autocorrect(push)


        rows.append({"client_code": cid, "product": best_product, "push_notification": push})
        sft_lines.append({
            "instruction": "Сгенерируй короткое персонализированное пуш-уведомление (180–220 символов) в TOV банка, учитывая поведение клиента и пользу продукта.",
            "input": {
                "client_code": cid,
                "name": c.get("name"),
                "behavior": behavior,
                "candidate_products": ["Карта для путешествий","Премиальная карта","Кредитная карта","Обмен валют","Кредит наличными","Депозит Мультивалютный","Депозит Сберегательный","Депозит Накопительный","Инвестиции","Золотые слитки"],
                "chosen_product": best_product
            },
            "output": push
        })

    out_df = pd.DataFrame(rows).sort_values("client_code")
    out_csv = os.path.join(OUT_DIR, "push_recommendations.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    with open(os.path.join(OUT_DIR, "push_sft.jsonl"), "w", encoding="utf-8") as f:
        for ex in sft_lines:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(os.path.join(INTER_DIR, "benefits.jsonl"), "w", encoding="utf-8") as f:
        for ex in benefits_dump:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("Saved:", out_csv)
    print("Also: out/push_sft.jsonl and intermediate/benefits.jsonl")

if __name__ == "__main__":
    main()
