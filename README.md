# PushGen — персональные пуш‑уведомления (CSV → рекомендации → генерация через Ollama)

Проект решает задачу:
1) анализ 3‑месячного поведения вымышленных клиентов (транзакции + переводы + профиль),
2) расчёт ожидаемой выгоды по каталогу продуктов,
3) выбор наилучшего продукта,
4) генерация персонализированного пуш‑уведомления в корректном тоне (TOV),
5) экспорт `client_code,product,push_notification` в CSV.

## Быстрый старт

```bash
# 1) Питон-зависимости
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Данные положить в data/:
#   clients.csv
#   client_<id>_transactions_3m.csv
#   client_<id>_transfers_3m.csv

# 3) Пробный прогон (без Ollama, шаблонный генератор TOV)
python generate.py --model none

# 4) Прогон с Ollama
python generate.py --model "mistral:7b"

# Результаты:
#   out/push_recommendations.csv
#   out/push_sft.jsonl (датасет для SFT)
```

## Структура

- `config.yaml` — ставки, лимиты, карты категорий и CTA.
- `data_loader.py` — загрузка клиентов/транзакций/переводов из CSV.
- `scoring.py` — расчёт выгод по продуктам и ранжирование.
- `prompts.py` — системный и пользовательский промпт, форматирование валют и месяцев.
- `validator.py` — проверка правил TOV (длина, капс, «вы», 1 CTA и т.д.), авто‑коррекция.
- `ollama_client.py` — вызов `ollama run`, повтор с self‑critique при нарушениях.
- `generate.py` — основной оркестратор: считывает, считает, вызывает LLM/шаблон, валидирует и экспортирует CSV/JSONL.

### Тренировка (опционально)
- `training/axolotl_push.yaml` — пример конфига LoRA‑дообучения.
- `training/Modelfile` — пример обёртки модели в Ollama с системным промптом.
- `training/README_finetune.md` — как собрать датасет и обучить LoRA.

## Формат входных данных

**clients.csv**
```
client_code,name,status,age,city,avg_monthly_balance_KZT
1,Айгерим,Студент,21,Алматы,350000
...
```

**client_<id>_transactions_3m.csv**
```
date,category,amount,currency,client_code
2025-06-02,Такси,1200,KZT,1
...
```

**client_<id>_transfers_3m.csv**
```
date,type,direction,amount,currency,client_code
2025-05-10,fx_buy,out,100000,KZT,1
...
```

## Оценка качества (из брифа)
- Метрика 1: точность продукта — сохраняем Top‑4 в `intermediate/benefits.jsonl` для последующей проверки.
- Метрика 2: качество пуша — `validator.py` проверяет правила, можно агрегировать отчёт.

