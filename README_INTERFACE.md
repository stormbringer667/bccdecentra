# Push Notification Generator - Model Interface

Интерфейс для генерации персонализированных push-уведомлений на основе ML модели и Ollama LLM.

## 🚀 Возможности

- **Гибридный подход**: Комбинирует ML модель с правилами бизнес-логики
- **Интеграция с Ollama**: Поддержка различных LLM моделей для генерации текста
- **Веб-интерфейс**: Удобный веб-интерфейс для тестирования
- **Пакетная обработка**: Обработка всех клиентов одной командой
- **Валидация**: Автоматическая проверка соответствия ТЗ

## 📋 Требования

```bash
pip install -r requirements.txt
```

## 🛠️ Компоненты системы

### 1. ModelInterface (`model_interface.py`)
Основной класс, который:
- Загружает ML модель из `model/model.pkl`
- Интегрируется с Ollama клиентом
- Предоставляет методы предсказания: `ml`, `rules`, `hybrid`
- Генерирует персонализированные push-уведомления

### 2. Web Interface (`web_interface.py`)
Flask веб-приложение с возможностями:
- Просмотр списка клиентов
- Генерация рекомендаций для отдельных клиентов
- Пакетная обработка всех клиентов
- Скачивание результатов в CSV

### 3. CLI Test (`cli_test.py`)
Простой скрипт для быстрого тестирования

## 🚀 Использование

### Веб-интерфейс (рекомендуется)

```bash
python web_interface.py
```

Откройте http://localhost:5000 в браузере.

### CLI интерфейс

```bash
# Быстрый тест
python cli_test.py --test

# Обработка всех клиентов
python model_interface.py --output out/results.csv

# С использованием Ollama модели
python model_interface.py --ollama-model mistral:7b --method hybrid --output out/results.csv

# Обработка одного клиента
python model_interface.py --client-id 1 --ollama-model mistral:7b
```

### Использование в коде

```python
from model_interface import ModelInterface
from data_loader import load_clients, load_client_tables

# Инициализация
interface = ModelInterface()

# Загрузка данных
clients = load_clients("data/clients.csv")
tables = load_client_tables()

# Получение данных клиента
client_row = clients.iloc[0]
client_id = int(client_row['client_code'])
tx = tables.get(client_id, {}).get("tx", pd.DataFrame())
tr = tables.get(client_id, {}).get("tr", pd.DataFrame())

# Генерация рекомендации
result = interface.generate_push_notification(
    client_row, tx, tr, 
    ollama_model="mistral:7b",  # опционально
    prediction_method="hybrid"  # ml, rules, или hybrid
)

print(f"Продукт: {result['product']}")
print(f"Push: {result['push_notification']}")
```

## 📊 Методы предсказания

### 1. `rules` - Правила бизнес-логики
- Использует только скоринг из `scoring.py`
- Быстро и предсказуемо
- Основано на ТЗ

### 2. `ml` - Машинное обучение
- Использует модель из `model/model.pkl`
- Требует совместимую версию sklearn
- Может давать более точные предсказания

### 3. `hybrid` - Гибридный (рекомендуется)
- Комбинирует ML и правила
- Если ML недоступна, использует правила
- Наилучший баланс точности и надежности

## 🤖 Интеграция с Ollama

Для использования LLM генерации установите Ollama:

```bash
# Установка Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Загрузка модели
ollama pull mistral:7b
# или
ollama pull llama2:7b
```

Поддерживаемые модели:
- `mistral:7b` - быстрая и качественная
- `llama2:7b` - классическая
- `codellama:7b` - для программистов
- любые другие Ollama модели

## 📁 Структура файлов

```
pushgen/
├── model_interface.py      # Основной интерфейс
├── web_interface.py        # Веб-интерфейс
├── cli_test.py            # CLI тест
├── templates/
│   └── index.html         # Веб-шаблон
├── model/
│   ├── model.pkl          # ML модель
│   └── model_meta.json    # Метаданные модели
├── data/                  # Данные клиентов
├── out/                   # Результаты
└── requirements.txt       # Зависимости
```

## 🔧 Конфигурация

Настройки в `config.yaml`:
- Ставки кешбэка для продуктов
- Категории транзакций
- CTA для каждого продукта

## 📝 Формат вывода

CSV файл с колонками:
- `client_code` - ID клиента
- `product` - Рекомендуемый продукт
- `push_notification` - Текст уведомления

## 🧪 Валидация

Автоматическая проверка по критериям из ТЗ:
- ✅ Длина 180-220 символов
- ✅ Обращение на "вы"
- ✅ Максимум 1 восклицательный знак
- ✅ Наличие правильного CTA
- ✅ Отсутствие КАПСА

## 🐛 Устранение неисправностей

### ML модель не загружается
```
Warning: Could not load ML model: STACK_GLOBAL requires str
```
**Решение**: Система автоматически переключится на правила. Для исправления нужно пересохранить модель в совместимой версии sklearn.

### Нет данных о клиентах
Убедитесь что файл `data/clients.csv` существует и правильно отформатирован.

### Ollama недоступна
Система автоматически переключится на шаблоны. Установите Ollama для полной функциональности.

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Сделайте коммит изменений
4. Отправьте pull request

## 📄 Лицензия

MIT License - смотрите файл LICENSE для деталей.
