#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration script for the Push Notification Generator Interface
Showcases all features of the model interface
"""

import sys
import pandas as pd
from model_interface import ModelInterface
from data_loader import load_clients, load_client_tables

def demo_interface():
    """Comprehensive demonstration of the interface"""
    print("🚀 ДЕМОНСТРАЦИЯ ИНТЕРФЕЙСА PUSH NOTIFICATION GENERATOR")
    print("=" * 60)
    
    # Step 1: Initialize Interface
    print("\n📋 Шаг 1: Инициализация интерфейса")
    interface = ModelInterface()
    
    print(f"✅ ML модель: {'Доступна' if interface.ml_model_available else 'Недоступна (используем правила)'}")
    print(f"✅ Продуктов в каталоге: {len(interface.product_classes)}")
    print(f"✅ Конфигурация: {interface.config_path}")
    
    # Step 2: Load Data
    print("\n📊 Шаг 2: Загрузка данных")
    clients = load_clients("data/clients.csv")
    tables = load_client_tables()
    
    print(f"✅ Загружено клиентов: {len(clients)}")
    print(f"✅ Клиентов с транзакциями: {len([c for c in tables.keys() if 'tx' in tables[c]])}")
    print(f"✅ Клиентов с переводами: {len([c for c in tables.keys() if 'tr' in tables[c]])}")
    
    # Step 3: Analyze Product Distribution
    print("\n📈 Шаг 3: Анализ рекомендаций (первые 10 клиентов)")
    product_count = {}
    benefit_stats = []
    
    for i in range(min(10, len(clients))):
        client_row = clients.iloc[i]
        client_id = int(client_row['client_code'])
        
        tx = tables.get(client_id, {}).get("tx", pd.DataFrame())
        tr = tables.get(client_id, {}).get("tr", pd.DataFrame())
        
        best_product, confidence = interface.get_best_product(client_row, tx, tr, "hybrid")
        
        product_count[best_product] = product_count.get(best_product, 0) + 1
        benefit_stats.append(confidence)
        
        print(f"  Клиент {client_id} ({client_row['name']}): {best_product} (уверенность: {confidence:.0f})")
    
    print(f"\n📊 Топ рекомендуемые продукты:")
    for product, count in sorted(product_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {product}: {count} клиентов")
    
    # Step 4: Detailed Client Analysis
    print("\n🔍 Шаг 4: Детальный анализ клиента")
    
    # Pick an interesting client (with good transaction data)
    best_client_idx = 0
    max_transactions = 0
    
    for i, client_row in clients.iterrows():
        client_id = int(client_row['client_code'])
        tx = tables.get(client_id, {}).get("tx", pd.DataFrame())
        if len(tx) > max_transactions:
            max_transactions = len(tx)
            best_client_idx = i
    
    client_row = clients.iloc[best_client_idx]
    client_id = int(client_row['client_code'])
    tx = tables.get(client_id, {}).get("tx", pd.DataFrame())
    tr = tables.get(client_id, {}).get("tr", pd.DataFrame())
    
    print(f"Выбран клиент: {client_row['name']} (ID: {client_id})")
    print(f"  Статус: {client_row['status']}")
    print(f"  Возраст: {client_row['age']} лет")
    print(f"  Город: {client_row['city']}")
    print(f"  Средний остаток: {client_row['avg_monthly_balance_KZT']:,.0f} ₸")
    
    if not tx.empty:
        print(f"  Транзакций: {len(tx)}")
        print(f"  Общие траты: {tx['amount'].sum():,.0f} ₸")
        top_cats = tx.groupby('category')['amount'].sum().nlargest(3)
        print(f"  Топ категории: {', '.join(top_cats.index.tolist())}")
    
    # Step 5: Compare All Methods
    print(f"\n⚖️ Шаг 5: Сравнение методов предсказания для клиента {client_id}")
    
    methods = ["rules", "hybrid"]
    if interface.ml_model_available:
        methods.append("ml")
    
    results = {}
    for method in methods:
        result = interface.generate_push_notification(
            client_row, tx, tr, 
            ollama_model=None,  # No Ollama for speed
            prediction_method=method
        )
        results[method] = result
        
        print(f"\n🔸 Метод: {method.upper()}")
        print(f"  Продукт: {result['product']}")
        print(f"  Уверенность: {result['confidence']:.3f}")
        print(f"  Выгода: {result['expected_benefit']:,.0f} ₸")
        print(f"  Push: {result['push_notification'][:100]}...")
        print(f"  Валидация: {'✅ OK' if result['validation']['ok'] else '❌ ' + ', '.join(result['validation']['issues'])}")
    
    # Step 6: Generate CSV Sample
    print(f"\n💾 Шаг 6: Генерация CSV для первых 5 клиентов")
    
    sample_results = []
    for i in range(min(5, len(clients))):
        client_row = clients.iloc[i]
        client_id = int(client_row['client_code'])
        
        tx = tables.get(client_id, {}).get("tx", pd.DataFrame())
        tr = tables.get(client_id, {}).get("tr", pd.DataFrame())
        
        result = interface.generate_push_notification(
            client_row, tx, tr, 
            ollama_model=None,
            prediction_method="hybrid"
        )
        
        sample_results.append({
            'client_code': result['client_code'],
            'product': result['product'],
            'push_notification': result['push_notification']
        })
    
    # Save sample
    sample_df = pd.DataFrame(sample_results)
    sample_path = "out/demo_sample.csv"
    sample_df.to_csv(sample_path, index=False, encoding='utf-8')
    
    print(f"✅ Образец сохранен: {sample_path}")
    print("\nОбразец данных:")
    for _, row in sample_df.iterrows():
        print(f"  Клиент {row['client_code']}: {row['product']}")
        print(f"    \"{row['push_notification'][:80]}...\"")
        print()
    
    # Step 7: Instructions for Full Usage
    print("🎯 Шаг 7: Инструкции для полного использования")
    print("\nДля обработки всех клиентов:")
    print("  python model_interface.py --output out/all_recommendations.csv")
    print("\nДля использования с Ollama:")
    print("  python model_interface.py --ollama-model mistral:7b --method hybrid")
    print("\nДля запуска веб-интерфейса:")
    print("  python web_interface.py")
    print("  # Затем откройте http://localhost:5000")
    
    print("\n" + "=" * 60)
    print("🏁 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("Интерфейс готов к использованию!")

def show_available_products():
    """Show all available products and their descriptions"""
    interface = ModelInterface()
    
    print("📦 ДОСТУПНЫЕ ПРОДУКТЫ")
    print("=" * 40)
    
    product_descriptions = {
        "Карта для путешествий": "Повышенный кешбэк на путешествия, такси, транспорт",
        "Премиальная карта": "Базовый кешбэк 2-4%, повышенный на рестораны/косметику/ювелирку",
        "Кредитная карта": "До 10% в 3 любимых категориях + 10% на онлайн-услуги",
        "Обмен валют": "Экономия на спреде, авто-покупка по целевому курсу",
        "Кредит наличными": "Быстрый доступ к финансированию, гибкие погашения",
        "Депозит Мультивалютный": "Проценты + удобство хранения валют",
        "Депозит Сберегательный": "Максимальная ставка за счёт заморозки средств",
        "Депозит Накопительный": "Повышенная ставка, пополнение да, снятие нет",
        "Инвестиции": "Нулевые/сниженные комиссии, низкий порог входа",
        "Золотые слитки": "Защитный актив/диверсификация"
    }
    
    for i, product in enumerate(interface.product_classes, 1):
        description = product_descriptions.get(product, "Описание недоступно")
        print(f"{i:2d}. {product}")
        print(f"    {description}")
        print()

def main():
    """Main function with CLI options"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            demo_interface()
        elif sys.argv[1] == "--products":
            show_available_products()
        elif sys.argv[1] == "--help":
            print("""
Push Notification Generator - Демонстрация

Использование:
  python demo.py --demo      # Полная демонстрация
  python demo.py --products  # Показать доступные продукты
  python demo.py --help      # Показать помощь

Другие команды:
  python web_interface.py    # Запустить веб-интерфейс
  python cli_test.py --test  # Быстрый тест
            """)
        else:
            print("Неизвестная команда. Используйте --help для справки")
    else:
        demo_interface()

if __name__ == "__main__":
    main()
