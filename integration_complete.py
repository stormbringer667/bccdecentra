#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Integration Example
Shows how the Ollama + ML model interface works together
"""

import os
import pandas as pd
from model_interface import ModelInterface
from data_loader import load_clients, load_client_tables

def create_complete_example():
    """Create a complete working example"""
    print("🎯 СОЗДАНИЕ ПОЛНОГО ПРИМЕРА ИНТЕГРАЦИИ")
    print("=" * 50)
    
    # Initialize
    interface = ModelInterface()
    clients = load_clients("data/clients.csv")
    tables = load_client_tables()
    
    # Process a few diverse clients
    interesting_clients = [1, 2, 7, 15, 25]  # Different statuses and patterns
    
    results = []
    
    for client_id in interesting_clients:
        client_row = clients[clients['client_code'] == client_id].iloc[0]
        tx = tables.get(client_id, {}).get("tx", pd.DataFrame())
        tr = tables.get(client_id, {}).get("tr", pd.DataFrame())
        
        # Generate recommendation (without Ollama for speed)
        result = interface.generate_push_notification(
            client_row, tx, tr, 
            ollama_model=None,
            prediction_method="hybrid"
        )
        
        results.append(result)
        
        print(f"\n👤 Клиент {client_id}: {client_row['name']}")
        print(f"   Статус: {client_row['status']}")
        print(f"   Продукт: {result['product']}")
        print(f"   Выгода: {result['expected_benefit']:,.0f} ₸")
        print(f"   Push: {result['push_notification']}")
        print(f"   Валидация: {'✅' if result['validation']['ok'] else '❌'}")
        
        if not result['validation']['ok']:
            print(f"   Проблемы: {', '.join(result['validation']['issues'])}")
    
    # Create output CSV
    output_data = []
    for result in results:
        output_data.append({
            'client_code': result['client_code'],
            'product': result['product'],
            'push_notification': result['push_notification']
        })
    
    output_df = pd.DataFrame(output_data)
    output_path = "out/integration_example.csv"
    os.makedirs("out", exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n💾 Результаты сохранены: {output_path}")
    
    # Show final CSV format
    print(f"\n📋 ФИНАЛЬНЫЙ CSV (формат для ТЗ):")
    print("client_code,product,push_notification")
    for _, row in output_df.iterrows():
        # Truncate for display
        push_short = row['push_notification'][:60] + "..." if len(row['push_notification']) > 60 else row['push_notification']
        print(f"{row['client_code']},{row['product']},\"{push_short}\"")
    
    return output_path

def show_integration_architecture():
    """Show how the integration works"""
    print("\n🏗️ АРХИТЕКТУРА ИНТЕГРАЦИИ")
    print("=" * 40)
    
    print("""
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client Data   │    │   ML Model       │    │   Ollama LLM    │
│                 │    │                  │    │                 │
│ • Transactions  │    │ • model.pkl      │    │ • mistral:7b    │
│ • Transfers     │    │ • Features       │    │ • llama2:7b     │
│ • Profile       │    │ • Predictions    │    │ • codellama     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    Model Interface       │
                    │                          │
                    │ • Hybrid predictions     │
                    │ • Rule-based scoring     │
                    │ • Template generation    │
                    │ • Validation system      │
                    └─────────────┬────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      Output              │
                    │                          │
                    │ • Product recommendation │
                    │ • Push notification      │
                    │ • Validation results     │
                    │ • CSV format             │
                    └──────────────────────────┘
    """)
    
    print("\n🔄 ПРОЦЕСС ИНТЕГРАЦИИ:")
    print("1. 📊 Загрузка данных клиента (транзакции, переводы, профиль)")
    print("2. 🤖 ML модель анализирует паттерны и предсказывает продукт")
    print("3. 📈 Правила бизнес-логики вычисляют ожидаемую выгоду")
    print("4. 🎯 Гибридный алгоритм выбирает лучший продукт")
    print("5. 🦙 Ollama LLM генерирует персонализированный текст")
    print("6. 📝 Шаблон как fallback если LLM недоступна")
    print("7. ✅ Валидация по требованиям ТЗ")
    print("8. 💾 Вывод в формате CSV")

def main():
    """Main function"""
    print("🚀 ПОЛНАЯ ИНТЕГРАЦИЯ OLLAMA + ML MODEL")
    print("=" * 50)
    
    # Show architecture
    show_integration_architecture()
    
    # Create example
    output_path = create_complete_example()
    
    print(f"\n🎉 ИНТЕГРАЦИЯ ЗАВЕРШЕНА!")
    print(f"✅ Создан рабочий интерфейс объединяющий:")
    print(f"   • ML модель из model/model.pkl")
    print(f"   • Ollama клиент для LLM генерации")
    print(f"   • Правила бизнес-логики из ТЗ")
    print(f"   • Валидацию и автокоррекцию")
    print(f"   • Веб-интерфейс для удобного использования")
    
    print(f"\n📁 Созданные файлы:")
    print(f"   • model_interface.py - Основной интерфейс")
    print(f"   • web_interface.py - Веб-интерфейс") 
    print(f"   • templates/index.html - Веб-шаблон")
    print(f"   • cli_test.py - CLI тестирование")
    print(f"   • demo.py - Демонстрация возможностей")
    print(f"   • {output_path} - Пример результата")
    print(f"   • README_INTERFACE.md - Документация")
    
    print(f"\n🚀 Для использования:")
    print(f"   python web_interface.py          # Веб-интерфейс")
    print(f"   python model_interface.py        # CLI обработка")
    print(f"   python demo.py --demo            # Полная демонстрация")

if __name__ == "__main__":
    main()
