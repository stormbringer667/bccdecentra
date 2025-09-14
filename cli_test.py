#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple CLI script to test the model interface
"""

import sys
import os
from model_interface import ModelInterface

def test_single_client():
    """Test with a single client"""
    print("🚀 Тестирование интерфейса модели\n")
    
    # Initialize interface
    print("Инициализация интерфейса модели...")
    interface = ModelInterface()
    print(f"✅ Интерфейс инициализирован (ML модель: {'доступна' if interface.ml_model_available else 'недоступна'})\n")
    
    # Load data
    from data_loader import load_clients, load_client_tables
    print("Загрузка данных клиентов...")
    clients = load_clients("data/clients.csv")
    tables = load_client_tables()
    print(f"✅ Загружено {len(clients)} клиентов\n")
    
    # Process first client
    client_row = clients.iloc[0]
    client_id = int(client_row['client_code'])
    
    print(f"Обработка клиента {client_id}: {client_row['name']}")
    print(f"  Статус: {client_row['status']}")
    print(f"  Возраст: {client_row['age']}")
    print(f"  Город: {client_row['city']}")
    print(f"  Средний остаток: {client_row['avg_monthly_balance_KZT']:,.0f} ₸\n")
    
    # Get client data
    tx = tables.get(client_id, {}).get("tx", None)
    tr = tables.get(client_id, {}).get("tr", None)
    
    if tx is not None and not tx.empty:
        print(f"  Транзакций: {len(tx)}")
        print(f"  Общая сумма трат: {tx['amount'].sum():,.0f} ₸")
        print(f"  Топ категории: {', '.join(tx.groupby('category')['amount'].sum().nlargest(3).index.tolist())}")
    else:
        print("  Нет данных о транзакциях")
    
    if tr is not None and not tr.empty:
        print(f"  Переводов: {len(tr)}")
    else:
        print("  Нет данных о переводах")
    
    print("\n" + "="*60)
    
    # Test different methods
    methods = ["rules", "hybrid"]
    if interface.ml_model_available:
        methods.append("ml")
    
    for method in methods:
        print(f"\n🔍 Метод: {method.upper()}")
        
        try:
            result = interface.generate_push_notification(
                client_row, tx, tr, 
                ollama_model=None,  # No Ollama for quick test
                prediction_method=method
            )
            
            print(f"✅ Рекомендуемый продукт: {result['product']}")
            print(f"📊 Уверенность: {result['confidence']:.3f}")
            print(f"💰 Ожидаемая выгода: {result['expected_benefit']:,.0f} ₸")
            print(f"📱 Push-уведомление:")
            print(f"   {result['push_notification']}")
            print(f"✅ Валидация: {'OK' if result['validation']['ok'] else 'Есть проблемы'}")
            
            if not result['validation']['ok']:
                print(f"⚠️  Проблемы: {', '.join(result['validation']['issues'])}")
                
        except Exception as e:
            print(f"❌ Ошибка в методе {method}: {e}")
    
    print("\n" + "="*60)
    print("🏁 Тест завершен!")

def main():
    """Main CLI function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            test_single_client()
        elif sys.argv[1] == "--help":
            print("""
Использование:
  python cli_test.py --test     # Тест с одним клиентом
  python cli_test.py --help     # Показать помощь
  
Для запуска веб-интерфейса:
  python web_interface.py
  
Для обработки всех клиентов:
  python model_interface.py --output out/results.csv
            """)
        else:
            print("Неизвестная команда. Используйте --help для справки")
    else:
        print("Запуск базового теста...")
        test_single_client()

if __name__ == "__main__":
    main()
