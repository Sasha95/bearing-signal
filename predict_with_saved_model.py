#!/usr/bin/env python3
"""
Скрипт для загрузки сохраненной модели и выполнения предсказаний
"""

import torch
import numpy as np
import pandas as pd
from utils import load_saved_model
from prediction import predict_test_bearing

def predict_with_saved_model(test_data_path, model_dir="saved_models"):
    """Выполнение предсказаний с использованием сохраненной модели"""
    
    print("=== Загрузка сохраненной модели ===")
    model, scaler, pca, metadata = load_saved_model(model_dir)
    
    if model is None:
        print("❌ Не удалось загрузить модель. Завершение работы.")
        return
    
    print(f"\n=== Анализ данных из файла: {test_data_path} ===")
    
    try:
        # Загрузка тестовых данных с правильным разделителем
        test_data = pd.read_csv(test_data_path, sep=';')
        print(f"✅ Данные загружены: {test_data.shape[0]} образцов, {test_data.shape[1]} признаков")
        
        # Предсказание состояния подшипника
        mean_pred, state, state_value = predict_test_bearing(model, scaler, pca, test_data)
        
        print(f"\n=== РЕЗУЛЬТАТЫ АНАЛИЗА ===")
        print(f"📊 Среднее предсказание: {mean_pred:.2f}%")
        print(f"🔍 Определенное состояние: {state}")
        print(f"📈 Процент износа: {100 - mean_pred:.2f}%")
        
        # Рекомендации
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        if mean_pred < 25:
            print(f"   • ⚠️  ПОДШИПНИК ТРЕБУЕТ НЕМЕДЛЕННОЙ ЗАМЕНЫ!")
            print(f"   • Износ составляет {100 - mean_pred:.1f}%")
            print(f"   • Рекомендуется остановить оборудование")
        elif mean_pred < 75:
            print(f"   • ⚡ ПОДШИПНИК В НОРМАЛЬНОМ СОСТОЯНИИ")
            print(f"   • Износ составляет {100 - mean_pred:.1f}%")
            print(f"   • Рекомендуется мониторинг")
        else:
            print(f"   • ✅ ПОДШИПНИК В ОТЛИЧНОМ СОСТОЯНИИ")
            print(f"   • Износ составляет {100 - mean_pred:.1f}%")
            print(f"   • Можно продолжать эксплуатацию")
            
    except FileNotFoundError:
        print(f"❌ Файл {test_data_path} не найден")
    except Exception as e:
        print(f"❌ Ошибка при обработке данных: {str(e)}")

def main():
    """Основная функция"""
    print("=== Анализ состояния подшипника с использованием сохраненной модели ===\n")
    
    # Пример использования с тестовыми данными
    test_files = [
        "data/test_bearing.csv",
        "data/test_bearing2.csv"
    ]
    
    for test_file in test_files:
        print(f"\n{'='*60}")
        predict_with_saved_model(test_file)
        print(f"{'='*60}")
    
    print(f"\n✅ Анализ завершен!")

if __name__ == "__main__":
    main()
