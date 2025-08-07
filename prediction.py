import torch
import pandas as pd
import numpy as np
from dataset import process_numeric_data
from training import get_device

def predict_test_bearing(model, scaler, pca, test_data=None):
    """Предсказание состояния тестового подшипника"""
    if test_data is None:
        print("\nЗагрузка тестовых данных...")
        test_data = pd.read_csv('data/test_bearing2.csv', sep=';')
    else:
        print(f"\nИспользование переданных данных размером: {test_data.shape}")
    
    print(f"Размер тестовых данных: {test_data.shape}")
    
    # Обработка NaN значений
    test_data = test_data.fillna(0)
    
    # Удаление столбца timestamp и строки с частотами
    test_data = test_data.iloc[1:, 1:]  # Убираем первую строку (частоты) и первый столбец (timestamp)
    
    # Обработка числовых данных
    test_data = process_numeric_data(test_data)
    
    # Проверяем, совпадает ли количество столбцов с обучающими данными
    expected_columns = scaler.n_features_in_
    actual_columns = test_data.shape[1]
    
    print(f"Ожидаемое количество признаков: {expected_columns}")
    print(f"Фактическое количество признаков: {actual_columns}")
    
    if actual_columns > expected_columns:
        # Берем только первые столбцы
        test_data = test_data.iloc[:, :expected_columns]
        print(f"Обрезано до {expected_columns} столбцов")
    elif actual_columns < expected_columns:
        # Добавляем нулевые столбцы
        missing_columns = expected_columns - actual_columns
        for i in range(missing_columns):
            test_data[f'extra_{i}'] = 0
        print(f"Добавлено {missing_columns} нулевых столбцов")
    
    # Приведение названий столбцов к числовому формату
    test_data.columns = [str(i) for i in range(len(test_data.columns))]
    
    # Нормализация тестовых данных (отключаем проверку названий)
    test_features_scaled = scaler.transform(test_data.values)
    test_features_scaled = np.nan_to_num(test_features_scaled, nan=0.0)
    
    # Применение PCA
    test_features_pca = pca.transform(test_features_scaled)
    
    device = get_device()
    model.eval()
    
    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_features_pca).to(device)
        predictions = model(test_tensor).squeeze()
        predictions = predictions * 100  # Масштабирование в проценты
        
        # Вычисление среднего предсказания
        mean_prediction = predictions.mean().item()
        
        # Определение состояния
        if mean_prediction < 25:
            state = "Плохой"
            state_value = 0.0
        elif mean_prediction < 75:
            state = "Нормальный"
            state_value = 50.0
        else:
            state = "Новый"
            state_value = 100.0
    
    print(f"\nРезультат предсказания:")
    print(f"Среднее предсказание: {mean_prediction:.2f}%")
    print(f"Состояние подшипника: {state} ({state_value}%)")
    print(f"Процент износа подшипника: {100 - mean_prediction:.2f}%")
    
    return mean_prediction, state, state_value 