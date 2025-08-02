#!/usr/bin/env python3
"""
Модуль для анализа состояния подшипника с использованием машинного обучения
"""

# Импорт модулей
from models import BearingClassifier
from dataset import load_and_preprocess_data, prepare_data_loaders
from training import train_model, evaluate_model
from prediction import predict_test_bearing
from visualization import visualize_results
from utils import set_seeds, print_summary, print_model_summary

def main():
    """Основная функция"""
    print("=== Анализ состояния подшипника с использованием машинного обучения ===\n")
    
    # Установка seed для воспроизводимости
    set_seeds()
    
    # Загрузка и предобработка данных
    X, y = load_and_preprocess_data()
    
    # Подготовка данных для обучения
    train_loader, test_loader, scaler, pca = prepare_data_loaders(X, y)
    
    # Создание модели с правильным количеством входных признаков
    input_size = train_loader.dataset.features.shape[1]  # Используем размерность после PCA
    model = BearingClassifier(input_size)
    print(f"\nСоздана модель с {input_size} входными признаками")
    
    # Обучение модели
    model, train_losses, test_losses, test_accuracies = train_model(
        model, train_loader, test_loader
    )
    
    # Оценка модели
    accuracy, predictions, labels = evaluate_model(model, test_loader, scaler)
    
    # Предсказание для тестового подшипника
    mean_pred, state, state_value = predict_test_bearing(model, scaler, pca)
    
    # Визуализация результатов
    print("\nСоздание визуализации...")
    visualize_results(train_losses, test_losses, test_accuracies, predictions, labels)
    
    # Вывод подробного самари по модели
    print_model_summary(model, input_size, train_losses, test_losses, test_accuracies,
                       accuracy, predictions, labels, mean_pred, state, state_value)
    
    # Вывод итогового резюме
    print_summary(accuracy, state, state_value, mean_pred)

if __name__ == "__main__":
    main()
