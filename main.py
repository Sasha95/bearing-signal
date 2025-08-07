#!/usr/bin/env python3
"""
Модуль для анализа состояния подшипника с использованием машинного обучения
"""

# Импорт модулей
import os
import torch
import pickle
from models import BearingClassifier
from dataset import load_and_preprocess_data, prepare_data_loaders
from training import train_model, evaluate_model
from prediction import predict_test_bearing
from visualization import visualize_results
from utils import set_seeds, print_summary, print_model_summary
import torch.nn as nn

def save_model_and_preprocessors(model, scaler, pca, input_size, model_dir="saved_models"):
    """Сохранение обученной модели и препроцессоров"""
    # Создание директории для сохранения моделей
    os.makedirs(model_dir, exist_ok=True)
    
    # Сохранение модели
    model_path = os.path.join(model_dir, "bearing_classifier.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'model_architecture': model.__class__.__name__
    }, model_path)
    print(f"Модель сохранена в: {model_path}")
    
    # Сохранение scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler сохранен в: {scaler_path}")
    
    # Сохранение PCA
    pca_path = os.path.join(model_dir, "pca.pkl")
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)
    print(f"PCA сохранен в: {pca_path}")
    
    # Сохранение метаданных модели
    metadata = {
        'input_size': input_size,
        'model_architecture': 'BearingClassifier',
        'description': 'Модель для классификации состояния подшипника'
    }
    metadata_path = os.path.join(model_dir, "model_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Метаданные модели сохранены в: {metadata_path}")

def create_model_visualization(model, input_size, save_dir="visualizations"):
    """Создание визуализации модели"""
    try:
        from visualtorch import graph
        
        # Создание директории для визуализаций
        os.makedirs(save_dir, exist_ok=True)
        
        # Перемещаем модель на CPU для визуализации
        model_cpu = model.cpu()
        model_cpu.eval()
        
        # Создание входного тензора для визуализации
        input_shape = (1, input_size)
        
        # Создание визуализации
        image = graph.graph_view(
            model=model_cpu,
            input_shape=input_shape,
            to_file=os.path.join(save_dir, "trained_model_architecture.png"),
            show_neurons=True,
            layer_spacing=300,
            node_size=60,
            background_fill='white',
            connector_fill='red',
            connector_width=2
        )
        
        print(f"✅ Визуализация модели сохранена в: {os.path.join(save_dir, 'trained_model_architecture.png')}")
        
        # Получение информации о модели
        print(f"📋 Информация о модели:")
        print(f"   • Размер изображения: {image.size}")
        print(f"   • Формат: {image.format}")
        print(f"   • Режим: {image.mode}")
        
        # Подсчет слоев
        layer_count = 0
        for name, module in model_cpu.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU, nn.BatchNorm1d, nn.Dropout, nn.Sigmoid)):
                layer_count += 1
        
        print(f"   • Количество слоев: {layer_count}")
            
    except ImportError:
        print("⚠️  visualtorch не установлен. Визуализация модели пропущена.")
        print("   Установите: pip install visualtorch")
    except Exception as e:
        print(f"❌ Ошибка при создании визуализации модели: {str(e)}")

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
    
    # Создание визуализации модели
    print("\nСоздание визуализации модели...")
    create_model_visualization(model, input_size)
    
    # Сохранение обученной модели и препроцессоров
    print("\nСохранение модели...")
    save_model_and_preprocessors(model, scaler, pca, input_size)
    
    # Вывод подробного самари по модели
    print_model_summary(model, input_size, train_losses, test_losses, test_accuracies,
                       accuracy, predictions, labels, mean_pred, state, state_value)
    
    # Вывод итогового резюме
    print_summary(accuracy, state, state_value, mean_pred)

if __name__ == "__main__":
    main()
