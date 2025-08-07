#!/usr/bin/env python3
"""
Скрипт для визуализации архитектуры нейронной сети
"""

import torch
import torch.nn as nn
from visualtorch import graph
from models import BearingClassifier
from utils import load_saved_model
import os

def visualize_model_architecture(input_size=100, save_path="model_architecture.png"):
    """Визуализация архитектуры модели"""
    print("=== Визуализация архитектуры модели ===")
    
    # Создание модели
    model = BearingClassifier(input_size)
    model.eval()  # Переводим в режим оценки
    
    # Создание входного тензора для визуализации
    input_shape = (1, input_size)
    
    print(f"📊 Архитектура модели:")
    print(f"   • Входной размер: {input_size}")
    print(f"   • Скрытые слои: 256 → 128 → 64 → 32 → 1")
    print(f"   • Активации: ReLU + BatchNorm + Dropout")
    print(f"   • Выход: Sigmoid")
    print(f"   • Общее количество параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    # Создание визуализации
    try:
        # Создание графа архитектуры
        image = graph.graph_view(
            model=model,
            input_shape=input_shape,
            to_file=save_path,
            show_neurons=True,
            layer_spacing=300,
            node_size=60,
            background_fill='white',
            connector_fill='blue',
            connector_width=2
        )
        
        print(f"✅ Визуализация сохранена в: {save_path}")
        
        # Получение информации о модели
        print(f"\n📋 Информация о модели:")
        print(f"   • Размер изображения: {image.size}")
        print(f"   • Формат: {image.format}")
        print(f"   • Режим: {image.mode}")
        
        # Подсчет слоев
        layer_count = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU, nn.BatchNorm1d, nn.Dropout, nn.Sigmoid)):
                layer_count += 1
        
        print(f"   • Количество слоев: {layer_count}")
            
    except Exception as e:
        print(f"❌ Ошибка при создании визуализации: {str(e)}")

def visualize_saved_model(model_dir="saved_models", save_path="saved_model_architecture.png"):
    """Визуализация сохраненной модели"""
    print("=== Визуализация сохраненной модели ===")
    
    # Загрузка сохраненной модели
    model, scaler, pca, metadata = load_saved_model(model_dir)
    
    if model is None:
        print("❌ Не удалось загрузить модель для визуализации")
        return
    
    # Получение входного размера из метаданных
    input_size = metadata.get('input_size', 100)
    input_shape = (1, input_size)
    
    # Перемещаем модель на CPU для визуализации
    model = model.cpu()
    model.eval()
    
    print(f"📊 Архитектура сохраненной модели:")
    print(f"   • Входной размер: {input_size}")
    print(f"   • Архитектура: {metadata.get('model_architecture', 'Unknown')}")
    print(f"   • Описание: {metadata.get('description', 'No description')}")
    print(f"   • Общее количество параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    # Создание визуализации
    try:
        # Создание графа архитектуры
        image = graph.graph_view(
            model=model,
            input_shape=input_shape,
            to_file=save_path,
            show_neurons=True,
            layer_spacing=300,
            node_size=60,
            background_fill='white',
            connector_fill='green',
            connector_width=2
        )
        
        print(f"✅ Визуализация сохранена в: {save_path}")
        
        # Получение информации о модели
        print(f"\n📋 Информация о модели:")
        print(f"   • Размер изображения: {image.size}")
        print(f"   • Формат: {image.format}")
        print(f"   • Режим: {image.mode}")
        
        # Подсчет слоев
        layer_count = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU, nn.BatchNorm1d, nn.Dropout, nn.Sigmoid)):
                layer_count += 1
        
        print(f"   • Количество слоев: {layer_count}")
            
    except Exception as e:
        print(f"❌ Ошибка при создании визуализации: {str(e)}")

def create_detailed_model_summary(input_size=100):
    """Создание детального описания модели"""
    print("=== Детальное описание модели ===")
    
    model = BearingClassifier(input_size)
    
    print(f"🏗️  АРХИТЕКТУРА МОДЕЛИ:")
    print(f"   • Входной слой: {input_size} признаков (после PCA)")
    print(f"   • Скрытый слой 1: {input_size} → 256 нейронов (ReLU + BatchNorm + Dropout 0.3)")
    print(f"   • Скрытый слой 2: 256 → 128 нейронов (ReLU + BatchNorm + Dropout 0.3)")
    print(f"   • Скрытый слой 3: 128 → 64 нейронов (ReLU + BatchNorm + Dropout 0.2)")
    print(f"   • Скрытый слой 4: 64 → 32 нейронов (ReLU)")
    print(f"   • Выходной слой: 32 → 1 нейрон (Sigmoid)")
    
    print(f"\n📊 СТАТИСТИКА ПАРАМЕТРОВ:")
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        print(f"   • {name}: {param_count:,} параметров")
    
    print(f"\n   • Всего параметров: {total_params:,}")
    print(f"   • Обучаемых параметров: {trainable_params:,}")
    print(f"   • Необучаемых параметров: {total_params - trainable_params:,}")
    
    print(f"\n🎯 ФУНКЦИЯ ПОТЕРЬ:")
    print(f"   • MSE Loss (Mean Squared Error)")
    print(f"   • Оптимизатор: Adam (lr=0.001, weight_decay=1e-5)")
    print(f"   • Scheduler: ReduceLROnPlateau (patience=10, factor=0.5)")
    
    print(f"\n⚡ УСКОРЕНИЕ:")
    print(f"   • Поддержка MPS (Metal Performance Shaders)")
    print(f"   • Поддержка CUDA")
    print(f"   • Fallback на CPU")

def create_simple_model_diagram(save_path="visualizations/model_diagram.txt"):
    """Создание простой текстовой диаграммы модели"""
    print("=== Создание текстовой диаграммы модели ===")
    
    os.makedirs("visualizations", exist_ok=True)
    
    diagram = """
🏗️  АРХИТЕКТУРА МОДЕЛИ BearingClassifier
==========================================

Входной слой (100 признаков после PCA)
    ↓
┌─────────────────────────────────────┐
│ Linear(100 → 256)                  │
│ ReLU                               │
│ BatchNorm1d(256)                   │
│ Dropout(0.3)                       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Linear(256 → 128)                  │
│ ReLU                               │
│ BatchNorm1d(128)                   │
│ Dropout(0.3)                       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Linear(128 → 64)                   │
│ ReLU                               │
│ BatchNorm1d(64)                    │
│ Dropout(0.2)                       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Linear(64 → 32)                    │
│ ReLU                               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Linear(32 → 1)                     │
│ Sigmoid                            │
└─────────────────────────────────────┘
    ↓
Выход (0-1, масштабируется до 0-100%)

📊 СТАТИСТИКА:
• Всего параметров: 70,017
• Обучаемых параметров: 70,017
• Слоев: 17 (включая активации и нормализацию)

🎯 НАЗНАЧЕНИЕ:
• Классификация состояния подшипника
• Выход: процент состояния (0-100%)
• Пороги: <25% (плохой), 25-75% (нормальный), >75% (новый)
"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(diagram)
    
    print(f"✅ Текстовая диаграмма сохранена в: {save_path}")
    print(diagram)

def main():
    """Основная функция"""
    print("=== Визуализация модели анализа состояния подшипника ===\n")
    
    # Создание директории для визуализаций
    os.makedirs("visualizations", exist_ok=True)
    
    # Создание текстовой диаграммы
    print("1️⃣ Создание текстовой диаграммы модели")
    create_simple_model_diagram()
    
    print("\n" + "="*60 + "\n")
    
    # Визуализация базовой архитектуры
    print("2️⃣ Визуализация базовой архитектуры модели")
    visualize_model_architecture(
        input_size=100, 
        save_path="visualizations/model_architecture.png"
    )
    
    print("\n" + "="*60 + "\n")
    
    # Визуализация сохраненной модели
    print("3️⃣ Визуализация сохраненной модели")
    visualize_saved_model(
        model_dir="saved_models",
        save_path="visualizations/saved_model_architecture.png"
    )
    
    print("\n" + "="*60 + "\n")
    
    # Детальное описание модели
    print("4️⃣ Детальное описание модели")
    create_detailed_model_summary(input_size=100)
    
    print(f"\n✅ Визуализация завершена!")
    print(f"📁 Файлы сохранены в папке: visualizations/")

if __name__ == "__main__":
    main()
