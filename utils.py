import torch
import numpy as np
import warnings
import pickle
import os
from models import BearingClassifier
warnings.filterwarnings('ignore')

# Установка seed для воспроизводимости
def set_seeds():
    """Установка seed для воспроизводимости результатов"""
    torch.manual_seed(42)
    np.random.seed(42)

def print_summary(accuracy, state, state_value, mean_prediction):
    """Вывод итогового резюме"""
    print(f"\n=== Анализ завершен ===")
    print(f"Финальная точность модели: {accuracy:.4f}")
    print(f"Состояние тестового подшипника: {state} ({state_value}%)")
    print(f"Процент износа подшипника: {100 - mean_prediction:.2f}%")

def print_model_summary(model, input_size, train_losses, test_losses, test_accuracies, 
                       accuracy, predictions, labels, mean_prediction, state, state_value):
    """Вывод подробного самари по модели"""
    print("\n" + "="*80)
    print("📊 ПОДРОБНЫЙ САМАРИ ПО МОДЕЛИ АНАЛИЗА СОСТОЯНИЯ ПОДШИПНИКА")
    print("="*80)
    
    # Архитектура модели
    print("\n🏗️  АРХИТЕКТУРА МОДЕЛИ:")
    print(f"   • Входной слой: {input_size} признаков (после PCA)")
    print(f"   • Скрытые слои: 256 → 128 → 64 → 32 нейронов")
    print(f"   • Активации: ReLU + BatchNorm + Dropout")
    print(f"   • Выходной слой: 1 нейрон с Sigmoid")
    print(f"   • Общее количество параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    # Устройство обучения
    device = next(model.parameters()).device
    print(f"\n⚡ УСТРОЙСТВО ОБУЧЕНИЯ:")
    print(f"   • Используемое устройство: {device}")
    if device.type == 'mps':
        print(f"   • Ускорение: MPS (Metal Performance Shaders)")
    elif device.type == 'cuda':
        print(f"   • Ускорение: CUDA")
    else:
        print(f"   • Ускорение: CPU")
    
    # Метрики обучения
    print(f"\n📈 МЕТРИКИ ОБУЧЕНИЯ:")
    print(f"   • Финальная точность: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   • Финальная обучающая потеря: {train_losses[-1]:.4f}")
    print(f"   • Финальная тестовая потеря: {test_losses[-1]:.4f}")
    print(f"   • Максимальная точность: {max(test_accuracies):.4f} ({max(test_accuracies)*100:.2f}%)")
    print(f"   • Минимальная тестовая потеря: {min(test_losses):.4f}")
    
    # Анализ предсказаний
    predictions_array = np.array(predictions)
    labels_array = np.array(labels)
    
    print(f"\n🎯 АНАЛИЗ ПРЕДСКАЗАНИЙ:")
    print(f"   • Количество образцов: {len(predictions)}")
    print(f"   • Среднее предсказание: {predictions_array.mean():.2f}%")
    print(f"   • Стандартное отклонение: {predictions_array.std():.2f}%")
    print(f"   • Минимальное предсказание: {predictions_array.min():.2f}%")
    print(f"   • Максимальное предсказание: {predictions_array.max():.2f}%")
    
    # Распределение предсказаний по классам
    bad_count = np.sum(predictions_array == 0.0)
    normal_count = np.sum(predictions_array == 50.0)
    good_count = np.sum(predictions_array == 100.0)
    
    print(f"\n📊 РАСПРЕДЕЛЕНИЕ ПРЕДСКАЗАНИЙ:")
    print(f"   • Плохие подшипники (0%): {bad_count} ({bad_count/len(predictions)*100:.1f}%)")
    print(f"   • Нормальные подшипники (50%): {normal_count} ({normal_count/len(predictions)*100:.1f}%)")
    print(f"   • Новые подшипники (100%): {good_count} ({good_count/len(predictions)*100:.1f}%)")
    
    # Результат для тестового подшипника
    print(f"\n🔍 РЕЗУЛЬТАТ ДЛЯ ТЕСТОВОГО ПОДШИПНИКА:")
    print(f"   • Среднее предсказание: {mean_prediction:.2f}%")
    print(f"   • Определенное состояние: {state}")
    print(f"   • Процент износа: {100 - mean_prediction:.2f}%")
    
    # Рекомендации
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    if mean_prediction < 25:
        print(f"   • ⚠️  ПОДШИПНИК ТРЕБУЕТ НЕМЕДЛЕННОЙ ЗАМЕНЫ!")
        print(f"   • Износ составляет {100 - mean_prediction:.1f}%")
        print(f"   • Рекомендуется остановить оборудование")
    elif mean_prediction < 75:
        print(f"   • ⚡ ПОДШИПНИК В НОРМАЛЬНОМ СОСТОЯНИИ")
        print(f"   • Износ составляет {100 - mean_prediction:.1f}%")
        print(f"   • Рекомендуется мониторинг")
    else:
        print(f"   • ✅ ПОДШИПНИК В ОТЛИЧНОМ СОСТОЯНИИ")
        print(f"   • Износ составляет {100 - mean_prediction:.1f}%")
        print(f"   • Можно продолжать эксплуатацию")
    
    # Качество модели
    print(f"\n🏆 КАЧЕСТВО МОДЕЛИ:")
    if accuracy >= 0.95:
        print(f"   • Отличное качество (≥95%)")
    elif accuracy >= 0.90:
        print(f"   • Хорошее качество (90-95%)")
    elif accuracy >= 0.80:
        print(f"   • Удовлетворительное качество (80-90%)")
    else:
        print(f"   • Требует улучшения (<80%)")
    
    print("\n" + "="*80) 

def load_saved_model(model_dir="saved_models"):
    """Загрузка сохраненной модели и препроцессоров"""
    try:
        # Определение устройства
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        # Загрузка модели
        model_path = os.path.join(model_dir, "bearing_classifier.pth")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Создание модели с правильной архитектурой
        input_size = checkpoint['input_size']
        model = BearingClassifier(input_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)  # Перемещаем модель на правильное устройство
        model.eval()
        
        # Загрузка scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Загрузка PCA
        pca_path = os.path.join(model_dir, "pca.pkl")
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        
        # Загрузка метаданных
        metadata_path = os.path.join(model_dir, "model_metadata.pkl")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"✅ Модель успешно загружена из {model_dir}")
        print(f"   • Архитектура: {metadata['model_architecture']}")
        print(f"   • Входной размер: {input_size}")
        print(f"   • Описание: {metadata['description']}")
        print(f"   • Устройство: {device}")
        
        return model, scaler, pca, metadata
        
    except FileNotFoundError as e:
        print(f"❌ Ошибка: Файлы модели не найдены в {model_dir}")
        print(f"   Убедитесь, что модель была сохранена с помощью функции save_model_and_preprocessors()")
        return None, None, None, None
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {str(e)}")
        return None, None, None, None 