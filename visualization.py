import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def visualize_results(train_losses, test_losses, test_accuracies, predictions, labels):
    """Визуализация результатов"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # График потерь
    axes[0, 0].plot(train_losses, label='Обучающая потеря', color='blue')
    axes[0, 0].plot(test_losses, label='Тестовая потеря', color='red')
    axes[0, 0].set_title('График потерь')
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('Потеря')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # График точности
    axes[0, 1].plot(test_accuracies, color='green')
    axes[0, 1].set_title('График точности')
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('Точность')
    axes[0, 1].grid(True)
    
    # Преобразование меток в целые числа для матрицы ошибок
    label_mapping = {0.0: 0, 50.0: 1, 100.0: 2}
    labels_int = np.array([label_mapping[label] for label in labels])
    predictions_int = np.array([label_mapping[pred] for pred in predictions])
    
    # Матрица ошибок
    cm = confusion_matrix(labels_int, predictions_int)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Плохой', 'Нормальный', 'Новый'],
                yticklabels=['Плохой', 'Нормальный', 'Новый'],
                ax=axes[1, 0])
    axes[1, 0].set_title('Матрица ошибок')
    axes[1, 0].set_xlabel('Предсказанные метки')
    axes[1, 0].set_ylabel('Истинные метки')
    
    # Распределение предсказаний
    axes[1, 1].hist(predictions, bins=20, alpha=0.7, color='orange')
    axes[1, 1].set_title('Распределение предсказаний')
    axes[1, 1].set_xlabel('Предсказанные значения (%)')
    axes[1, 1].set_ylabel('Частота')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/bearing_analysis_results.png', dpi=300, bbox_inches='tight')