import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def get_device():
    """Определение устройства для обучения"""
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Используется устройство: MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Используется устройство: CUDA")
    else:
        device = torch.device('cpu')
        print("Используется устройство: CPU")
    return device

def train_model(model, train_loader, test_loader, epochs=100, learning_rate=0.001):
    """Обучение модели"""
    device = get_device()
    
    model = model.to(device)
    criterion = nn.MSELoss()  # Используем MSE loss для регрессии
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    print("Начало обучения...")
    
    for epoch in range(epochs):
        # Обучение
        model.train()
        train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features).squeeze()
            
            # Масштабирование выходов в диапазон [0, 100]
            outputs = outputs * 100
            
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            # Градиентное клиппирование для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        # Валидация
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_features).squeeze()
                outputs = outputs * 100  # Масштабирование в проценты
                loss = criterion(outputs, batch_labels)
                test_loss += loss.item()
                
                # Определение предсказанного класса для точности
                predicted = torch.zeros_like(outputs)
                predicted[outputs < 25] = 0.0
                predicted[(outputs >= 25) & (outputs < 75)] = 50.0
                predicted[outputs >= 75] = 100.0
                
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        test_accuracy = correct / total
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        scheduler.step(test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Test Loss: {test_loss:.4f}, '
                  f'Test Accuracy: {test_accuracy:.4f}')
    
    return model, train_losses, test_losses, test_accuracies

def evaluate_model(model, test_loader, scaler):
    """Оценка модели"""
    device = get_device()
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features).squeeze()
            outputs = outputs * 100  # Масштабирование в проценты
            
            # Определение предсказанного класса
            predictions = torch.zeros_like(outputs)
            predictions[outputs < 25] = 0.0
            predictions[(outputs >= 25) & (outputs < 75)] = 50.0
            predictions[outputs >= 75] = 100.0
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    # Преобразование в целые числа для метрик классификации
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Преобразование меток в целые числа (0, 1, 2)
    label_mapping = {0.0: 0, 50.0: 1, 100.0: 2}
    all_labels_int = np.array([label_mapping[label] for label in all_labels])
    all_predictions_int = np.array([label_mapping[pred] for pred in all_predictions])
    
    # Вычисление метрик
    accuracy = accuracy_score(all_labels_int, all_predictions_int)
    
    print(f"\nТочность модели: {accuracy:.4f}")
    print("\nОтчет о классификации:")
    print(classification_report(all_labels_int, all_predictions_int, 
                              target_names=['Плохой (0%)', 'Нормальный (50%)', 'Новый (100%)']))
    
    return accuracy, all_predictions, all_labels 