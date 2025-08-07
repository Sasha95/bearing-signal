import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class BearingDataset(Dataset):
    """Датасет для данных подшипника"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def process_numeric_data(df):
    """Обработка числовых данных - замена запятых на точки"""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    return df

def load_and_preprocess_data():
    """Загрузка и предобработка данных"""
    print("Загрузка данных...")
    
    # Загрузка данных с правильным разделителем
    bad_data = pd.read_csv('data/bad_bearing(0).csv', sep=';')
    normal_data = pd.read_csv('data/normal_bearing(0.5).csv', sep=';')
    new_data = pd.read_csv('data/new_bearing(1).csv', sep=';')
    
    print(f"Размеры данных:")
    print(f"Плохой подшипник: {bad_data.shape}")
    print(f"Нормальный подшипник: {normal_data.shape}")
    print(f"Новый подшипник: {new_data.shape}")
    
    # Обработка NaN значений
    bad_data = bad_data.fillna(0)
    normal_data = normal_data.fillna(0)
    new_data = new_data.fillna(0)
    
    # Удаление столбца timestamp (первый столбец) и строки с частотами (первая строка)
    bad_data = bad_data.iloc[1:, 1:]  # Убираем первую строку (частоты) и первый столбец (timestamp)
    normal_data = normal_data.iloc[1:, 1:]
    new_data = new_data.iloc[1:, 1:]
    
    print(f"Размеры данных после удаления timestamp и частот:")
    print(f"Плохой подшипник: {bad_data.shape}")
    print(f"Нормальный подшипник: {normal_data.shape}")
    print(f"Новый подшипник: {new_data.shape}")
    
    # Обработка числовых данных
    bad_data = process_numeric_data(bad_data)
    normal_data = process_numeric_data(normal_data)
    new_data = process_numeric_data(new_data)
    
    # Добавление меток в процентах (0-100)
    bad_data['label'] = 0.0    # 0% - плохой подшипник
    normal_data['label'] = 50.0 # 50% - нормальный подшипник
    new_data['label'] = 100.0  # 100% - новый подшипник
    
    # Объединение данных
    all_data = pd.concat([bad_data, normal_data, new_data], ignore_index=True)
    
    # Разделение на признаки и метки
    X = all_data.drop('label', axis=1)
    y = all_data['label']
    
    print(f"Общий размер данных: {all_data.shape}")
    print(f"Количество признаков (частот): {X.shape[1]}")
    print(f"Уникальные метки: {y.unique()}")
    
    return X, y

def prepare_data_loaders(X, y, batch_size=32):
    """Подготовка DataLoader для обучения"""
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Нормализация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Дополнительная обработка NaN значений
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)
    
    # Уменьшение размерности с помощью PCA
    n_components = min(100, X_train_scaled.shape[1])  # Максимум 100 компонент
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"Размерность данных после PCA: {X_train_pca.shape[1]}")
    print(f"Объясненная дисперсия: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Создание датасетов
    train_dataset = BearingDataset(X_train_pca, y_train.values)
    test_dataset = BearingDataset(X_test_pca, y_test.values)
    
    # Создание DataLoader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler, pca 