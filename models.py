import torch
import torch.nn as nn

class BearingClassifier(nn.Module):
    """Нейронная сеть для классификации состояния подшипника"""
    def __init__(self, input_size):
        super(BearingClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Для вывода в диапазоне [0, 1]
        )
    
    def forward(self, x):
        return self.layers(x) 