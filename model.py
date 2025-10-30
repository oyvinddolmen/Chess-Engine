import torch.nn as nn


class ChessModel(nn.Module):
    def __init__(self, num_classes):
        super(ChessModel, self).__init__()
        # conv1 -> relu -> conv2 -> relu -> flatten -> fc1 -> relu -> fc2

        # 2 konvuleringslag med relu som aktiveringsfunksjon
        # inngangen har 13 kanaler, 6 hvite + 6 svarte + 1 for hvor trekket går
        # CNN fordi den passer på data som har romlig struktur, som et sjakkbrett.

        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

        # Initialiser vektene
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output raw logits
        return x
    
'''
Input (13x8x8)
  ↓
Conv1 + ReLU
  ↓
Conv2 + ReLU
  ↓
Flatten (→ 8192)
  ↓
FC1 + ReLU (→ 256)
  ↓
FC2 (→ num_classes logits)

'''
