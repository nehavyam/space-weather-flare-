# tiny cnn from a tutorial
# picture in -> one number out

import torch.nn as nn


class FlareCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16 -> 8
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
