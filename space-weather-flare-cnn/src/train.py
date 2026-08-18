# training script

import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from generate_data import make_dataset
from model import FlareCNN

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "dataset.npz")
MODEL_PATH = os.path.join(HERE, "..", "models", "flare_cnn.pt")
REPORT = os.path.join(HERE, "..", "reports", "metrics.json")


def accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            p = model(x)
            pred = (torch.sigmoid(p) >= 0.5).float()
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def main():
    os.makedirs(os.path.join(HERE, "..", "models"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "..", "reports"), exist_ok=True)

    if os.path.exists(DATA):
        d = np.load(DATA)
        images, labels = d["images"], d["labels"]
        print("loaded", images.shape)
    else:
        images, labels = make_dataset(out_folder=os.path.join(HERE, "..", "data"))

    n = len(labels)
    idx = np.arange(n)
    np.random.seed(0)
    np.random.shuffle(idx)
    images, labels = images[idx], labels[idx]
    cut = int(n * 0.8)
    xtr = torch.tensor(images[:cut])
    ytr = torch.tensor(labels[:cut], dtype=torch.float32)
    xva = torch.tensor(images[cut:])
    yva = torch.tensor(labels[cut:], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(xva, yva), batch_size=32)

    model = FlareCNN()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for ep in range(6):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        acc = accuracy(model, val_loader)
        print("epoch", ep + 1, "val acc", round(acc, 3), "loss", round(loss.item(), 3))

    torch.save(model.state_dict(), MODEL_PATH)
    with open(REPORT, "w") as f:
        json.dump({"val_accuracy": accuracy(model, val_loader)}, f)
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()
