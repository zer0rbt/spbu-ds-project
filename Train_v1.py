import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from DataPreprocessing import load_data
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt

train_loader, test_loader = load_data()


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.model = models.alexnet(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet(num_classes=len(train_loader.dataset.dataset.tensors[1].unique())).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if not os.path.exists("metrics"):
    os.makedirs("metrics")


def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    metrics = {"epochs": []}
    accuracies = []
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)


            optimizer.zero_grad()


            outputs = model(images)


            loss = criterion(outputs, labels)


            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / len(loop))


            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        accuracies.append(accuracy)

        metrics["epochs"].append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": accuracy})

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


        with open("metrics/train_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)


    plot_accuracy_graph(accuracies)


def plot_accuracy_graph(accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', color='b', label='Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


    plt.savefig('metrics/accuracy_vs_epoch.png')
    plt.close()


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")


    with open("metrics/train_metrics.json", "r") as f:
        metrics = json.load(f)

    metrics["accuracy"] = accuracy


    with open("metrics/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


train_model(model, train_loader, criterion, optimizer, epochs=10)
evaluate_model(model, test_loader)

os.system("dvc add metrics/train_metrics.json")
os.system("dvc add metrics/accuracy_vs_epoch.png")
os.system("git add metrics/train_metrics.json.dvc metrics/accuracy_vs_epoch.png.dvc")
os.system('git commit -m "feat: add data for train_v1"')
os.system("dvc push")
