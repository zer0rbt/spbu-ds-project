import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from DataPreprocessing import load_data
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt

# Загрузка данных
train_loader, test_loader = load_data()


# Определение AlexNet
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.model = models.alexnet(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)


# Инициализация модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet(num_classes=len(train_loader.dataset.dataset.tensors[1].unique())).to(device)

# Оптимизатор и функция потерь
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Создание директории для метрик и графиков, если она не существует
if not os.path.exists("metrics"):
    os.makedirs("metrics")


# Обучение модели с прогресс-баром и записью метрик в файл
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    metrics = {"epochs": []}
    accuracies = []  # Список для хранения точности на каждой эпохе
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            # Обнуление градиентов
            optimizer.zero_grad()

            # Прямой проход
            outputs = model(images)

            # Функция потерь
            loss = criterion(outputs, labels)

            # Обратный проход
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / len(loop))

            # Подсчёт точности для текущего батча
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        accuracies.append(accuracy)

        metrics["epochs"].append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": accuracy})

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Сохраняем метрики после каждой эпохи в файл
        with open("metrics/train_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

    # После завершения обучения строим график
    plot_accuracy_graph(accuracies)


# Функция для построения графика
def plot_accuracy_graph(accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', color='b', label='Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Сохраняем график в файл
    plt.savefig('metrics/accuracy_vs_epoch.png')
    plt.close()


# Тестирование модели и сохранение accuracy в метриках
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

    # Добавляем accuracy в метрики
    with open("metrics/train_metrics.json", "r") as f:
        metrics = json.load(f)

    metrics["accuracy"] = accuracy

    # Сохраняем метрики с accuracy в файл
    with open("metrics/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


# Запуск обучения и тестирования
train_model(model, train_loader, criterion, optimizer, epochs=10)
evaluate_model(model, test_loader)

# Сохраняем метрики и график в DVC
os.system("dvc add metrics/train_metrics.json")
os.system("dvc add metrics/accuracy_vs_epoch.png")
os.system("git add metrics/train_metrics.json.dvc metrics/accuracy_vs_epoch.png.dvc")
os.system('git commit -m "feat: add data for train_v1"')
os.system("dvc push")
