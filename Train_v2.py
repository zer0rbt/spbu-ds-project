import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt


class TransformDataset(TensorDataset):
    def __init__(self, images, labels, transform=None):
        super().__init__(images, labels)
        self.transform = transform

    def __getitem__(self, index):
        images, labels = super().__getitem__(index)
        if self.transform:
            image = transforms.ToPILImage()(images)
            images = self.transform(image)
        return images, labels


def load_data_with_transforms(images_path="images_tensor.pt",
                              labels_path="labels_tensor.pt",
                              batch_size=128):

    use_cuda = torch.cuda.is_available()


    images = torch.load(images_path)
    labels = torch.load(labels_path)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = TransformDataset(images, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform


    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': 4 if use_cuda else 0,
        'pin_memory': use_cuda,
        'persistent_workers': True if use_cuda else False
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs
    )

    return train_loader, test_loader


def train_model(model, train_loader, criterion, optimizer, scheduler, epochs=2):

    use_cuda = torch.cuda.is_available()
    if use_cuda:

        cudnn.benchmark = True

        scaler = GradScaler()

    model.train()
    metrics = {"epochs": []}
    accuracies = []
    early_stopping = EarlyStopping(patience=3)

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for images, labels in loop:
            if use_cuda:

                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_cuda:

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)


                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:

                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item()

            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=running_loss / len(loop))

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        accuracies.append(accuracy)

        scheduler.step(accuracy)

        metrics["epochs"].append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "accuracy": accuracy
        })

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        if early_stopping(avg_loss):
            print("Early stopping triggered")
            break

        with open("metrics/train_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

    plot_accuracy_graph(accuracies)


@torch.no_grad()
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    use_cuda = torch.cuda.is_available()

    for images, labels in test_loader:
        if use_cuda:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            with autocast():
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
        else:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

    with open("metrics/train_metrics2.json", "r") as f:
        metrics = json.load(f)

    metrics["accuracy"] = accuracy

    with open("metrics/train_metrics2.json", "w") as f:
        json.dump(metrics, f, indent=4)


class ImprovedModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedModel, self).__init__()
        self.model = models.resnet50(pretrained=True)


        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


train_loader, test_loader = load_data_with_transforms()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedModel(num_classes=len(torch.load("labels_tensor.pt").unique())).to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def plot_accuracy_graph(accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', color='b', label='Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


    plt.savefig('metrics/accuracy_vs_epoch2.png')
    plt.close()


train_model(model, train_loader, criterion, optimizer, scheduler, epochs=10)
evaluate_model(model, test_loader)

os.system("dvc add metrics/train_metrics2.json")
os.system("dvc add metrics/accuracy_vs_epoch2.png")
os.system("git add metrics/train_metrics2.json.dvc metrics/accuracy_vs_epoch2.png.dvc")
os.system('git commit -m "feat: add data for train_v2"')
os.system("dvc push")