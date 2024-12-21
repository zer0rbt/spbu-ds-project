import torch
from torch.utils.data import TensorDataset, DataLoader

def load_data(images_path="images_tensor.pt", labels_path="labels_tensor.pt", batch_size=32):
    # Загрузка тензоров
    images = torch.load(images_path)
    labels = torch.load(labels_path)

    # Создание TensorDataset
    dataset = TensorDataset(images, labels)

    # Разделение на обучающую и тестовую выборки (80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Создание DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
