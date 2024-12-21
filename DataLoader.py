import os
import shutil
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import yaml


def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


config = load_config()

csv_file = config["clowns"]["dataset"]
image_column = config["image_url"]
images_dir = config["img_dir"]
output_dir = "sorted_images"
image_column = "image_url"
class_column = "scientific_name"

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_file)

existing_files = set(os.listdir(images_dir))

for i, row in tqdm(df.iterrows(), total=len(df)):
    image_name = f"image_{i}.jpg"
    class_name = row[class_column]

    if image_name in existing_files:

        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        src_path = os.path.join(images_dir, image_name)
        dest_path = os.path.join(class_dir, image_name)
        shutil.move(src_path, dest_path)
    else:
        print(f"Изображение {image_name} отсутствует в {images_dir}")

print("Данные успешно разделены по классам!")

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_file)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = ImageFolder(root=output_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

all_images = []
all_labels = []

for images, labels in dataloader:
    all_images.append(images)
    all_labels.append(labels)

all_images = torch.cat(all_images)
all_labels = torch.cat(all_labels)

torch.save(all_images, "images_tensor.pt")
torch.save(all_labels, "labels_tensor.pt")

print(1)