import os
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import yaml

def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

csv_file = config["clowns"]["dataset"]
output_dir = config["img_dir"]
image_column = config["image_url"]

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_file)

for idx, url in tqdm(enumerate(df[image_column]), total=len(df)):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img_path = os.path.join(output_dir, f"image_{idx}.jpg")
        img.save(img_path)
    except Exception as e:
        print(f"Ошибка загрузки {url}: {e}")

print("Все изображения загружены.")