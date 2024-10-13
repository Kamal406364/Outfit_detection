import cv2
import pandas as pd
import torch
from PIL import Image
from transformers import ViTImageProcessor, AutoModelForImageClassification
import json


img_path = r'C:\Users\kamal\OneDrive\Desktop\outfit\photos\top.jpeg'
img = cv2.imread(img_path)


index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv(r'C:\Users\kamal\OneDrive\Desktop\outfit\colors.csv', names=index)


model_name = "facebook/deit-base-patch16-224"  
processor = ViTImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
labels = model.config.id2label  # Load class labels


def getColorName(R, G, B):
    min_distance = float('inf')
    color_name = ''
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if d <= min_distance:
            min_distance = d
            color_name = csv.loc[i, "color_name"]
    return color_name

def classify_clothing(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    model.eval()  
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return labels[predicted_class_idx]


height, width, _ = img.shape
center_x, center_y = width // 2, height // 2
b, g, r = img[center_y, center_x]
r, g, b = int(r), int(g), int(b)


color_name = getColorName(r, g, b)


clothing_type = classify_clothing(img_path)


results = {
    "color_name": color_name,
    "color_rgb": {"R": r, "G": g, "B": b},
    "clothing_type": clothing_type
}


json_output = json.dumps(results, indent=4)  
print(json_output)  
