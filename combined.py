import cv2
import pandas as pd
import torch
from PIL import Image
from transformers import ViTImageProcessor, AutoModelForImageClassification
import json
import os
import google.generativeai as genai

# Load the image and color CSV
img_path = r'C:\Users\kamal\OneDrive\Desktop\outfit\photos\top.jpeg'
img = cv2.imread(img_path)

index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv(r'C:\Users\kamal\OneDrive\Desktop\outfit\colors.csv', names=index)

# Load model for image classification
model_name = "facebook/deit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
labels = model.config.id2label  # Load class labels

# Function to get the color name based on RGB values
def getColorName(R, G, B):
    min_distance = float('inf')
    color_name = ''
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if d <= min_distance:
            min_distance = d
            color_name = csv.loc[i, "color_name"]
    return color_name

# Function to classify the type of clothing based on the image
def classify_clothing(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    model.eval()  
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return labels[predicted_class_idx]

# Get the central pixel color and determine its name
height, width, _ = img.shape
center_x, center_y = width // 2, height // 2
b, g, r = img[center_y, center_x]
r, g, b = int(r), int(g), int(b)

color_name = getColorName(r, g, b)

# Classify the clothing type from the image
clothing_type = classify_clothing(img_path)

# Prepare the results as JSON
results = {
    "color_name": color_name,
    "color_rgb": {"R": r, "G": g, "B": b},
    "clothing_type": clothing_type
}

json_output = json.dumps(results, indent=4)
#print(json_output)

# Generative AI API (Google Gemini)
api_key = "AIzaSyAXd_GrVgQj0TVFp5qX_D8_PdbZBPm8wlI"  # Replace with your actual API key
genai.configure(api_key=api_key)

# Model configuration for Gemini AI
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_output_tokens": 100,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",
    generation_config=generation_config,
)

chat_session = model.start_chat(
    history=[
        {
            "role": "user",
            "parts": [
                "hi\n",
            ],
        },
        {
            "role": "model",
            "parts": [
                "welcome, i will  help you with your fashion question and output will be short and sweet in one line perfect match\n",

            ],
        },
    ]
)

# Using the JSON output as input for the user prompt
user_prompt = f"Based on my current clothing type '{results['clothing_type']}' and the color '{results['color_name']}', can you suggest an outfit?"
response = chat_session.send_message(user_prompt)

print()
print(response.text)
