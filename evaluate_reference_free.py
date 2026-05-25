# -*- coding: utf-8 -*-

# ChartVE_Evalaute
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

import os
import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Paths
image_dir = "/content/pilotSimple600_images"
caption_file = "/content/ChartAltpilot_inference_Separate_schema_L1L2L3.txt"

model_name = "khhuang/chartve"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# Load model
model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
processor = DonutProcessor.from_pretrained(model_name)

def format_query(sentence):
    return f"Does the image entails this statement: \"{sentence}\""
image_to_caption = {}

with open(caption_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        match = re.match(r"Generated text for image (.*?): (.*)", line)
        if match:
            image_name = match.group(1)
            caption = match.group(2)
            image_to_caption[image_name] = caption

print(f"Loaded {len(image_to_caption)} captions")

results = []

for image_name, caption in image_to_caption.items():

    image_path = os.path.join(image_dir, image_name)

    if not os.path.exists(image_path):
        print(f"Missing image: {image_name}")
        continue

    try:
        img = Image.open(image_path).convert("RGB")
    except:
        print(f"Error loading image: {image_name}")
        continue

    pixel_values = processor(img, random_padding=False, return_tensors="pt").pixel_values.to(device)

    sentences = sent_tokenize(caption)

    sentence_scores = []

    for sentence in sentences:
        query = format_query(sentence)

        decoder_input_ids = processor.tokenizer(
            query,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=510
        ).input_ids.to(device)

        outputs = model(pixel_values, decoder_input_ids=decoder_input_ids)

        prob_yes = torch.nn.functional.softmax(
            outputs['logits'].squeeze()[-1, [2334, 49922]],
            dim=0
        )[1].item()

        sentence_scores.append(prob_yes)

    if len(sentence_scores) == 0:
        continue

    final_score = min(sentence_scores)

    results.append({
        "image": image_name,
        "score": final_score,
        "sentence_scores": sentence_scores
    })

    print(f"{image_name} → {final_score:.4f}")

all_scores = [r["score"] for r in results]
dataset_score = sum(all_scores) / len(all_scores)

output_file = "/content/CharAltPilot_ChartVE_scores_Separate_schema_L1L2L3.txt"

all_scores = [r["score"] for r in results]
dataset_score = sum(all_scores) / len(all_scores) if all_scores else 0
print("\nFinal Dataset CHARTVE Score:", dataset_score)

with open(output_file, "w", encoding="utf-8") as out_f:
    for res in results:
        out_f.write(f"Image: {res['image']}, ChartVE: {res['score']:.4f}\n")

print("\nFinal Dataset CHARTVE Score:", dataset_score)
print(f"Scores successfully saved to {output_file}")


#CLIPScore
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model.eval()

image_folder = "/content/pilotSimple600_images"
txt_file = "/content/ChartAltpilot_inference_Separate_schema_L1L2L3.txt"
output_file = "/content/ChartAltpilot_CLIPScore_Separate_schema_L1L2L3.txt"

pairs = []
with open(txt_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        try:
            left, caption = line.split(":", 1)
            image_name = left.replace("Generated text for image", "").strip()
            caption = caption.strip()
            pairs.append((image_name, caption))
        except Exception:
            continue

print("Total samples parsed:", len(pairs))

results = []

with torch.no_grad():
    for image_name, caption in pairs:
        image_path = os.path.join(image_folder, image_name)

        if not os.path.exists(image_path):
            print("Missing:", image_path)
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_name}: {e}")
            continue

        inputs = processor(
            text=[caption],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        outputs = model(**inputs)

        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        score = (image_embeds * text_embeds).sum().item()
        score = max(score, 0)

        results.append({
            "image": image_name,
            "score": score
        })

all_scores = [r["score"] for r in results]
avg_clipscore = sum(all_scores) / len(all_scores) if all_scores else 0

print("\n--- Execution Summary ---")
print("Valid samples evaluated:", len(results))
print(f"Average CLIPScore: {avg_clipscore:.4f}")

with open(output_file, "w", encoding="utf-8") as out_f:
    for res in results:
        out_f.write(f"Image: {res['image']}, CLIPScore: {res['score']:.4f}\n")

print(f"CLIPScore records successfully saved to: {output_file}")