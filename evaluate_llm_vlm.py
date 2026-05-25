# -*- coding: utf-8 -*-

#information concentration
import os
import re
import json
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")


image_dir = "/content/image"
txt_file = "/content/Vistext_inference_Separate_schema_L1L2L3.txt"
gt_file = "/content/test.json"
output_txt = "/content/Vistext_info_Separate_schema_L1L2L3_qwen.txt"

open(output_txt, "w").close()

image_to_type = {}

with open(gt_file, "r", encoding="utf-8") as f:
    gt_data = json.load(f)

for item in gt_data:
    image_id = item["img_id"]
    chart_type = item["L1_properties"][0]

    image_to_type[image_id] = chart_type

print(f"Loaded chart types: {len(image_to_type)}")

open(output_txt, "w").close()

# Load generated captions
image_to_caption = {}

with open(txt_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if "Generated text for image" in line:
        match = re.match(r"Generated text for image\s+(.+?):\s*(.*)",line)
        if match:
            image_name = match.group(1).strip()
            caption = match.group(2).strip()

            # remove optional tags
            caption = re.sub(r"<[^>]+>", "", caption)
            caption = re.sub(r"\s+", " ", caption).strip()

            image_to_caption[image_name] = caption

print(f"Loaded {len(image_to_caption)} samples")

def build_prompt(chart_type, generated_text):
    return f"""
You are an unbiased and strict evaluator.  Your task is to evaluate INFORMATION CONCENTRATION.

Information concentration measures how well the description focuses on the key visual in the chart, and avoids unnecessary or low-importance details.

Important information depends on chart type and must be inferred from the IMAGE.

## Key Information Definition

For Bar Charts, important information may include:
- The chart type
- The chart title, if present
- The label and value range of the horizontal axis
- The label and value range of the vertical axis
- The category with the highest value
- The category with the lowest value
- Important category values or comparisons between categories

For Line Charts, important information may include:
- The chart type
- The chart title, if present
- The label and value range of the horizontal axis
- The label and value range of the vertical axis
- The overall direction or trend of a line
- Significant short-term trend of a line
- Peak values
- Lowest values

For Pie Charts, important information may include:
- The chart type
- The chart title, if present
- The category labels of the segments
- The category with the largest percentage
- The category with the smallest percentage
- Major proportions or percentage distributions

## Scoring scale (1–5)

1 = focuses mostly on irrelevant or minor details
2 = weak focus, many low-importance details
3 = mixed focus, some key info but noticeable noise
4 = mostly focused on key information
5 = highly focused on key information only


Chart Type:
{chart_type}

Generated Description:
{generated_text}

Output format:
Score:
"""

results = []

model.eval()

for image_name, gen_caption in tqdm(image_to_caption.items()):

    image_path = os.path.join(image_dir, image_name)+'.png'

    if image_name not in image_to_type:
        print(f"Missing chart type: {image_name}")
        continue

    if not os.path.exists(image_path):
        print(f"Missing image: {image_name}")
        continue

    chart_type = image_to_type[image_name]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": build_prompt(chart_type, gen_caption)},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.2
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True
    )[0]

    match = re.search(r"\b([1-5])\b", output_text)
    score = int(match.group(1)) if match else None

    results.append(score)
    with open(output_txt, "a", encoding="utf-8") as f:
        f.write(f"Image: {image_name}\n")
        f.write(f"Chart Type: {chart_type}\n")
        f.write(f"Score: {score}\n")
        f.write(f"Caption: {gen_caption}\n")
        f.write(f"Reason: {output_text}\n")
        f.write("-" * 60 + "\n\n")

    print(f"{image_name} ({chart_type}) → {output_text}")


valid_scores = [s for s in results if s is not None]

print("\n========================")
print("Valid samples:", len(valid_scores))
print("Average Information Concentration Score:", sum(valid_scores) / len(valid_scores))
print("========================")

##### Structure Clartiy
import re
import json
import torch
from tqdm import tqdm
from transformers import pipeline
from huggingface_hub import login
import os

os.environ["HF_TOKEN"] = "enter_token"
HF_TOKEN = os.environ["HF_TOKEN"]

# Load Model
model_id = "google/gemma-2-2b-it"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

txt_file = "/content/Vistext_inference_Separate_schema_L2L3.txt"
gt_file = "/content/test.json"
output_txt = "/content/Vistext_structure_seperate_schema_L2L3_gemma.txt"

open(output_txt, "w").close()

image_to_type = {}

with open(gt_file, "r", encoding="utf-8") as f:
    gt_data = json.load(f)

for item in gt_data:
    image_id = item["img_id"]
    chart_type = item["L1_properties"][0]

    image_to_type[image_id] = chart_type

print(f"Loaded chart types: {len(image_to_type)}")

image_to_caption = {}

with open(txt_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if "Generated text for image" in line:
        match = re.match(r"Generated text for image\s+(.+?):\s*(.*)",line)
        if match:
            image_name = match.group(1).strip()
            caption = match.group(2).strip()

            caption = re.sub(r"<[^>]+>", "", caption)
            caption = re.sub(r"\s+", " ", caption).strip()

            image_to_caption[image_name] = caption

print(f"Loaded {len(image_to_caption)} samples")


def build_prompt(chart_type, generated_text):

    return f"""
  Given an alternative text(alt text) for a chart image , please serve as an unbiased and fair evaluator.

  Your task is to evaluate the alt text based on STRUCTURE CLARITY.

  Structure clarity refers to how well the alt text organizes information for easy understanding, including:
  - Logical ordering of information
  - Grouping of related information into coherent units or sentences
  - Smooth progression of ideas without abrupt jumps
  - Clear separation of distinct concepts or visual elements

  The alt text may contain different structures of information depending on the chart type.

  For Bar Charts, the structure may follow:
  - The chart type
  - The chart title, if present
  - The label and value range of the horizontal axis
  - The label and value range of the vertical axis
  - The category with the highest value
  - The category with the lowest value
  - Important category values or comparisons between categories

  For Line Charts, the structure may follow:
  - The chart type
  - The chart title, if present
  - The label and value range of the horizontal axis
  - The label and value range of the vertical axis
  - The overall direction or trend of a line
  - Significant short-term trend of a line
  - Peak values
  - Lowest values

  For Pie Charts, the structure may follow:
  - The chart type
  - The chart title, if present
  - The category labels of the segments
  - The category with the largest percentage
  - The category with the smallest percentage
  - Major proportions or percentage distributions


  Rate the caption on a scale from 1 to 5:
  1 = very poor structure (confusing order, scattered ideas, no grouping)
  2 = weak structure (some grouping but inconsistent or unclear flow)
  3 = acceptable structure (generally understandable but imperfect organization)
  4 = good structure (clear grouping and logical progression with minor issues)
  5 = excellent structure (well-organized, highly logical, and easy to follow)

  Task: evaluate the alt text based on STRUCTURE CLARITY.

  Chart type: {chart_type}

  Alt text: {generated_text}

  Output format:
  Score:
  """.strip()

results = []

for image_name, gen_caption in tqdm(image_to_caption.items()):

    if image_name not in image_to_type:
        print(f"Missing chart type: {image_name}")
        continue

    chart_type = image_to_type[image_name]

    prompt = build_prompt(chart_type, gen_caption)

    response = pipe(
        prompt,
        max_new_tokens=32,
        do_sample=False,
        temperature=0.0
    )

    output_text = response[0]["generated_text"]
    match = re.search(r"Score:\s*([1-5])", output_text)

    score = int(match.group(1)) if match else None
    results.append(score)

    with open(output_txt, "a", encoding="utf-8") as f:

        f.write(f"Image: {image_name}\n")
        f.write(f"Chart Type: {chart_type}\n")
        f.write(f"Score: {score}\n")
        f.write("-" * 60 + "\n\n")

    print(f"{image_name} ({chart_type}) → {score}")

valid_scores = [s for s in results if s is not None]

print("\n========================")
print("Valid samples:", len(valid_scores))

if len(valid_scores) > 0:
    print(
        "Average Information Concentration Score:",
        sum(valid_scores) / len(valid_scores)
    )

print("========================")