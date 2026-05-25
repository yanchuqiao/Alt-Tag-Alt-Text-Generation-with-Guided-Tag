# -*- coding: utf-8 -*-
#Seperate Tag with Schema provided Inference
import torch
import json
from PIL import Image
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, PaliGemmaForConditionalGeneration

# ==== Paths ====
json_path = "/content/test.json"
image_dir = "/content/image"
output_file = "/content/Vistext_inference_Separate_schema_L2L3.txt"

def format_tag_content(tag_content):
    if isinstance(tag_content, dict):
        return "\n".join(f"{k}: {v}" for k, v in tag_content.items())
    elif isinstance(tag_content, str):
        return tag_content.strip()
    elif tag_content is None:
        return ""
    return str(tag_content).strip()

# ==== Base model ====
base_model_id = "ahmed-masry/chartgemma"

processor = AutoProcessor.from_pretrained(base_model_id)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = PaliGemmaForConditionalGeneration.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    quantization_config=quant_config,
    device_map="auto"
)

# ==== Load LoRA checkpoint ====
model = PeftModel.from_pretrained(base_model, "/content/trained_model/ChartGemma-checkpoint-epoch=2-step=561")

model.eval()

# ==== Load Dataset ====
with open(json_path, "r") as f:
    data = json.load(f)


def build_prompt(tags: str) -> str:
    tags = (tags or "").strip()
    if tags == "":
        tags = "NONE"
    return (
            "<image>\n"
            "You are given a chart, tags scehmas for each chart type, and a list of guidance tags.\n"
            "Determine the chart type. The chart type must be one of: line chart, bar chart, or pie chart.\n"
            "Then write the full L2L3 description using the appropriate tag schema below for that chart type.\n\n"

            "Schema for Bar chart: <MAX_EXTREME>, <MIN_EXTREME>, <STATISTIC>, <TREND>\n"
            "Schema for Line chart: <OVERALL_TREND>, <LOCAL_TREND>, <MAX_EXTREME>, <MIN_EXTREME>, <STATISTIC>\n"
            "Schema for Pie chart: <MAX_EXTREME>, <MIN_EXTREME>, <PROPORTION>\n\n"

            # "Schema for Bar chart: <CHART_TYPE>, <CHART_TITLE>, <X_AXIS_LABEL>, <X_AXIS_SCALE>, <Y_AXIS_LABEL>, <Y_AXIS_SCALE>, <MAX_EXTREME>, <MIN_EXTREME>, <STATISTIC>, <TREND>\n"
            # "Schema for Line chart: <CHART_TYPE>, <CHART_TITLE>, <X_AXIS_LABEL>, <X_AXIS_SCALE>, <Y_AXIS_LABEL>, <Y_AXIS_SCALE>, <OVERALL_TREND>, <LOCAL_TREND>, <MAX_EXTREME>, <MIN_EXTREME>, <STATISTIC>\n"
            # "Schema for Pie chart: <CHART_TYPE>, <CHART_TITLE>, <SEGMENT_LABEL>, <MAX_EXTREME>, <MIN_EXTREME>, <PROPORTION>\n\n"

            "Rules:\n"
            "- Write exactly one sentence for each tag.\n"
            "- Follow the same order as the tags.\n"
            "- Each sentence must correspond to one tag.\n"
            "- Do NOT output the tags themselves.\n"
            "- Do NOT add extra information.\n\n"

            "Now write the L2L3 description for this chart.\n\n"
            "Tags:\n"
            f"{tags}\n\n"
            "Output:"
    )

# ==== Inference ====
with open(output_file, "w") as fout:
    for item in tqdm(data):
        image_id = item["img_id"]
        image_path = os.path.join(image_dir, image_id)+'.png'
        image = Image.open(image_path).convert("RGB")

        # Use GT tags from the json
        tags = item.get("L2L3_tags", "").strip() #+ " " + item.get("L2L3_tags", "").strip()
        tags = format_tag_content(tags)
        prompt_text = build_prompt(tags)

        inputs = processor(
            text=prompt_text,
            images=[image],
            return_tensors="pt"
        ).to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=128
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_text = processor.batch_decode(
            generated_ids[:, prompt_len:], skip_special_tokens=True
        )[0].strip()

        fout.write(f"Generated text for image {image_id}: {generated_text}\n")