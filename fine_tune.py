# -*- coding: utf-8 -*-

#Seperate Tag with Schema provided Finetune
import os
import torch
import json
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
import numpy as np
import pdb

# ==== Configuration ====
config = {
    "use_lora": False,
    "use_qlora": True,
    "max_epochs": 5,
    "check_val_every_n_epoch": 1,
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 8,
    "lr": 1e-4,
    "batch_size": 1,
    "warmup_steps": 50,
    "verbose": True,
    "result_path": "trained_model"
}

image_dir = "/content/image"
train_json = "/content/train.json"
val_json = "/content/val.json"
checkpoint_dir = Path("trained_model")

# ==== Load Processor and Model ====
processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")

if config["use_qlora"]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        "ahmed-masry/chartgemma",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
elif config["use_lora"]:
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        "ahmed-masry/chartgemma",
        torch_dtype=torch.float16,
    )
else:
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        "ahmed-masry/chartgemma",
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
    )
    for param in model.vision_tower.parameters():
        param.requires_grad = False
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False


# ==== Apply PEFT ====
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)


class ChartGemmaDataset(Dataset):
    def __init__(self, json_path, image_dir):
        with open(json_path, "r") as f:
            raw_data = json.load(f)

        self.image_dir = image_dir
        self.data = []

        for item in raw_data:
            tags = item.get("L2L3_tags", "").strip() #+ " " + item.get("L2L3_tags", "").strip()

            prompt = (
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


            label = item["caption_L2L3"].strip() #+ ' ' + item["caption_L2L3"].strip()

            self.data.append({
                "image_id": item["img_id"],
                "prompt": prompt,
                "label": label
            })
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = os.path.join(self.image_dir, sample["image_id"] + ".png")
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        return image, sample["prompt"], sample["label"]


train_dataset = ChartGemmaDataset(train_json, image_dir)
val_dataset = ChartGemmaDataset(val_json, image_dir)


# ==== Collate ====
def train_collate_fn(examples):
    images, input_texts, output_texts = zip(*examples)
    inputs = processor(
        text=input_texts,
        images=list(images),
        suffix=output_texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        max_length=128
    )
    return inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"], inputs["pixel_values"], inputs["labels"]

def eval_collate_fn(examples):
    images, texts, answers = zip(*examples)
    inputs = processor(
        text=list(texts),
        images=list(images),
        suffix=list(answers),  # This will be used as labels
        return_tensors="pt",
        padding=True,
        truncation=False,
        max_length=128
    )
    return inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"], inputs["labels"]

# ==== Lightning Module ====
class ChartGemmaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, pixel_values, labels = batch
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            labels=labels
        )
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def compute_metric(self, gt, pred):
        try:
            return abs(float(gt) - float(pred)) / abs(float(gt)) <= 0.05
        except:
            return str(gt).lower() == str(pred).lower()

    def validation_step(self, batch, batch_idx):
            input_ids, attention_mask, pixel_values, labels = batch  # `labels` is already tokenized

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )
            val_loss = outputs.loss
            self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True)
            return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.config["batch_size"], shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=self.config["batch_size"], shuffle=False, num_workers=2)

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = os.path.join(
            self.config['result_path'],
            f'ChartGemma-checkpoint-epoch={self.current_epoch}-step={self.global_step}'
        )
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

# ==== Training ====
wandb_logger = WandbLogger(project="ChartGemma", id=None)
model_module = ChartGemmaModelPLModule(config, processor, model)

checkpoint_callback = ModelCheckpoint(
    dirpath=str(checkpoint_dir),
    filename="checkpoint-{epoch}",
    save_top_k=-1,
    save_weights_only=False,
    every_n_epochs=1
)


# Auto-resume logic
resume_path = checkpoint_dir / "last.ckpt"
resume_kwargs = {"resume_from_checkpoint": str(resume_path)} if resume_path.exists() else {}

trainer = L.Trainer(
    accelerator="gpu",
    devices=[0],
    max_epochs=config["max_epochs"],
    accumulate_grad_batches=config["accumulate_grad_batches"],
    check_val_every_n_epoch=config["check_val_every_n_epoch"],
    gradient_clip_val=config["gradient_clip_val"],
    precision="16-mixed",
    num_sanity_val_steps=0,
    callbacks=[checkpoint_callback],
    logger=wandb_logger,
    default_root_dir=str(checkpoint_dir),
    **resume_kwargs
)

trainer.fit(model_module)