# Alt-Tag: Enhancing Alt Text Generation for Charts via Tag Guidance

[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/yanchuqiao/Alt-tag-Dataset)
[![Model](https://img.shields.io/badge/Based%20on-ChartGemma-blue)](https://huggingface.co/ahmed-masry/chartgemma)

This repository contains the official implementation and experiments for the paper:

**“Alt-Tag: Enhancing Alt Text Generation for Charts via Tag Guidance”**

---

## 📦 Dataset

The dataset used in this work is publicly available on Hugging Face:

👉 https://huggingface.co/datasets/yanchuqiao/Alt-tag-Dataset

---

## 🧠 Repository Structure

### 1. Tag Schema Clustering

This module extracts key words from the dataset and performs clustering and visualization of tag schemas.

- `extracting_words.py` — Extracts keywords from chart data  
- `clustering_and_visulization.py` — Performs clustering and generates visualizations  

---

### 2. Fine-tuning and Inference

The fine-tuning and inference pipeline is adapted from the original ChartGemma implementation:

https://huggingface.co/ahmed-masry/chartgemma

Scripts:

- `fine_tune.py` — Fine-tunes the model on the Alt-Tag dataset  
- `inference.py` — Runs inference using trained models  

---

### 3. Evaluation

This repository includes multiple evaluation protocols:

- `evaluate_reference_based.py` — Reference-based metrics (e.g., BLEU-4, ROUGE_L, TER, SBERT)  
- `evaluate_reference_free.py` — Reference-free evaluation metrics (ChartVE and CLIPScore)  
- `evaluate_llm_vlm.py` — Evaluation using LLM/VLM judges  
- `paired_bootstrap.py` — Statistical significance testing using paired bootstrap  
