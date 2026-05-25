# -*- coding: utf-8 -*-

#Seperate Tag with Schema provided Evaluate
import json
import evaluate
from sentence_transformers import SentenceTransformer, util

# Load the ground truth data
with open('/content/test.json', 'r') as f:
    ground_truth_data = json.load(f)

# Build combined ground truth dict
ground_truth_combined = {}
for example in ground_truth_data:
    img_id = example['img_id']
    caption_combined = example['caption_L2L3'].strip() #+ " " + example['caption_L2L3'].strip() #example['caption_L1'].strip() + " " +
    ground_truth_combined[img_id] = caption_combined

# Load the generated summaries
generated = {}
with open('/content/Vistext_inference_Separate_schema_L2L3.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("Generated text for image"):
            parts = line.strip().split(": ", 1)
            if len(parts) == 2:
                image_filename = parts[0].split(" ")[4]
                generated_summary = parts[1].strip()
                generated[image_filename] = generated_summary

# Load models and metrics
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
bleu = evaluate.load("bleu")
ter = evaluate.load("ter")
rouge = evaluate.load("rouge")

def compute_cosine_similarity(text1, text2):
    embedding_1 = sentence_model.encode(text1, convert_to_tensor=True)
    embedding_2 = sentence_model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding_1, embedding_2).item()

# Evaluation loop
bleu_scores = []
rougeL_scores = []
ter_scores = []
cosine_sim_scores = []

with open('/content/Vistext_score_Separate_schema_L2L3.txt', 'w') as output_file:
    for image_id, gen_caption in generated.items():
        gt_caption = ground_truth_combined.get(image_id)
        if not gt_caption:
            continue

        bleu_score = bleu.compute(predictions=[gen_caption], references=[gt_caption])['bleu']
        rougeL_score = rouge.compute(predictions=[gen_caption], references=[gt_caption])['rougeL']
        ter_score = ter.compute(predictions=[gen_caption], references=[gt_caption])['score']
        cosine_score = compute_cosine_similarity(gen_caption, gt_caption)

        bleu_scores.append(bleu_score)
        rougeL_scores.append(rougeL_score)
        ter_scores.append(ter_score)
        cosine_sim_scores.append(cosine_score)

        output_file.write(f"Image: {image_id}\n")
        output_file.write(f"Ground Truth: {gt_caption}\n")
        output_file.write(f"Generated: {gen_caption}\n")
        output_file.write(f"BLEU: {bleu_score:.4f}, ROUGE-L: {rougeL_score:.4f}, TER: {ter_score:.4f}, CosineSim: {cosine_score:.4f}\n\n")

    # Averages
    output_file.write(f"\nAverage BLEU: {sum(bleu_scores)/len(bleu_scores):.4f}\n")
    output_file.write(f"Average ROUGE-L: {sum(rougeL_scores)/len(rougeL_scores):.4f}\n")
    output_file.write(f"Average TER: {sum(ter_scores)/len(ter_scores):.4f}\n")
    output_file.write(f"Average Cosine Similarity: {sum(cosine_sim_scores)/len(cosine_sim_scores):.4f}\n")