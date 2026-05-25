# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import hdbscan

file_path = "/content/Statista_top_500_words_bar.txt"

words = []
counts = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")

        if len(parts) == 1:
            word = parts[0].lower()
            count = 1
        else:
            word = parts[0].lower()
            count = int(parts[1])

        words.append(word)
        counts.append(count)

print(f"Loaded {len(words)} words")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(words, normalize_embeddings=True)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=3,
    min_samples=1,
    metric="euclidean",
    cluster_selection_method="eom"
)

labels = clusterer.fit_predict(embeddings)

clusters = defaultdict(list)

for w, c, l in zip(words, counts, labels):
    clusters[l].append((w, c))

# sort inside each cluster
for k in clusters:
    clusters[k] = sorted(clusters[k], key=lambda x: x[1], reverse=True)

for cid, items in clusters.items():
    print(f"\nCluster {cid} ({len(items)} items):")
    for w, c in items[:20]:
        print(f"{w}\t{c}")
output_file = "/content/bar_sbert_clusters_nopos.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for cid, items in clusters.items():
        f.write(f"\nCluster {cid} ({len(items)} items):\n")
        for w, c in items:
            f.write(f"{w}\t{c}\n")

print(f"\nSaved to: {output_file}")

#UMAP

import re
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import hdbscan

file_path = "/content/bar_sbert_clusters_nopos_selected.txt"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

cluster_pattern = r"Cluster\s+(-?\d+)\s+\((\d+)\s+items\):"

matches = list(re.finditer(cluster_pattern, text))

words = []
cluster_ids = []

for i, match in enumerate(matches):

    cluster_id = int(match.group(1))

    start = match.end()

    if i < len(matches) - 1:
        end = matches[i + 1].start()
    else:
        end = len(text)

    block = text[start:end].strip()

    lines = block.split("\n")

    for line in lines:

        parts = line.strip().split("\t")

        if len(parts) != 2:
            continue

        word = parts[0]

        words.append(word)
        cluster_ids.append(cluster_id)

print(f"Loaded {len(words)} words")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    words,
    normalize_embeddings=True
)

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

umap_embeddings = reducer.fit_transform(embeddings)
unique_clusters = sorted(list(set(cluster_ids)))
cmap = plt.cm.get_cmap("tab20", len(unique_clusters))

cluster_to_color = {
    cid: cmap(i)
    for i, cid in enumerate(unique_clusters)
}

colors = [cluster_to_color[c] for c in cluster_ids]

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    umap_embeddings[:, 0],
    umap_embeddings[:, 1],
    c=colors,
    s=50,
    alpha=0.8
)


for i, word in enumerate(words):

    plt.text(
        umap_embeddings[i, 0],
        umap_embeddings[i, 1],
        word,
        fontsize=8,
        alpha=0.9
    )
handles = []

for cid in unique_clusters:

    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markerfacecolor=cluster_to_color[cid],
            markersize=15,
            label=f"Cluster {cid}"
        )
    )

plt.legend(
    handles=handles,
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)


plt.title(
    "UMAP Semantic Space of Bar Chart Clusters",
    fontsize=14
)

plt.xlabel("UMAP Dimension 1", fontsize=14)
plt.ylabel("UMAP Dimension 2", fontsize=14)

plt.tight_layout()

plt.show()