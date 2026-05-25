# -*- coding: utf-8 -*-

import json
from collections import Counter
import re

file_path = "/content/filtered_metadata.json"
output_path = "/content/statista_top_500_words_bar.txt"

word_counter = Counter()

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    chart_type = item["chart_type"].strip().lower()

    if chart_type == "bar":
        caption = item["caption"].lower()
        words = re.findall(r"\b\w+\b", caption)
        word_counter.update(words)

# Get top 500
top_500 = word_counter.most_common(500)

# Save to file
with open(output_path, "w", encoding="utf-8") as f:
    for word, count in top_500:
        f.write(f"{word}\t{count}\n")

print(f"Saved top 500 words to: {output_path}")