# -*- coding: utf-8 -*-

import re
import numpy as np

def parse_scores(path):
    metrics = {
        "BLEU": [],
        "ROUGE-L": [],
        "TER": [],
        "CosineSim": []
    }

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = r"BLEU:\s*([-\d.]+),\s*ROUGE-L:\s*([-\d.]+),\s*TER:\s*([-\d.]+),\s*CosineSim:\s*([-\d.]+)"

    matches = re.findall(pattern, text)

    for bleu, rouge, ter, cos in matches:

        metrics["BLEU"].append(float(bleu))
        metrics["ROUGE-L"].append(float(rouge))
        metrics["TER"].append(float(ter))
        metrics["CosineSim"].append(float(cos))

    return metrics

def eval_measure(scores):

    return np.mean(scores)

def eval_with_paired_bootstrap(
    sys1,
    sys2,
    metric_name,
    num_samples=10000,
    sample_ratio=1.0
):

    assert len(sys1) == len(sys2)

    sys1_scores = []
    sys2_scores = []

    wins = [0, 0, 0]

    n = len(sys1)

    ids = list(range(n))

    for _ in range(num_samples):

        reduced_ids = np.random.choice(
            ids,
            int(len(ids) * sample_ratio),
            replace=True
        )

        reduced_sys1 = [sys1[i] for i in reduced_ids]
        reduced_sys2 = [sys2[i] for i in reduced_ids]

        sys1_score = eval_measure(reduced_sys1)
        sys2_score = eval_measure(reduced_sys2)

        if metric_name == "TER":

            if sys1_score < sys2_score:
                wins[0] += 1

            elif sys2_score < sys1_score:
                wins[1] += 1

            else:
                wins[2] += 1

        else:

            if sys1_score > sys2_score:
                wins[0] += 1

            elif sys2_score > sys1_score:
                wins[1] += 1

            else:
                wins[2] += 1

        sys1_scores.append(sys1_score)
        sys2_scores.append(sys2_score)

    wins = [x / float(num_samples) for x in wins]
    print("\n" + "=" * 60)
    print(f"Metric: {metric_name}")
    print("=" * 60)

    print('Win ratio: sys1=%.3f, sys2=%.3f, tie=%.3f' % (wins[0], wins[1], wins[2]))

    if wins[0] > wins[1]:

        print(
            '(sys1 is superior with p value p=%.4f)\n'
#             % (1 - wins[0])
        )

    elif wins[1] > wins[0]:

        print(
            '(sys2 is superior with p value p=%.4f)\n'
#             % (1 - wins[1])
        )

    sys1_scores.sort()
    sys2_scores.sort()

    print('sys1 mean=%.4f, median=%.4f, 95%% CI=[%.4f, %.4f]'
        % (
            np.mean(sys1_scores),
            np.median(sys1_scores),
            sys1_scores[int(num_samples * 0.025)],
            sys1_scores[int(num_samples * 0.975)]
        ))

    print(
        'sys2 mean=%.4f, median=%.4f, 95%% CI=[%.4f, %.4f]'
        % (
            np.mean(sys2_scores),
            np.median(sys2_scores),
            sys2_scores[int(num_samples * 0.025)],
            sys2_scores[int(num_samples * 0.975)]
        )
    )

if __name__ == "__main__":
    sys1 = parse_scores("/content/ChartAltpilot_score_baseline_L1L2L3.txt")
    sys2 = parse_scores("/content/ChartAltpilot_score_Separate_schema_L1L2L3.txt")

    for metric in ["BLEU", "ROUGE-L", "TER", "CosineSim"]:

        eval_with_paired_bootstrap(
            sys1[metric],
            sys2[metric],
            metric_name=metric,
            num_samples=10000,
            sample_ratio=1.0
        )