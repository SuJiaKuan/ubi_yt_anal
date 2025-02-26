import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.const import STANCE_MAJOR_CATEGORY
from src.const import STANCE_MINOR_CATEGORY
from src.io import load_json
from src.io import save_json
from src.io import mkdir_p
from src.wordcloud import generate_wordcloud


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for analyzing YouTube comments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=str,
        help="The input data path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output",
        help="Path to output folder",
    )

    args = parser.parse_args()

    return args


def cluster_by_stance(flatten_comments) -> dict:
    stance_cluster = {}
    for comment in flatten_comments:
        major_category = comment["stance_cls"]["major_category"]
        minor_category = comment["stance_cls"]["minor_category"]

        if major_category not in stance_cluster:
            stance_cluster[major_category] = {}
        if minor_category not in stance_cluster[major_category]:
            stance_cluster[major_category][minor_category] = []

        stance_cluster[major_category][minor_category].append(comment["content"])

    return stance_cluster


def analyze_stance(stance_cluster, output_dir):
    stance_result = {}
    for major_category, minor_dict in stance_cluster.items():
        stance_result[major_category] = {
            "count": 0,
            "minor": {},
        }
        for minor_category, comments in stance_cluster[major_category].items():
            stance_result[major_category]["minor"][minor_category] = {
                "count": len(comments),
                "comments": comments,
            }
            stance_result[major_category]["count"] += len(comments)

    save_json(os.path.join(output_dir, "stance_count.json"), stance_result)

    return stance_result


def generate_stance_wordclouds(stance_result, output_dir):
    overall_text = ""
    for major_category, major_dict in stance_result.items():
        major_text = ""
        for minor_dict in major_dict["minor"].values():
            major_text += "".join(minor_dict["comments"])

        generate_wordcloud(
            major_text,
            os.path.join(output_dir, f"stance_wordcloud_{major_category}.png"),
        )

        overall_text += major_text

    generate_wordcloud(
        overall_text, os.path.join(output_dir, "stance_wordcloud_overall.png")
    )


def analyze_scoring(flatten_comments, output_dir):
    support_scores = [
        comment["support_scoring"]["score"] for comment in flatten_comments
    ]
    info_depth_scores = [
        comment["info_depth_scoring"]["score"] for comment in flatten_comments
    ]

    df = pd.DataFrame(
        {"support_score": support_scores, "info_depth_score": info_depth_scores}
    )

    sns.set_style("whitegrid")

    # Distribution for support scores
    plt.figure(figsize=(8, 5))
    sns.histplot(
        df["support_score"], bins=np.arange(1, 12) - 0.5, kde=True, color="blue"
    )
    plt.xticks(range(1, 11))
    plt.xlabel("Support Score (1-10)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Support Scores")
    plt.savefig(os.path.join(output_dir, "support_dist.png"))

    # Distribution for information depth scores
    plt.figure(figsize=(8, 5))
    sns.histplot(
        df["info_depth_score"],
        bins=np.arange(1, 12) - 0.5,
        kde=True,
        color="green",
    )
    plt.xticks(range(1, 11))
    plt.xlabel("Information Depth Score (1-10)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Information Depth Scores")
    plt.savefig(os.path.join(output_dir, "info_depth_dist.png"))

    # Heatmap for support vs information depth scores
    plt.figure(figsize=(8, 6))
    heatmap_data = df.pivot_table(
        index="info_depth_score",
        columns="support_score",
        aggfunc=len,
        fill_value=0,
    )
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
    plt.savefig(os.path.join(output_dir, "support_vs_info_depth_heatmap.png"))

    plt.xlabel("Support Score (1-10)")
    plt.ylabel("Information Depth Score (1-10)")
    plt.title("Heatmap: Support Score vs Information Depth Score")


def main(args):
    input_path = args.input
    output_dir = args.output

    mkdir_p(output_dir)

    comments = load_json(input_path)

    flatten_comments = []
    for outter_comment in comments:
        flatten_comments.append(outter_comment)
        for inner_comment in outter_comment["replies"]:
            flatten_comments.append(inner_comment)

    stance_cluster = cluster_by_stance(flatten_comments)
    stance_result = analyze_stance(stance_cluster, output_dir)
    generate_stance_wordclouds(stance_result, output_dir)

    analyze_scoring(flatten_comments, output_dir)

    print(f"Results saved in {output_dir}")


if __name__ == "__main__":
    main(parse_args())
