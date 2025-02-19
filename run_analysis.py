import argparse
import os

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


def cluster_by_stance(comments) -> dict:
    flatten_comments = []
    for outter_comment in comments:
        flatten_comments.append(outter_comment)
        for inner_comment in outter_comment["replies"]:
            flatten_comments.append(inner_comment)

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


def main(args):
    input_path = args.input
    output_dir = args.output

    mkdir_p(output_dir)

    comments = load_json(input_path)

    stance_cluster = cluster_by_stance(comments)
    stance_result = analyze_stance(stance_cluster, output_dir)
    generate_stance_wordclouds(stance_result, output_dir)

    print(f"Results saved in {output_dir}")


if __name__ == "__main__":
    main(parse_args())
