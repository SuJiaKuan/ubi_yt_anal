import argparse
import os

from src.scoring import SupportScorer
from src.scoring import InfomationDepthScorer
from src.io import load_json
from src.io import save_json
from src.io import mkdir_p


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for scoring YouTube comments",
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
        default="output/scored.json",
        help="Path to output file",
    )

    args = parser.parse_args()

    return args


def run_scoring(support_scorer, info_depth_scorer, comment, comment_type="Outer"):
    comment_text = comment["content"]
    support_result = support_scorer.score(comment_text)
    info_depth_result = info_depth_scorer.score(comment_text)

    print("=========")
    print(f"{comment_type} Comment:")
    print(comment_text)
    print("Support Scoring:")
    print(f"- Score: {support_result.score}")
    print(f"- Detail: {support_result.detail}")
    print("Information Depth Scoring:")
    print(f"- Score: {info_depth_result.score}")
    print(f"- Detail: {info_depth_result.detail}")

    return {
        "support_scoring": {
            "score": support_result.score,
            "detail": support_result.detail,
        },
        "info_depth_scoring": {
            "score": info_depth_result.score,
            "detail": info_depth_result.detail,
        },
    }


def main(args):
    input_path = args.input
    output_path = args.output

    if os.path.dirname(output_path):
        mkdir_p(os.path.dirname(output_path))

    support_scorer = SupportScorer()
    info_depth_scorer = InfomationDepthScorer()

    comments = load_json(input_path)

    for outter_comment in comments:
        result = run_scoring(support_scorer, info_depth_scorer, outter_comment)
        outter_comment.update(result)

        for inner_comment in outter_comment["replies"]:
            # XXX (JiaKuanSu): Treat inner comments specially?
            result = run_scoring(
                support_scorer, info_depth_scorer, inner_comment, comment_type="Inner"
            )
            inner_comment.update(result)

    save_json(output_path, comments)

    print()
    print(f"Results saved in {output_path}")


if __name__ == "__main__":
    main(parse_args())
