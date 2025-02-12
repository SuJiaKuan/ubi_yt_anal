import argparse
import os

from src.stance_cls import StanceClassifier
from src.io import load_json
from src.io import save_json
from src.io import mkdir_p


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
        default="output/classified.json",
        help="Path to output file",
    )

    args = parser.parse_args()

    return args


def run_classification(stance_classifier, comment, comment_type="Outer"):
    comment_text = comment["content"]
    stance_result = stance_classifier._classify(comment_text)

    print("=========")
    print(f"{comment_type} Comment:")
    print(comment_text)
    print("Stance Classification:")
    print(f"- Major Category: {stance_result.major_category}")
    print(f"- Minor Category: {stance_result.minor_category}")
    print(f"- Detail: {stance_result.detail}")

    return {
        "stance_cls": {
            "major_category": stance_result.major_category,
            "minor_category": stance_result.minor_category,
            "detail": stance_result.detail,
        }
    }


def main(args):
    input_path = args.input
    output_path = args.output

    if os.path.dirname(output_path):
        mkdir_p(os.path.dirname(output_path))

    stance_classifier = StanceClassifier()

    comments = load_json(input_path)

    for outter_comment in comments:
        result = run_classification(stance_classifier, outter_comment)
        outter_comment.update(result)

        for inner_comment in outter_comment["replies"]:
            # XXX (JiaKuanSu): Treat inner comments specially?
            result = run_classification(stance_classifier, inner_comment)
            inner_comment.update(result)

    save_json(output_path, comments)

    print()
    print(f"Results saved in {output_path}")


if __name__ == "__main__":
    main(parse_args())
