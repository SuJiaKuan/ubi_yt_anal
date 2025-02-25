import argparse
import os

from src.scoring import SupportScorer
from src.scoring import SentimentScorer
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


def run_scoring(support_scorer, sentiment_scorer, comment, comment_type="Outer"):
    comment_text = comment["content"]
    support_result = support_scorer.score(comment_text)
    sentiment_result = sentiment_scorer.score(comment_text)

    print("=========")
    print(f"{comment_type} Comment:")
    print(comment_text)
    print("Support Scoring:")
    print(f"- Score: {support_result.score}")
    print(f"- Detail: {support_result.detail}")
    print("Sentiment Scoring:")
    print(f"- Score: {sentiment_result.score}")
    print(f"- Detail: {sentiment_result.detail}")

    return {
        "support_scoring": {
            "score": support_result.score,
            "detail": support_result.detail,
        },
        "sentiment_scoring": {
            "score": sentiment_result.score,
            "detail": sentiment_result.detail,
        },
    }


def main(args):
    input_path = args.input
    output_path = args.output

    if os.path.dirname(output_path):
        mkdir_p(os.path.dirname(output_path))

    support_scorer = SupportScorer()
    sentiment_scorer = SentimentScorer()

    comments = load_json(input_path)

    for outter_comment in comments:
        result = run_scoring(support_scorer, sentiment_scorer, outter_comment)
        outter_comment.update(result)

        for inner_comment in outter_comment["replies"]:
            # XXX (JiaKuanSu): Treat inner comments specially?
            result = run_scoring(
                support_scorer, sentiment_scorer, inner_comment, comment_type="Inner"
            )
            inner_comment.update(result)

    save_json(output_path, comments)

    print()
    print(f"Results saved in {output_path}")


if __name__ == "__main__":
    main(parse_args())
