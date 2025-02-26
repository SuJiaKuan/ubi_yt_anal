import argparse
import os

from src.tagging import TopicTagger
from src.io import load_json
from src.io import save_json
from src.io import mkdir_p


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for tagging YouTube comments",
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
        default="output/tagged.json",
        help="Path to output file",
    )

    args = parser.parse_args()

    return args


def run_tagging(topic_tagger, comment, comment_type="Outer"):
    comment_text = comment["content"]
    topic_result = topic_tagger.tag(comment_text)

    print("=========")
    print(f"{comment_type} Comment:")
    print(comment_text)
    print("Topic Tagging:")
    print(f"- Tags: {topic_result.tags}")
    print(f"- Detail: {topic_result.detail}")

    return {
        "topic_tagging": {
            "tags": topic_result.tags,
            "detail": topic_result.detail,
        }
    }


def main(args):
    input_path = args.input
    output_path = args.output

    if os.path.dirname(output_path):
        mkdir_p(os.path.dirname(output_path))

    topic_tagger = TopicTagger()

    comments = load_json(input_path)

    for outter_comment in comments:
        result = run_tagging(topic_tagger, outter_comment)
        outter_comment.update(result)

        for inner_comment in outter_comment["replies"]:
            # XXX (JiaKuanSu): Treat inner comments specially?
            result = run_tagging(topic_tagger, inner_comment, comment_type="Inner")
            inner_comment.update(result)

    save_json(output_path, comments)

    print()
    print(f"Results saved in {output_path}")


if __name__ == "__main__":
    main(parse_args())
