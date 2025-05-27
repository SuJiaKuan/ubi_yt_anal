import argparse
import os
import traceback

from src.framing import ArgumentFramer
from src.io import load_json
from src.io import save_json
from src.io import mkdir_p


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for argument-based framing on YouTube comments",
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
        default="output/framed.json",
        help="Path to output file",
    )

    args = parser.parse_args()

    return args


def run_framing(argument_framer, comment, comment_type="Outer"):
    comment_text = comment["content"]
    try:
        argument_results = argument_framer.frame(comment_text)
    except Exception as e:
        print(f"Failed to frame comment: {comment_text}")
        print("Stack trace:")
        traceback.print_exc()
        argument_results = []

    print("=========")
    print(f"{comment_type} Comment:")
    print(comment_text)
    if not argument_results:
        print("No Related Argument Labels")
    else:
        print("Argument Framing:")
        for argument_result in argument_results:
            print(f"- {argument_result.label}: {argument_result.stance}")
            print(f"  {argument_result.reason}")

    return {
        "argument_framing": [
            {
                "label": argument_result.label,
                "stance": argument_result.stance,
                "reason": argument_result.reason,
            }
            for argument_result in argument_results
        ]
    }


def main(args):
    input_path = args.input
    output_path = args.output

    if os.path.dirname(output_path):
        mkdir_p(os.path.dirname(output_path))

    argument_framer = ArgumentFramer()

    comments = load_json(input_path)

    for outter_comment in comments:
        result = run_framing(argument_framer, outter_comment)
        outter_comment.update(result)

        for inner_comment in outter_comment["replies"]:
            # XXX (JiaKuanSu): Treat inner comments specially?
            result = run_framing(argument_framer, inner_comment, comment_type="Inner")

            inner_comment.update(result)

    save_json(output_path, comments)

    print()
    print(f"Results saved in {output_path}")


if __name__ == "__main__":
    main(parse_args())
