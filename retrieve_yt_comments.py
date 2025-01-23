import argparse
import json
import os

from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for retrieving YouTube comments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=str,
        help="The input YouTube video ID",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="yt_comments.json",
        help="Path to output file",
    )

    args = parser.parse_args()

    return args


def get_all_replies(youtube, parent_id):
    """Retrieves all replies to a comment."""
    replies = []
    next_page_token = None
    while True:
        results = (
            youtube.comments()
            .list(
                part="snippet",
                parentId=parent_id,
                textFormat="plainText",
                pageToken=next_page_token,
                maxResults=100,
            )
            .execute()
        )

        for item in results.get("items", []):  # Handle cases where there are no replies
            reply_snippet = item["snippet"]
            replies.append(
                {
                    "author": reply_snippet["authorDisplayName"],
                    "content": reply_snippet["textDisplay"],
                    "likes": reply_snippet["likeCount"],
                    "dislikes": reply_snippet.get("dislikeCount", 0),
                    "date": reply_snippet["publishedAt"],
                }
            )
        next_page_token = results.get("nextPageToken")
        if not next_page_token:
            break
    return replies


def get_video_comments(video_id, max_results=10000):
    """Downloads comments from a YouTube video.

    Args:
        video_id: The ID of the YouTube video.
        max_results: The maximum number of comments to retrieve.

    Returns:
        A list of JSON objects, where each object represents a comment.
        Returns None if there's an error or no comments are found.
    """
    try:
        youtube = build(
            YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=GOOGLE_API_KEY
        )

        comments = []
        next_page_token = None
        total_results = 0

        while True:
            results = (
                youtube.commentThreads()
                .list(
                    part="snippet,replies",
                    videoId=video_id,
                    textFormat="plainText",  # Get plain text comments
                    pageToken=next_page_token,
                    maxResults=min(max_results - total_results, 100),  # API max is 100
                )
                .execute()
            )

            for item in results["items"]:
                comment = {}
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comment["author"] = snippet["authorDisplayName"]
                comment["content"] = snippet["textDisplay"]
                comment["likes"] = snippet["likeCount"]
                comment["dislikes"] = snippet.get(
                    "dislikeCount", 0
                )  # dislikeCount is not always present
                comment["date"] = snippet["publishedAt"]

                comment["replies"] = get_all_replies(
                    youtube, item["snippet"]["topLevelComment"]["id"]
                )
                comments.append(comment)
                total_results += 1
                if total_results >= max_results:
                    break
            next_page_token = results.get("nextPageToken")
            if not next_page_token or total_results >= max_results:
                break
        return comments if comments else None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def main(args):
    video_id = args.input
    output_path = args.output

    comments = get_video_comments(video_id)

    if comments:
        num_comments = len(comments) + sum(
            [len(comment["replies"]) for comment in comments]
        )
        print(f"Found {num_comments} comments (including replies).")

        # Save to a JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comments, f, indent=4, ensure_ascii=False)
        print(f"Comments saved to {output_path}")
    else:
        print("Could not retrieve comments.")


if __name__ == "__main__":
    main(parse_args())
