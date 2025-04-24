import argparse
import os
from collections import Counter
from collections import defaultdict
from itertools import combinations


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from src.const import TOPIC_MAJOR_TAG
from src.const import TOPIC_MINOR_TAG
from src.const import ARGUMENT_STANCE_CATEGORY
from src.io import load_json
from src.io import save_json
from src.io import mkdir_p
from src.nlp import generate_wordcloud
from src.nlp import generate_cooccurrence_graph
from src.nlp import count_word_frequencies

FIG_DPI = 300

TOPIC_MINOR_TAGS_LIST = [
    TOPIC_MINOR_TAG.ROLE_OF_GOVERNMENT.value,
    TOPIC_MINOR_TAG.COMPARISON_WITH_SOCIAL_POLICIES.value,
    TOPIC_MINOR_TAG.IDEOLOGIES.value,
    TOPIC_MINOR_TAG.FEASIBILITY_AND_GOVERNANCE.value,
    TOPIC_MINOR_TAG.INFLATION_AND_COST_OF_LIVING.value,
    TOPIC_MINOR_TAG.TAXATION_AND_BUDGETING.value,
    TOPIC_MINOR_TAG.LABOR_MARKET_AND_EMPLOYMENT.value,
    TOPIC_MINOR_TAG.ECONOMIC_GROWTH_AND_PRODUCTIVITY.value,
    TOPIC_MINOR_TAG.POVERTY_AND_WEALTH_DISTRIBUTION.value,
    TOPIC_MINOR_TAG.WORK_ETHIC_AND_MOTIVATION.value,
    TOPIC_MINOR_TAG.SOCIAL_STABILITY_AND_CRIME_RATE.value,
    TOPIC_MINOR_TAG.MENTAL_HEALTH_AND_WELL_BEING.value,
    TOPIC_MINOR_TAG.EQUALITY_AND_FAIRNESS.value,
    TOPIC_MINOR_TAG.TECHNOLOGY_AND_THE_FUTURE.value,
    TOPIC_MINOR_TAG.HUMAN_NATURE_AND_BEHAVIOR.value,
    TOPIC_MINOR_TAG.FREEDOM_VS_DEPENDENCY.value,
]


TOPIC_MAJOR_TAG_TO_MINOR_TAGS = {
    TOPIC_MAJOR_TAG.POLITICS.value: [
        TOPIC_MINOR_TAG.ROLE_OF_GOVERNMENT.value,
        TOPIC_MINOR_TAG.COMPARISON_WITH_SOCIAL_POLICIES.value,
        TOPIC_MINOR_TAG.IDEOLOGIES.value,
        TOPIC_MINOR_TAG.FEASIBILITY_AND_GOVERNANCE.value,
    ],
    TOPIC_MAJOR_TAG.ECONOMICS.value: [
        TOPIC_MINOR_TAG.INFLATION_AND_COST_OF_LIVING.value,
        TOPIC_MINOR_TAG.TAXATION_AND_BUDGETING.value,
        TOPIC_MINOR_TAG.LABOR_MARKET_AND_EMPLOYMENT.value,
        TOPIC_MINOR_TAG.ECONOMIC_GROWTH_AND_PRODUCTIVITY.value,
    ],
    TOPIC_MAJOR_TAG.SOCIETY.value: [
        TOPIC_MINOR_TAG.POVERTY_AND_WEALTH_DISTRIBUTION.value,
        TOPIC_MINOR_TAG.WORK_ETHIC_AND_MOTIVATION.value,
        TOPIC_MINOR_TAG.SOCIAL_STABILITY_AND_CRIME_RATE.value,
        TOPIC_MINOR_TAG.MENTAL_HEALTH_AND_WELL_BEING.value,
    ],
    TOPIC_MAJOR_TAG.PHILOSOPHY_AND_ETHICS.value: [
        TOPIC_MINOR_TAG.EQUALITY_AND_FAIRNESS.value,
        TOPIC_MINOR_TAG.TECHNOLOGY_AND_THE_FUTURE.value,
        TOPIC_MINOR_TAG.HUMAN_NATURE_AND_BEHAVIOR.value,
        TOPIC_MINOR_TAG.FREEDOM_VS_DEPENDENCY.value,
    ],
}


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


def analyze_all(flatten_comments, output_dir):
    text_comments = [comment["content"] for comment in flatten_comments]

    count_word_frequencies(text_comments, os.path.join(output_dir, "all_word_freq.txt"))
    generate_cooccurrence_graph(
        text_comments, os.path.join(output_dir, "all_cooccur.html")
    )


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

    ### Distribution for support scores ###

    plt.figure(figsize=(8, 5))
    sns.histplot(
        df["support_score"], bins=np.arange(1, 12) - 0.5, kde=True, color="blue"
    )
    plt.xticks(range(1, 11))
    plt.xlabel("Support Score (1-10)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Support Scores")
    plt.savefig(os.path.join(output_dir, "support_dist.png"), dpi=FIG_DPI)
    plt.close()

    ### Distribution for information depth scores ###

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
    plt.savefig(os.path.join(output_dir, "info_depth_dist.png"), dpi=FIG_DPI)
    plt.close()

    ### Heatmap for support vs information depth scores ###

    plt.figure(figsize=(8, 6))
    heatmap_data = df.pivot_table(
        index="info_depth_score",
        columns="support_score",
        aggfunc=len,
        fill_value=0,
    )
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
    plt.savefig(
        os.path.join(output_dir, "support_vs_info_depth_heatmap.png"), dpi=FIG_DPI
    )
    plt.close()
    plt.xlabel("Support Score (1-10)")
    plt.ylabel("Information Depth Score (1-10)")
    plt.title("Heatmap: Support Score vs Information Depth Score")

    ### Group by support and information depth scores and generate wordclouds & word co-occurence graphs ###

    group_names = ["hh", "hl", "lh", "ll"]
    comment_texts_group = {group_name: [] for group_name in group_names}
    for comment in flatten_comments:
        comment_text = comment["content"]
        support_score = comment["support_scoring"]["score"]
        info_depth_score = comment["info_depth_scoring"]["score"]

        if support_score >= 7 and info_depth_score >= 7:
            comment_texts_group["hh"].append(comment_text)

        if support_score >= 7 and info_depth_score <= 4:
            comment_texts_group["hl"].append(comment_text)

        if support_score <= 4 and info_depth_score >= 7:
            comment_texts_group["lh"].append(comment_text)

        if support_score <= 4 and info_depth_score <= 4:
            comment_texts_group["ll"].append(comment_text)

    for group_name, comment_texts in comment_texts_group.items():
        save_json(
            os.path.join(output_dir, f"support_vs_info_depth_{group_name}.json"),
            comment_texts,
        )
        generate_wordcloud(
            "".join(comment_texts),
            os.path.join(
                output_dir, f"support_vs_info_depth_{group_name}_wordcloud.png"
            ),
        )
        count_word_frequencies(
            comment_texts,
            os.path.join(output_dir, f"support_vs_info_depth_{group_name}_cooccur.txt"),
        )
        generate_cooccurrence_graph(
            comment_texts,
            os.path.join(
                output_dir, f"support_vs_info_depth_{group_name}_cooccur.html"
            ),
        )


def analyze_tagging(flatten_comments, output_dir):
    ### Major and minor tags distribution ###
    tags_lst = [comment["topic_tagging"]["tags"] for comment in flatten_comments]
    df = pd.DataFrame({"tags": tags_lst})
    df_exploded = df.explode("tags")

    tag_counts = df_exploded["tags"].value_counts()
    category_counts = {
        category: {sub: 0 for sub in subs}
        for category, subs in TOPIC_MAJOR_TAG_TO_MINOR_TAGS.items()
    }

    for sub_tag, count in tag_counts.items():
        for category, subs in TOPIC_MAJOR_TAG_TO_MINOR_TAGS.items():
            if sub_tag in subs:
                category_counts[category][sub_tag] = count

    df_stacked = pd.DataFrame(category_counts)

    df_stacked.plot(kind="bar", stacked=True, figsize=(12, 8), colormap="viridis")
    plt.xlabel("Minor Tags")
    plt.ylabel("Number of Comments")
    plt.title("Topic Major and Minor Tags Distribution")
    plt.legend(title="Major Tag")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "topic_dist.png"), dpi=FIG_DPI)
    plt.close()

    ### Group by major tags and generate wordclouds & co-occurence graph ###

    for major_tag, minor_tags in TOPIC_MAJOR_TAG_TO_MINOR_TAGS.items():
        comment_texts = []
        for comment in flatten_comments:
            for minor_tag in minor_tags:
                if minor_tag in comment["topic_tagging"]["tags"]:
                    comment_texts.append(comment["content"])
                    break

        save_json(os.path.join(output_dir, f"topic_{major_tag}.json"), comment_texts)
        generate_wordcloud(
            "".join(comment_texts),
            os.path.join(output_dir, f"topic_{major_tag}_wordcloud.png"),
        )
        count_word_frequencies(
            comment_texts,
            os.path.join(output_dir, f"topic_{major_tag}_cooccur.txt"),
        )
        generate_cooccurrence_graph(
            comment_texts,
            os.path.join(output_dir, f"topic_{major_tag}_cooccur.html"),
        )


def analyze_cross(flatten_comments, output_dir):
    tags_lst = [comment["topic_tagging"]["tags"] for comment in flatten_comments]
    support_scores = [
        comment["support_scoring"]["score"] for comment in flatten_comments
    ]
    info_depth_scores = [
        comment["info_depth_scoring"]["score"] for comment in flatten_comments
    ]
    like_counts = [comment["likes"] for comment in flatten_comments]
    reply_counts = [
        len(comment["replies"]) if "replies" in comment else 0
        for comment in flatten_comments
    ]

    df = pd.DataFrame(
        {
            "support_score": support_scores,
            "info_depth_score": info_depth_scores,
            "like_count": like_counts,
            "reply_count": reply_counts,
            "tags": tags_lst,
        }
    )

    df_exploded = df.explode("tags")

    ### Heatmap for topic tagging vs support scores / information depth scores ###

    for score_type in ["support_score", "info_depth_score"]:
        plt.figure(figsize=(12, 8))
        heatmap_data = (
            df_exploded.groupby(["tags", score_type]).size().unstack(fill_value=0)
        )
        heatmap_data_ordered = (
            df_exploded.groupby(["tags", score_type])
            .size()
            .unstack(fill_value=0)
            .reindex(TOPIC_MINOR_TAGS_LIST)
        )
        sns.heatmap(
            heatmap_data_ordered, cmap="coolwarm", linewidths=0.5, annot=True, fmt="d"
        )

        plt.xlabel(f"{score_type.title()} (1-10)")
        plt.ylabel("Tags")
        plt.title(f"Heatmap: Topic Tags vs {score_type.title()}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir, f"topic_vs_{score_type.replace('_score', '')}_heatmap.png"
            ),
            dpi=FIG_DPI,
        )
        plt.close()

    ### Scatter plot for support scores vs like count / reply count ###

    plt.figure(figsize=(12, 6))

    df_filtered = df[df["like_count"] > 0]
    sns.scatterplot(
        data=df_filtered,
        x="support_score",
        y="like_count",
        marker="x",
        alpha=0.7,
        color="blue",
    )
    df_filtered = df[df["reply_count"] > 0]
    sns.scatterplot(
        data=df_filtered,
        x="support_score",
        y="reply_count",
        marker="x",
        alpha=0.7,
        color="red",
    )

    plt.xlabel("Support Score (1-10)")
    plt.ylabel("Count")
    plt.title("Support Score vs Like Count & Reply Count")
    plt.legend(title="Metrics", labels=["Like Count", "Reply Count"])
    plt.savefig(
        os.path.join(output_dir, "support_vs_like_reply_count.png"), dpi=FIG_DPI
    )
    plt.close()


def count_framing_freq(flatten_comments, output_path):
    # Count the frequency of each argument label and stance
    argument_counts = Counter()
    for comment in flatten_comments:
        for framed_argument in comment["argument_framing"]:
            label = framed_argument["label"]
            stance = framed_argument["stance"]
            argument_counts[(label, stance)] += 1

    # Convert to DataFrame for plotting
    arg_df = pd.DataFrame(
        [
            {"label": label, "stance": stance, "count": count}
            for (label, stance), count in argument_counts.items()
        ]
    )

    # Pivot to get pro and con counts for each label
    pivot_df = arg_df.pivot(index="label", columns="stance", values="count").fillna(0)

    # Ensure both 'Pro' and 'Con' columns exist
    for col in [ARGUMENT_STANCE_CATEGORY.PRO.value, ARGUMENT_STANCE_CATEGORY.CON.value]:
        if col not in pivot_df.columns:
            pivot_df[col] = 0

    # Sort by total frequency
    pivot_df["Total"] = (
        pivot_df[ARGUMENT_STANCE_CATEGORY.PRO.value]
        + pivot_df[ARGUMENT_STANCE_CATEGORY.CON.value]
    )
    pivot_df = pivot_df.sort_values("Total", ascending=False)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.barh(
        pivot_df.index,
        pivot_df[ARGUMENT_STANCE_CATEGORY.PRO.value],
        color="mediumseagreen",
        label="pro",
    )
    ax.barh(
        pivot_df.index,
        -pivot_df[ARGUMENT_STANCE_CATEGORY.CON.value],
        color="indianred",
        label="con",
    )

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Argument Label")
    ax.legend()
    plt.tight_layout()
    plt.gca().invert_yaxis()

    plt.savefig(output_path, dpi=FIG_DPI)
    plt.close()


def plot_radial_concept_network(
    G: nx.Graph,
    centrality: dict,
    pos_radial: dict,
    edge_colors: list,
    edge_widths: list,
    num_shells: int,
    output_path: str,
    figsize=(12, 12),
    alpha_range=(0.01, 0.9),
):
    """
    Plot a radial concept network with edge transparency varying by weight.
    """
    plt.figure(figsize=figsize)

    node_sizes = [centrality[n] * 2500 for n in G.nodes()]

    # Transparency calculation
    edge_alphas = []
    min_width = min(edge_widths)
    max_width = max(edge_widths)
    for width in edge_widths:
        if max_width == min_width:
            alpha = sum(alpha_range) / 2
        else:
            alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * (
                (width - min_width) / (max_width - min_width)
            )
        edge_alphas.append(alpha)

    # Draw graph
    nx.draw_networkx_nodes(G, pos_radial, node_color="lightgray", node_size=node_sizes)
    nx.draw_networkx_labels(G, pos_radial, font_size=8)
    for (u, v), color, width, alpha in zip(
        G.edges(), edge_colors, edge_widths, edge_alphas
    ):
        nx.draw_networkx_edges(
            G, pos_radial, edgelist=[(u, v)], edge_color=color, width=width, alpha=alpha
        )

    # Draw concentric green rings
    for r in range(1, num_shells + 1):
        circle = plt.Circle(
            (0, 0),
            r * 1.5,
            color="mediumseagreen",
            fill=False,
            linestyle="--",
            alpha=0.2,
        )
        plt.gca().add_patch(circle)

    plt.axis("off")
    plt.savefig(output_path, dpi=FIG_DPI)
    plt.close()


def build_and_plot_concept_network(
    comments: list, output_path: str, threshold: float = 0.03, num_shells: int = 5
):
    """
    Build and visualize a radial concept co-occurrence network from annotated comments.
    """
    # Step 1: Co-occurrence counts
    total_comments = len(comments)
    cooccur_counts = {}
    for entry in comments:
        labels_by_stance = {"Pro": [], "Con": []}
        for item in entry["argument_framing"]:
            labels_by_stance[item["stance"]].append(item["label"])
        for stance, label_list in labels_by_stance.items():
            for pair in combinations(sorted(set(label_list)), 2):
                key = (pair[0], pair[1], stance)
                cooccur_counts[key] = cooccur_counts.get(key, 0) + 1

    # Step 2: Normalize and filter
    normalized_counts = defaultdict(float)
    for (label1, label2, stance), count in cooccur_counts.items():
        normalized_counts[(label1, label2, stance)] = count / total_comments

    G = nx.Graph()
    for (label1, label2, stance), norm_value in normalized_counts.items():
        if norm_value < threshold:
            continue
        if G.has_edge(label1, label2):
            G[label1][label2][stance.lower()] += norm_value
        else:
            G.add_edge(
                label1,
                label2,
                pro=norm_value if stance == "Pro" else 0,
                con=norm_value if stance == "Con" else 0,
            )

    # Step 3: Centrality and edge style
    centrality = nx.degree_centrality(G)
    edges = G.edges()
    edge_colors = []
    edge_widths = []
    for u, v in edges:
        pro = G[u][v]["pro"]
        con = G[u][v]["con"]
        total = pro + con
        edge_widths.append(total * 30)
        edge_colors.append(
            "mediumseagreen" if pro > con else "indianred" if con > pro else "gray"
        )

    # Step 4: Radial layout with concentric logic
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    top_node = sorted_nodes[0][0]
    remaining_nodes = [n[0] for n in sorted_nodes[1:]]

    shell_counts = [1, 6, 10, 12]
    layered_nodes = [[] for _ in range(num_shells)]
    current_index = 0
    for layer in range(num_shells - 1):
        count = shell_counts[layer] if layer < len(shell_counts) else 0
        layered_nodes[layer] = remaining_nodes[current_index : current_index + count]
        current_index += count
    layered_nodes[num_shells - 1] = remaining_nodes[current_index:]

    pos_radial = {top_node: (0, 0)}
    for layer_idx, nodes in enumerate(layered_nodes):
        radius = (layer_idx + 1) * 1.5
        angle_step = 2 * np.pi / len(nodes) if nodes else 1
        angle_offset = (layer_idx % 2) * (angle_step / 2)
        for i, node in enumerate(nodes):
            angle = i * angle_step + angle_offset
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            pos_radial[node] = (x, y)

    # Step 5: Plot
    plot_radial_concept_network(
        G,
        centrality,
        pos_radial,
        edge_colors,
        edge_widths,
        num_shells,
        output_path,
    )


def analyze_framing(flatten_comments, output_dir):
    count_framing_freq(flatten_comments, os.path.join(output_dir, "framing_freq.png"))
    build_and_plot_concept_network(
        flatten_comments, os.path.join(output_dir, "framing_concept_network.png")
    )


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

    comment_texts = [comment["content"] for comment in flatten_comments]

    analyze_all(flatten_comments, output_dir)

    stance_cluster = cluster_by_stance(flatten_comments)
    stance_result = analyze_stance(stance_cluster, output_dir)
    generate_stance_wordclouds(stance_result, output_dir)

    analyze_scoring(flatten_comments, output_dir)
    analyze_tagging(flatten_comments, output_dir)
    analyze_cross(flatten_comments, output_dir)
    analyze_framing(flatten_comments, output_dir)

    print(f"Results saved in {output_dir}")


if __name__ == "__main__":
    main(parse_args())
