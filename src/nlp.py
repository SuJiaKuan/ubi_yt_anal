from collections import Counter
from itertools import combinations

import jieba
import jieba.analyse
import networkx as nx
import numpy as np
from pyvis.network import Network
from wordcloud import WordCloud
from wordcloud import STOPWORDS


# Reference:
# https://tech.havocfuture.tw/blog/python-wordcloud-jieba
# https://hackmd.io/@aaronlife/python-bigdata-02
def generate_wordcloud(
    text,
    output_path,
    top_k=50,
    dictfile="data/nlp/dict.txt",
    stopfile="data/nlp/stopwords.txt",
    fontpath="data/nlp/msjh.ttc",
):
    jieba.set_dictionary(dictfile)
    jieba.analyse.set_stop_words(stopfile)

    tags = jieba.analyse.extract_tags(text, topK=top_k)

    seg_list = jieba.lcut(text, cut_all=False)
    dictionary = Counter(seg_list)

    freq = {}
    for ele in dictionary:
        if ele in tags:
            freq[ele] = dictionary[ele]

    wordcloud = WordCloud(
        background_color="white",
        margin=5,
        max_words=200,
        width=1280,
        height=720,
        font_path=fontpath,
    ).generate_from_frequencies(freq)

    wordcloud.to_file(output_path)


def generate_cooccurrence_graph(
    text_comments,
    output_path,
    dictfile="data/nlp/dict.txt",
    stopfile="data/nlp/stopwords.txt",
    max_nodes=300,
    max_edges=300,
    min_node_size=5,
    max_node_size=50,
):
    """
    Generate a co-occurrence graph from a list of text comments.

    Parameters:
    - text_comments (list of str): List of comments to process.
    - output_path (str): Path to save the generated HTML file.
    - dictfile (str): Path to the Jieba dictionary file.
    - stopfile (str): Path to the stopwords file.
    - max_nodes (int): Maximum number of nodes to retain.
    - max_edges (int): Maximum number of edges to retain.
    - min_node_size (int): Minimum node size.
    - max_node_size (int): Maximum node size.
    """

    # Load custom dictionary and stopwords
    jieba.set_dictionary(dictfile)
    with open(stopfile, "r", encoding="utf-8") as f:
        stopwords = set(line.strip() for line in f)

    # Tokenize and filter stopwords
    comments_tokenized = [
        [
            word
            for word in set(jieba.cut(comment))
            if (word not in stopwords) and (len(word) > 1)
        ]
        for comment in text_comments
    ]

    # Compute word co-occurrence
    co_occurrence = Counter()
    for sentence in comments_tokenized:
        for pair in combinations(sentence, 2):
            co_occurrence[tuple(sorted(pair))] += 1

    # Automatically filter low-weight edges
    all_weights = [freq for _, freq in co_occurrence.items()]
    threshold = np.percentile(all_weights, 75) if all_weights else 1

    # Build NetworkX graph
    G = nx.Graph()
    for (word1, word2), freq in co_occurrence.items():
        if freq >= threshold:
            G.add_edge(word1, word2, weight=freq)

    # Retain only the largest connected component
    if nx.is_connected(G):
        G_largest = G
    else:
        largest_cc = max(nx.connected_components(G), key=len, default=set())
        G_largest = G.subgraph(largest_cc).copy() if largest_cc else nx.Graph()

    # Select the most important nodes
    degree_centrality = nx.degree_centrality(G_largest)
    top_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[
        :max_nodes
    ]
    G_filtered = G_largest.subgraph(top_nodes).copy()

    # Limit the number of edges
    sorted_edges = sorted(
        G_filtered.edges(data=True), key=lambda x: x[2].get("weight", 1), reverse=True
    )
    G_final = nx.Graph()
    for i, (u, v, d) in enumerate(sorted_edges):
        if i < max_edges:
            G_final.add_edge(u, v, weight=d.get("weight", 1))

    # Remove empty nodes
    G_final = G_final.subgraph(
        [node for node in G_final.nodes() if node.strip()]
    ).copy()

    # Compute min/max edge weights
    if len(G_final.edges) > 0:
        min_weight = min(nx.get_edge_attributes(G_final, "weight").values())
        max_weight = max(nx.get_edge_attributes(G_final, "weight").values())
    else:
        min_weight = 1
        max_weight = 1

    # Compute Degree Centrality and set node colors
    degree_centrality_final = nx.degree_centrality(G_final)
    centrality_values = list(degree_centrality_final.values())
    high_centrality_threshold = np.percentile(centrality_values, 90)  # Top 10% → Red
    medium_centrality_threshold = np.percentile(
        centrality_values, 50
    )  # Top 50% → Orange

    node_colors = {}
    for node, centrality in degree_centrality_final.items():
        if centrality >= high_centrality_threshold:
            node_colors[node] = "red"
        elif centrality >= medium_centrality_threshold:
            node_colors[node] = "orange"
        else:
            node_colors[node] = "gray"

    # Generate the interactive network graph
    net = Network(width="1000px", height="800px", notebook=True, directed=False)

    net.set_options(
        """
    var options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.001,
          "springLength": 200,
          "springConstant": 0.05,
          "damping": 0.8
        }
      },
      "edges": {
        "smooth": false
      }
    }
    """
    )

    net.from_nx(G_final)

    # Compute min/max centrality values
    if len(degree_centrality_final) > 0:
        min_centrality = min(degree_centrality_final.values())
        max_centrality = max(degree_centrality_final.values())
    else:
        min_centrality = 0
        max_centrality = 1  # Avoid division by zero

    # Assign node sizes and colors
    for node in net.nodes:
        node_id = node["id"]
        centrality = degree_centrality_final.get(node_id, 0)
        degree = G_final.degree(node_id)

        # Normalize node size within min_node_size to max_node_size
        if max_centrality == min_centrality:
            node_size = (min_node_size + max_node_size) / 2
        else:
            node_size = min_node_size + (centrality - min_centrality) / (
                max_centrality - min_centrality
            ) * (max_node_size - min_node_size)

        node["color"] = {
            "background": node_colors.get(node_id, "gray"),
            "border": "black",
            "highlight": {
                "background": "yellow",
                "border": "red",
            },
        }
        node["size"] = node_size
        node["title"] = f"{node_id} (Degree: {degree})"

    # Assign edge properties
    for edge in net.edges:
        u, v = edge["from"], edge["to"]
        weight = G_final.get_edge_data(u, v).get("width")

        # Set transparency based on edge weight
        alpha = (
            0.2 + (weight - min_weight) / (max_weight - min_weight) * 0.6
        )  # Transparency range 0.2 ~ 0.8

        edge["color"] = {
            "color": f"rgba(100, 100, 100, {alpha})",
            "highlight": f"rgba(200, 50, 100, {alpha})",
        }

        normalized_width = (
            1
            if max_weight == min_weight
            else 1 + (weight - min_weight) / (max_weight - min_weight) * 5
        )
        edge["width"] = normalized_width
        edge["title"] = f"(Weight: {weight})"

    # Save the network graph to an HTML file
    net.show(output_path)


def count_word_frequencies(
    text_comments,
    output_path,
    dictfile="data/nlp/dict.txt",
    stopfile="data/nlp/stopwords.txt",
):
    """
    Tokenizes the comments, removes stop words, and saves word frequencies in descending order to a file.

    Parameters:
    - text_comments (list of str): List of comments to process.
    - output_path (str): Path to save the word frequency file.
    - dictfile (str): Path to the Jieba dictionary file.
    - stopfile (str): Path to the stopwords file.
    """

    # Load custom dictionary and stopwords
    jieba.set_dictionary(dictfile)
    with open(stopfile, "r", encoding="utf-8") as f:
        stopwords = set(line.strip() for line in f)

    # Tokenize and remove stopwords
    word_counts = Counter()
    for comment in text_comments:
        words = [
            word
            for word in jieba.cut(comment)
            if word not in stopwords and len(word.strip()) > 1
        ]
        word_counts.update(words)

    # Sort words by frequency in descending order
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    # Save to a text file
    with open(output_path, "w", encoding="utf-8") as f:
        for word, count in sorted_word_counts:
            f.write(f"{word} {count}\n")
