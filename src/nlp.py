from collections import Counter

import jieba


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
