#!/usr/bin/env python3
"""TF-IDF embedding matrix"""

import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use (if None, use all words)

    Returns:
        embeddings: numpy.ndarray of shape (s, f)
        features: numpy.ndarray of features used for embeddings
    """
    # Tokenize sentences
    tokenized = []
    for sentence in sentences:
        clean = sentence.lower()
        # remove possessives e.g. children's
        clean = re.sub(r"'s\b", "", clean)
        clean = re.sub(r"[^a-zA-Z ]", " ", clean)
        tokenized.append(clean.split())

    # Build vocabulary
    if vocab is None:
        all_words = set()
        for words in tokenized:
            all_words.update(words)
        features = sorted(list(all_words))
    else:
        features = list(vocab)

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f))

    feature_index = {word: i for i, word in enumerate(features)}

    for i, words in enumerate(tokenized):
        # --- TF: count occurrences / total words in sentence ---
        tf = np.zeros(f)
        for word in words:
            if word in feature_index:
                tf[feature_index[word]] += 1
        if len(words) > 0:
            tf = tf / len(words)

        # --- IDF: log((1 + s) / (1 + df)) + 1 ---
        idf = np.zeros(f)
        for j, feat in enumerate(features):
            df = sum(1 for doc in tokenized if feat in doc)
            idf[j] = np.log((1 + s) / (1 + df)) + 1

        # --- TF-IDF ---
        tfidf = tf * idf

        # --- Normalize (L2) ---
        norm = np.linalg.norm(tfidf)
        if norm > 0:
            tfidf = tfidf / norm

        embeddings[i] = tfidf

    return embeddings, np.array(features)
