#!/usr/bin/env python3
"""Bag of Words embedding matrix"""

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use (if None, use all words)

    Returns:
        embeddings: numpy.ndarray of shape (s, f)
        features: list of features used for embeddings
    """
    tokenized = []
    for sentence in sentences:
        clean = sentence.lower()
        # remove possessives e.g. children's
        clean = re.sub(r"'s\b", "", clean)
        clean = re.sub(r"[^a-zA-Z ]", " ", clean)
        words = clean.split()
        tokenized.append(words)

    if vocab is None:
        all_words = set()
        for words in tokenized:
            all_words.update(words)
        features = sorted(list(all_words))
    else:
        features = list(vocab)

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    feature_index = {word: i for i, word in enumerate(features)}

    for i, words in enumerate(tokenized):
        for word in words:
            if word in feature_index:
                embeddings[i][feature_index[word]] += 1

    return embeddings, np.array(features)
