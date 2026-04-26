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
    # Tokenize: lowercase, remove punctuation, split into words
    tokenized = []
    for sentence in sentences:
        # lowercase, remove possessives, then remove non-alpha characters
        clean = sentence.lower()
        clean = re.sub(r"'s\b", "", clean)   # remove possessives e.g. children's
        clean = re.sub(r"[^a-zA-Z ]", " ", clean)
        words = clean.split()
        tokenized.append(words)

    # Build vocabulary if not provided
    if vocab is None:
        all_words = set()
        for words in tokenized:
            all_words.update(words)
        features = sorted(list(all_words))
    else:
        features = list(vocab)

    # Build embedding matrix
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # Map feature to index for fast lookup
    feature_index = {word: i for i, word in enumerate(features)}

    for i, words in enumerate(tokenized):
        for word in words:
            if word in feature_index:
                embeddings[i][feature_index[word]] += 1

    return embeddings, features
