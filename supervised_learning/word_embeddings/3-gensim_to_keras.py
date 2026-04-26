#!/usr/bin/env python3
"""Convert a gensim Word2Vec model to a Keras Embedding layer"""


def gensim_to_keras(model):
    """
    Converts a gensim Word2Vec model to a Keras Embedding layer.

    Args:
        model: trained gensim Word2Vec model

    Returns:
        trainable Keras Embedding layer
    """
    return model.wv.get_keras_embedding(train_embeddings=True)
