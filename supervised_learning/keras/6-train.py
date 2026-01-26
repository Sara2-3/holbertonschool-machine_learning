#!/usr/bin/env python3
"""
Module that trains a Keras model with optional validation data
and early stopping.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    Trains a Keras model using mini-batch gradient descent.

    Parameters
    ----------
    network : K.Model
        The model to train.
    data : numpy.ndarray
        Input data of shape (m, nx).
    labels : numpy.ndarray
        One-hot labels of shape (m, classes).
    batch_size : int
        Size of the batch used for mini-batch gradient descent.
    epochs : int
        Number of passes through the data.
    validation_data : tuple, optional
        Data to validate the model with, as (X_valid, Y_valid).
    early_stopping : bool, optional
        If True, use early stopping based on validation loss.
    patience : int, optional
        Number of epochs with no improvement after which training will stop.
    verbose : bool, optional
        If True, output is printed during training.
    shuffle : bool, optional
        If True, shuffle the batches every epoch.

    Returns
    -------
    History
        The History object generated after training the model.
    """
    callbacks = []
    if early_stopping and validation_data is not None:
        es = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(es)

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )
    return history
