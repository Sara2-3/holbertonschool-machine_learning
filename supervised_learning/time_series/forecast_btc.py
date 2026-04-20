#!/usr/bin/env python3
"""
Module for training and evaluating a GRU model for BTC forecasting.

This module loads preprocessed data, builds a tf.data.Dataset,
trains a stacked GRU model and evaluates it using MSE loss.
"""
import numpy as np
import tensorflow as tf


def load_data(path='data/preprocessed.npz'):
    """
    Load preprocessed data from a .npz file.

    Args:
        path (str): Path to the .npz file.

    Returns:
        tuple: X_train, y_train, X_val, y_val, X_test, y_test arrays.
    """
    data = np.load(path)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']

    print("Train:      {:>10,} samples".format(len(X_train)))
    print("Validation: {:>10,} samples".format(len(X_val)))
    print("Test:       {:>10,} samples".format(len(X_test)))

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_dataset(X_train, y_train, X_val, y_val,
                  X_test, y_test, batch_size=64):
    """
    Build tf.data.Dataset pipelines for train, val and test.

    Args:
        X_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation input data.
        y_val (np.ndarray): Validation labels.
        X_test (np.ndarray): Test input data.
        y_test (np.ndarray): Test labels.
        batch_size (int): Batch size for training.

    Returns:
        tuple: train_dataset, val_dataset, test_dataset.
    """
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)
    )
    train_dataset = train_dataset.shuffle(1000).batch(
        batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (X_val, y_val)
    )
    val_dataset = val_dataset.batch(
        batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test)
    )
    test_dataset = test_dataset.batch(
        batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset


def build_model(window_size=24):
    """
    Build a stacked GRU model for BTC price forecasting.

    Args:
        window_size (int): Number of timesteps in each input window.

    Returns:
        tf.keras.Sequential: Compiled GRU model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(64, return_sequences=True,
                            input_shape=(window_size, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    model.summary()
    return model


def train_model(model, train_dataset, val_dataset):
    """
    Train the GRU model with early stopping.

    Args:
        model (tf.keras.Sequential): Compiled GRU model.
        train_dataset (tf.data.Dataset): Training dataset.
        val_dataset (tf.data.Dataset): Validation dataset.

    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=val_dataset,
        callbacks=[early_stop],
        verbose=1
    )

    return history


def evaluate_model(model, test_dataset):
    """
    Evaluate the trained model on the test set.

    Args:
        model (tf.keras.Sequential): Trained GRU model.
        test_dataset (tf.data.Dataset): Test dataset.

    Returns:
        tuple: test_loss (MSE) and test_mae values.
    """
    test_loss, test_mae = model.evaluate(test_dataset, verbose=0)
    print("\nTest MSE: {:.6f}".format(test_loss))
    print("Test MAE: {:.6f}".format(test_mae))
    return test_loss, test_mae


if __name__ == '__main__':
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    print("\nBuilding datasets...")
    train_dataset, val_dataset, test_dataset = build_dataset(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    print("\nBuilding model...")
    model = build_model(window_size=24)

    print("\nTraining...")
    train_model(model, train_dataset, val_dataset)

    print("\nEvaluating...")
    evaluate_model(model, test_dataset)

    model.save('data/btc_gru_model.keras')
    print("\nModel saved -> data/btc_gru_model.keras")
