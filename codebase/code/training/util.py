"""Function to train a model."""
from time import time

from tensorflow.keras.callbacks import EarlyStopping, Callback

from font_classifier.datasets.dataset import Dataset
from font_classifier.models.base import Model

import tensorflow as tf
tf.random.set_seed(42)

EARLY_STOPPING = True

def train_model(model: Model, dataset: Dataset, epochs: int, batch_size: int) -> Model:
    """Train model."""
    callbacks = []

    if EARLY_STOPPING:
        early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=3, verbose=1, mode="auto")
        callbacks.append(early_stopping)

    t = time()
    _history = model.fit(dataset=dataset, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    print("Training took {:2f} s".format(time() - t))

    return model