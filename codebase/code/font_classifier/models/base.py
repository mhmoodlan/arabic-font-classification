from pathlib import Path
from typing import Callable, Dict, Optional

from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.data import Dataset
import numpy as np

DIRNAME = Path(__file__).parents[1].resolve() / "weights"

np.random.seed(42)

class Model:
  
    def __init__(
        self,
        dataset_cls: type,
        network_fn: Callable[..., KerasModel],
        dataset_args: Dict = None,
        network_args: Dict = None,
    ):
        self.name = f"{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}"

        if dataset_args is None:
            dataset_args = {}
        self.data = dataset_cls(**dataset_args)

        if network_args is None:
            network_args = {}
        self.network = network_fn(self.data.input_shape, self.data.output_shape, **network_args)

        self.network.summary()


    @property
    def image_shape(self):
        return self.data.input_shape

    @property
    def weights_filename(self) -> str:
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f"{self.name}_weights.h5")

    def fit(
        self, dataset, batch_size: int = 32, epochs: int = 10, augment_val: bool = True, callbacks: list = None,
    ):
        if callbacks is None:
            callbacks = []

        self.network.compile(loss=self.loss(), optimizer=self.optimizer(), metrics=self.metrics())

        train_ds = dataset.train_ds
        val_ds = dataset.val_ds

        self.network.fit(
            train_ds,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_ds
        )

    def evaluate(self, dataset: type):
      return self.network.evaluate(dataset)

    def loss(self):
        return BinaryCrossentropy(from_logits=False)

    def optimizer(self):
        return 'adam'

    def metrics(self):
        return ["accuracy"]

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename)