"""FontModel class."""
from typing import Callable, Dict, Tuple

import numpy as np

from font_classifier.models.base import Model
from font_classifier.datasets.rufa_dataset import RuFaDataset
from font_classifier.networks.cnn import cnn
np.random.seed(42)

class FontModel(Model):
    """FontModel class."""

    def __init__(
        self,
        dataset_cls: type = RuFaDataset,
        network_fn: Callable = cnn,
        dataset_args: Dict = None,
        network_args: Dict = None,
    ):
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)

        
    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)

        pred_raw = self.network.predict(np.expand_dims(image, 0), batch_size=1).flatten()
        ind = np.argmax(pred_raw)
        confidence_of_prediction = pred_raw[ind]
        predicted_font = self.data.mapping[ind]
        return predicted_font, confidence_of_prediction



