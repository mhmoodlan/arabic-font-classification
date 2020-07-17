"""FontPredictor class"""
from typing import Tuple, Union

import numpy as np

from font_classifier.models import FontModel
import font_classifier.util as util


class FontPredictor:
    """Given an image of text, classify its font."""

    def __init__(self):
        self.model = FontModel()
        self.model.load_weights()

    def predict(self, image_or_filename: Union[np.ndarray, str]) -> Tuple[str, float]:
        """Predict on a single image."""
        if isinstance(image_or_filename, str):
            image = util.read_image(image_or_filename, grayscale=True)
        else:
            image = image_or_filename
        return self.model.predict_on_image(image)

    def evaluate(self, dataset):
        """Evaluate on a dataset."""
        return self.model.evaluate(dataset)
