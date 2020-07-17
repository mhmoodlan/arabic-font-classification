from typing import Tuple
from tensorflow.keras.models import Model
import tensorflow as tf

tf.random.set_seed(42)


def cnn(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:

    num_classes = output_shape[0]
    dropout_seed = 708090
    kernel_seed = 42
  
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=input_shape, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seed)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Dropout(0.1, seed=dropout_seed),
      tf.keras.layers.Conv2D(32, 5, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seed)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Dropout(0.1, seed=dropout_seed),
      tf.keras.layers.Conv2D(64, 10, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seed)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Dropout(0.1, seed=dropout_seed),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seed)),
      tf.keras.layers.Dropout(0.2, seed=dropout_seed),
      tf.keras.layers.Dense(16, activation='relu', kernel_regularizer='l2', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seed)),
      tf.keras.layers.Dropout(0.2, seed=dropout_seed),
      tf.keras.layers.Dense(num_classes, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seed))
    ])

    return model
