import json
import toml
from font_classifier import util
from font_classifier.datasets.dataset import Dataset, _download_raw_dataset
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
rng = tf.random.experimental.Generator.from_seed(1234)

RAW_DATA_DIRNAME = Dataset.data_dirname() / "raw" / "rufa"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
PATCHES_DIRNAME = RAW_DATA_DIRNAME / "patches"


class RuFaDataset(Dataset):
    """
    RuFa dataset class.
    """

    def __init__(self, test_mode_on: bool = False):
        self.metadata = toml.load(METADATA_FILENAME)
        if not os.path.exists(RAW_DATA_DIRNAME / self.metadata["filename"]):
            curdir = os.getcwd()
            os.chdir(RAW_DATA_DIRNAME)
            _download_raw_dataset(self.metadata)
            os.chdir(curdir)
        
        with open(RAW_DATA_DIRNAME / self.metadata["filename"]) as f:
            self.patch_data = json.load(f)
        self.data_by_patch_id = {
            id_: data for id_, data in (_extract_id_and_data(patch_datum) for patch_datum in self.patch_data['data'])
        }
        self.mapping = {0: 'farsi', 1: 'ruqaa'}
        self.num_classes = len(self.mapping)
        self.input_shape = (100, 100, 1)
        self.output_shape = (1,)

        self.test_mode_on = test_mode_on

        self.train_ds = None
        self.val_ds = None
        self.mismatch_ds = None
        self.test_ds = None

    def load_or_generate_data(self):
        if len(self.patch_filenames) < len(self.data_by_patch_id):
          self._download_patches()
        self._split_patches()
        self._process_patches()

    @property
    def patch_filenames(self):
        return list(PATCHES_DIRNAME.glob("*/*/*.jpg"))

    def _download_patches(self):
        ids, urls, sources, labels = zip(*[(id_, data["url"], data["source"], data["label"]) for id_, data in self.data_by_patch_id.items()])
        filenames = [PATCHES_DIRNAME / source / label / (str(id_)+'.jpg') for id_, source, label in zip(ids, sources, labels)]
        for source in np.unique(sources):
          for label in np.unique(labels):
            (PATCHES_DIRNAME/source/label).mkdir(exist_ok=True, parents=True)
        print('downloading data....')
        util.download_urls(urls, filenames)

    def _split_patches(self):
        _max_data_size = 2**np.int(self.patch_data['test_set_check']['max_data_size_power'])
        _test_ratio = self.patch_data['test_set_check']['test_ratio']

        synth_paths = tf.data.Dataset.list_files(str(PATCHES_DIRNAME / 'synth/*/*.jpg'), seed=42)
        real_paths = tf.data.Dataset.list_files(str(PATCHES_DIRNAME / 'real/*/*.jpg'), seed=42)

        def test_set_check(item):
            id = tf.strings.split(tf.strings.split(item, os.sep)[-1], '.')[0]
            hash = tf.strings.to_hash_bucket_fast(id, _max_data_size)
            return tf.cast(hash, tf.float64) < float(_test_ratio) * _max_data_size
        
        def train_set_check(item):
            id = tf.strings.split(tf.strings.split(item, os.sep)[-1], '.')[0]
            hash = tf.strings.to_hash_bucket_fast(id, _max_data_size)
            return tf.cast(hash, tf.float64) >= float(_test_ratio) * _max_data_size

        self.train_paths = synth_paths.filter(train_set_check)
        self.val_paths = synth_paths.filter(test_set_check)
        self.mismatch_paths = real_paths.filter(test_set_check)
        self.test_paths = real_paths.filter(train_set_check)

    def _process_patches(self):
      AUTOTUNE = tf.data.experimental.AUTOTUNE
      BATCH_SIZE = 32

      def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
        if cache:
          if isinstance(cache, str):
            ds = ds.cache(cache)
          else:
            ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=42)
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

      def parse_image(data_instance):
        parts = tf.strings.split(data_instance, os.sep)
        label = tf.cast(tf.argmax(tf.cast(parts[-2] == np.array(list(self.mapping.values())), dtype=tf.float16)), tf.float16)

        image = tf.io.read_file(data_instance)
        image = tf.image.decode_jpeg(image, 1)
        image = tf.image.convert_image_dtype(image, tf.float32)

        if parts[-3] == 'synth':
          noise = rng.normal(shape=tf.shape(image), mean=0.0, stddev=0.015, dtype=tf.float32)
          image = tf.add( image, noise)
          image = tf.clip_by_value(image, 0.0, 1.0)

          image = tf.image.adjust_jpeg_quality(image, 90)

        return image, label

      self.train_ds = self.train_paths.map(parse_image)
      self.val_ds = self.val_paths.map(parse_image)
      self.mismatch_ds = self.mismatch_paths.map(parse_image)
      if self.test_mode_on:
        self.train_ds = self.train_ds.concatenate(self.val_ds.concatenate(self.mismatch_ds))
        self.val_ds = None
      self.test_ds = self.test_paths.map(parse_image)

      self.train_ds = prepare_for_training(self.train_ds)
      if not self.test_mode_on:
        self.val_ds = prepare_for_training(self.val_ds)
      self.mismatch_ds = prepare_for_training(self.mismatch_ds)
      self.test_ds = prepare_for_training(self.test_ds)
      
    def __repr__(self):
        return (
            "RuFa Real Dataset\n"
            f"Num classes: {self.num_classes}\n"
            f"Mapping: {self.mapping}\n"
            f"Input shape: {self.input_shape}\n"
        )


def _extract_id_and_data(patch_datum):
    """
    patch_datum is of the form
        {
            "url": "url/to/image.jpg",
            "label": "farsi",
            "source": "real"
        }
    """
    url = patch_datum["url"]
    id_ = url.split(os.sep)[-1].split('.')[0] 
    label = patch_datum["label"]
    source = patch_datum["source"]

    return id_, {"url": url, "label": label, "source": source}