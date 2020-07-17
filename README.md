# Arabic Font Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mhmoodlan/arabic-font-classification/blob/master/codebase/code/notebooks/arabic_font_classification.ipynb)

## Acknowledgement

The structure and some fundamental parts of this code are adapted from [Full Stack Deep Learning (FSDL)](https://github.com/full-stack-deep-learning/fsdl-text-recognizer).

## Demo and Notebook

You can see this project in action in the [accompanied demo and post](https://mhmoodlan.github.io/blog/arabic-font-classification), or run the code in [this notebook](https://github.com/mhmoodlan/arabic-font-classification/blob/master/codebase/code/notebooks/arabic_font_classification.ipynb).

## Project Structure

The `/cloud` folder imitates storing data in the cloud. In real world settings, the dataset will be stored on a cloud storage service such as Amazon S3. The actual code lives in the `/codebase` folder. There is a clear seperation between training code (under `/codebase/training`) and everything else including models, networks, datasets, and other utilities (under `/codebase/font_classifier`). This seperation makes system deployment easier and cleaner.

As presented in the FSDL course, to version control the data, we don't check the actual images in git. Instead, a json file is created containing one entry per data instance. Each entry consists of the data instance URL (cloud storage), label, and other metadata if relevant. This json file is what gets tracked by git and therefore we can get the data at the required version by checking the corresponding git commit. As the dataset gets bigger the size of the json file gets larger, in which case git-lfs can be used. Benefits of this way of handling data:

  1. **Reproducibility:** since it is tracked by git, we can get the exact data that we used a week ago or a year ago.

  2. **Extendibility:** the dataset can be extended to incorporate new data while making sure to never use previous test set instances as training instances and vise versa.

  3. **Portability:** reduces disk space required for the project, which makes it portable over git or any other means.


## Running the Code

To run the code locally:

1. Install requirements:

    ```bash
    $ pip install -r requirements.txt
    ```

2. Fetch and extract data from [releases](https://github.com/mhmoodlan/arabic-font-classification/releases/) to /cloud folder:

    ```bash
    $ wget 'https://github.com/mhmoodlan/arabic-font-classification/releases/download/v1.0/rufa.tar.gz' -O  ./cloud/rufa.tar.gz
    $ cd /cloud && tar -xzf 'rufa.tar.gz'
    ```

3. Spin a simple server in the `/cloud` folder at http://0.0.0.0:8000/ :

    ```bash
    $ cd /cloud && python -m http.server
    ```

4. Run an experiment:

    ```bash
    $ cd /codebase/code && export PYTHONPATH=. && python training/run_experiment.py --save \
        '{"dataset": "RuFaDataset", "model": "FontModel", "network": "cnn", "train_args": {"epochs": 6, "mode": "test", "validate_mismatch": "False"}}'
    ```

    The `'mode'` config in `'train_args'` takes one of two values: `'val'` or `'test'`.

    In `'val'` mode: the model is trained and validated on synthetic data only. If `'validate_mismatch'` is set to True, further data mismatch validation is performed on a subset of the real data.

    In `'test'` mode: the model is trained on the entire synthetic data + the part of the real data used in data mismatch validation in `'val'` mode. After training, the final generalization error is reported on the remainder of the real data.

    This command should output something similar to the following:

    ```plaintext
    Epoch 1/6
    1254/1254 [==============================] - 119s 95ms/step - loss: 0.3185 - accuracy: 0.8751
    Epoch 2/6
    1254/1254 [==============================] - 40s 32ms/step - loss: 0.0539 - accuracy: 0.9918
    Epoch 3/6
    1254/1254 [==============================] - 40s 32ms/step - loss: 0.0386 - accuracy: 0.9953
    Epoch 4/6
    1254/1254 [==============================] - 40s 32ms/step - loss: 0.0270 - accuracy: 0.9976
    Epoch 5/6
    1254/1254 [==============================] - 40s 32ms/step - loss: 0.0264 - accuracy: 0.9973
    Epoch 6/6
    1254/1254 [==============================] - 40s 32ms/step - loss: 0.0246 - accuracy: 0.9979
    Training took 323.854642 s
    In test mode, mismatch data isn't validated since it's used during training.

    14/14 [==============================] - 0s 10ms/step - loss: 0.2316 - accuracy: 0.9712
    Test score: [0.2316255271434784, 0.971222996711731]
    ```
