# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from tqdm import tqdm
import torch
import numpy as np

import time
import argparse
import keras
import os
import sys
import math
import collections
import logging
import tensorflow as tf
from collections.abc import Mapping, Sequence
from deepctr_torch.layers import activation_layer
from itertools import islice


class Timer:
    def __init__(self, message="Execution time"):
        self.message = message  # Custom message to be printed with the execution time

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        tqdm.write(f"{self.message:10}: {self.elapsed:.7f} seconds")


logger = logging.getLogger(__name__)

from tensorflow.python.feature_column import utils as fc_utils

result_dir = "/tmp/tianchi/result/DeepFM/"
result_path = result_dir + "result"
global_time_cost = 0
global_auc = 0

# Definition of some constants
CONTINUOUS_COLUMNS = ["I" + str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ["C" + str(i) for i in range(1, 27)]  # 1-26 inclusive
LABEL_COLUMN = ["clicked"]
TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

HASH_BUCKET_SIZE = 10000
EMBEDDING_DIMENSIONS = 16

DEEP_COLUMNS = []
WIDE_COLUMNS = []
FM_COLUMNS = []

for column_name in FEATURE_COLUMNS:
    if column_name in CATEGORICAL_COLUMNS:
        DEEP_COLUMNS.append(column_name)
        FM_COLUMNS.append(column_name)
        WIDE_COLUMNS.append(column_name)
    else:
        WIDE_COLUMNS.append(column_name)
        DEEP_COLUMNS.append(column_name)


# generate dataset pipline
def build_model_input(filename, batch_size, num_epochs):
    def parse_csv(value):
        logger.info("Parsing {}".format(filename))
        cont_defaults = [[0.0] for _ in range(1, 14)]
        cate_defaults = [[" "] for _ in range(1, 27)]
        label_defaults = [[0]]
        column_headers = TRAIN_DATA_COLUMNS
        record_defaults = label_defaults + cont_defaults + cate_defaults
        columns = tf.io.decode_csv(value, record_defaults=record_defaults)
        all_columns = collections.OrderedDict(zip(column_headers, columns))
        labels = all_columns.pop(LABEL_COLUMN[0])
        features = all_columns
        return features, labels

    files = filename
    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(files)
    dataset = dataset.shuffle(
        buffer_size=20000, seed=args.seed
    )  # fix seed for reproducing
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_csv, num_parallel_calls=28)
    dataset = dataset.prefetch(2)
    return dataset


def preprocess_features(features):
    processed = {}
    for column_name in FEATURE_COLUMNS:
        if column_name in CATEGORICAL_COLUMNS:
            hashed = (
                keras.layers.Hashing(
                    num_bins=HASH_BUCKET_SIZE - 1,
                    output_mode="int",
                )(features[column_name])
                + 1
            )
            processed[column_name] = hashed
        else:
            processed[column_name] = features[column_name]

    return processed


class FM(torch.nn.Module):
    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1)
        cross_term = 0.5 * (square_of_sum - sum_of_square)

        return cross_term


class DNN(torch.nn.Module):
    def __init__(self, hidden_units):
        super(DNN, self).__init__()
        self._layers = torch.nn.ModuleList()
        for i in range(len(hidden_units) - 1):
            self._layers.append(torch.nn.Linear(hidden_units[i], hidden_units[i + 1]))
            if hidden_units[i + 1] != 1:
                self._layers.append(torch.nn.ReLU())
                self._layers.append(torch.nn.BatchNorm1d(hidden_units[i + 1]))

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class Linear(torch.nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, inputs):
        linear_input = inputs

        linear_logit = torch.sum(linear_input, dim=1, keepdim=True)

        return linear_logit


class DeepFM(torch.nn.Module):
    def __init__(self):
        super(DeepFM, self).__init__()
        self._embeddings = self._get_embedding_tables()
        self._fm = FM()
        self._dnn = DNN([429, 1024, 256, 32])
        self._linear = Linear()
        self._final_dnn = DNN([49, 128, 64, 1])

    def _get_embedding_tables(self):
        embeddings = {}

        for column_name in FEATURE_COLUMNS:
            if column_name in CATEGORICAL_COLUMNS:
                emb = torch.nn.Embedding(
                    HASH_BUCKET_SIZE, EMBEDDING_DIMENSIONS, padding_idx=0
                )
                embeddings[column_name] = emb

        return torch.nn.ModuleDict(embeddings)

    def forward(self, features):
        with Timer("emb"):
            for k, emb in self._embeddings.items():
                features[k] = emb(features[k])

        with Timer("dnn_input"):
            dnn_input = torch.cat(
                [features[col].reshape(features[col].shape[0], -1) for col in DEEP_COLUMNS],
                dim=1,
            )
        with Timer("wide_input"):
            wide_input = torch.cat(
                [features[col].reshape(features[col].shape[0], -1) for col in WIDE_COLUMNS],
                dim=1,
            )
        with Timer("fm_input"):
            fm_input = torch.stack([features[col] for col in FM_COLUMNS], dim=1)

        with Timer("dnn"):
            dnn_output = self._dnn(dnn_input)
        with Timer("linear"):
            linear_output = self._linear(wide_input)
        with Timer("fm"):
            fm_output = self._fm(fm_input)

        all_input = torch.cat([dnn_output, linear_output, fm_output], dim=1)
        with Timer("final dnn"):
            dnn_logits = self._final_dnn(all_input)
        prob = torch.sigmoid(dnn_logits).flatten()

        return prob


def tensorflow_to_pytorch(obj) -> torch.Tensor | Mapping | Sequence:
    if isinstance(obj, tf.RaggedTensor):
        return torch.from_numpy(obj.to_tensor().numpy())
    elif tf.is_tensor(obj):
        return torch.from_numpy(obj.numpy())
    elif isinstance(obj, Mapping):
        return {key: tensorflow_to_pytorch(value) for key, value in obj.items()}
    elif isinstance(obj, Sequence):
        return [tensorflow_to_pytorch(item) for item in obj]
    else:
        return obj


def main():
    # check dataset
    print("Checking dataset...")
    train_file = os.path.join(args.data_location, "train.csv")
    test_file = os.path.join(args.data_location, "eval.csv")
    if (not os.path.exists(train_file)) or (not os.path.exists(test_file)):
        print("Dataset does not exist in the given data_location.")
        sys.exit()
    no_of_training_examples = sum(1 for line in open(train_file))
    no_of_test_examples = sum(1 for line in open(test_file))
    print("Numbers of training dataset is {}".format(no_of_training_examples))
    print("Numbers of test dataset is {}".format(no_of_test_examples))

    # set batch size, eporch & steps
    batch_size = args.batch_size

    if args.steps == 0:
        no_of_epochs = 1
        train_steps = math.ceil(
            (float(no_of_epochs) * no_of_training_examples) / batch_size
        )
    else:
        no_of_epochs = math.ceil(
            (float(batch_size) * args.steps) / no_of_training_examples
        )
        train_steps = args.steps
    test_steps = math.ceil(float(no_of_test_examples) / batch_size)
    print("The training steps is {}".format(train_steps))
    print("The testing steps is {}".format(test_steps))

    # set fixed random seed
    tf.random.set_seed(args.seed)
    torch.manual_seed(args.seed)

    # set directory path
    # model_dir = os.path.join(args.output_dir, "model_DeepFM_" + str(int(time.time())))
    # checkpoint_dir = args.checkpoint if args.checkpoint else model_dir
    # print("Saving model checkpoints to " + checkpoint_dir)

    # create data pipline of train & test dataset
    train_dataset = build_model_input(train_file, batch_size, no_of_epochs)
    test_dataset = build_model_input(test_file, batch_size, 1)

    model = DeepFM()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    time_start = time.perf_counter()
    with tqdm(total=train_steps, desc=f"", unit="it") as pbar:
        start_time = time.perf_counter()
        for step, (features, label) in islice(enumerate(train_dataset), train_steps):
            features = tensorflow_to_pytorch(preprocess_features(features))
            label = tensorflow_to_pytorch(label).float()

            # Forward
            with Timer('fwd'):
                pred = model(features)
            with Timer('loss'):
                loss = loss_fn(pred, label)

            # Backward
            model.zero_grad()
            with Timer('backward'):
                loss.backward()

            # Optmizer step
            with Timer('optimizer'):
                optimizer.step()

            tqdm.write("=" * 50)
            pbar.update(1)
            if (step + 1) % 100 == 0:
                elapsed_time = time.perf_counter() - start_time
                tqdm.write(
                    f"loss = {loss.detach().numpy():.8f}, steps = {step+1}/{train_steps} ({elapsed_time:.3f} sec)"
                )
                start_time = time.perf_counter()  # Reset the start time
    time_end = time.perf_counter()
    print("Training completed.")
    time_cost = time_end - time_start
    global global_time_cost
    global_time_cost = time_cost

    os.makedirs(result_dir, exist_ok=True)
    print(global_auc)
    print(global_time_cost)
    with open(result_path, "w") as f:
        f.write(str(global_time_cost) + "\n")
        f.write(str(global_auc) + "\n")


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return low_string == "true"


# Get parse
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_location",
        help="Full path of train data",
        required=False,
        default="./data",
    )
    parser.add_argument(
        "--steps", help="set the number of steps on train dataset", type=int, default=0
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size to train. Default is 512",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--seed", help="set the random seed for tensorflow", type=int, default=2021
    )
    parser.add_argument(
        "--learning_rate",
        help="Learning rate for deep model",
        type=float,
        default=0.001,
    )

    return parser


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    main()
